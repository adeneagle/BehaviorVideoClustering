# Distilled keypoint model (part network + SLEAP keypoints)

One ResNet34 U-Net (~25 M parameters) that gives, from each 768 px mouse-centred crop, in a single
pass: the 7 part masks (body, 4 paws, tail) **and** the 6 SLEAP keypoints (snout, left/right front
paw, left/right hind paw, tail base). The keypoints were distilled from the lab's SLEAP model, which
is ~200 M parameters and ~100x slower. Made 2026-10-19; the packaged default since then.

## Where it is

| What | Path |
|---|---|
| Packaged model (default `seg_model.WEIGHTS`) | `Data/Behavior/PartSeg/models/partseg_20261019_185549_resnet34_kp.pt` |
| Its training run (logs, report, evals) | `Data/Behavior/PartSeg/video2026-04-02T12_54_39_first_20/seg_runs/20261019_185549_resnet34/` |
| TensorRT engines (fp16, RTX 3090 / sm86, TRT 10.16) | `Data/Behavior/PartSeg/models/trt_engines/partseg_b42fbc327872_*.engine`, `loc2_d67c0c086b4e_*.engine` (localizer) |
| Code | `Code/BehaviorVideoClustering/`: `seg_model.py` (network, `decode_keypoints`), `run_video.py`, `trt_engines.py` |

Engines are rebuilt automatically (~3 min, once) for a different GPU, TensorRT version or weights.

## Use

```bash
conda activate sam3; cd ~/Aden/Code/BehaviorVideoClustering
python run_video.py VIDEO.mkv --out-dir <PartSeg>/<session>/full_video/partseg_20261019_185549_resnet34_kp
```

`frames.csv` gets, per frame and node: `kp_<node>_x`, `kp_<node>_y` (full-resolution image px) and
`kp_<node>_score` (heatmap peak, 0-1). **Threshold ~0.2, not SLEAP's 0.5**: the scores run lower;
0.15-0.3 reproduces SLEAP's presence rate. The masks (`labels/`) and the rest of `frames.csv` are as
before. In Python: `net(x, keypoints=True)` -> `(seg logits, keypoint logits)`, then
`seg_model.decode_keypoints()`; `net(x)` alone still returns only the segmentation.

Run long jobs as their own memory-capped unit, not from the Claude desktop app's shell (an
out-of-memory kill inside the app's scope takes the whole app down):

```bash
systemd-run --user --unit=pose -p MemoryMax=12G -p MemorySwapMax=0 \
  -p StandardOutput=file:pose.log -p StandardError=file:pose.log \
  ~/miniconda3/envs/sam3/bin/python -u run_video.py VIDEO.mkv --out-dir OUT
```

## Speed (RTX 3090, TensorRT fp16)

| | |
|---|---|
| Network alone, batch 64 | 1083 crops/s (PyTorch bf16: 366); same as the segmentation-only network |
| `run_video.py` end to end (decode, localizer, crop, network, keypoint decode, writing) | 451 fps with the preview video, 555 without |
| A 3.4 h, 60 fps session | ~25 min (the SLEAP model: ~2.5 frames/s here, i.e. days) |

## Accuracy

**Held-out human labels** (videos 2026-03-12 and 2026-06-01, never trained on; 429 frames):
82-95% of points within 10 px; snout 82% (median 4.2 px), paws median 2.8-3.4 px; paw label correct
at the paw keypoints 98% (the previous part network: ~95%); part IoU vs SAM 0.867.

**60 hand-clicked grooming frames** (Leonidas 10-08, a session neither model saw; within 20 px of
the click, paired bootstrap 95% CI):

| Node | Distilled | SLEAP | Difference |
|---|---|---|---|
| Snout | 73% | 77% | -3% [-13, +7]: not distinguishable |
| Left front paw | 87% | 75% | +12% [+2, +22] |
| Left hind paw | 98% | 70% | +28% [+17, +40] |
| Right front paw | 62% | 72% | -10% [-20, 0] |
| Right hind paw | 32% | 28% | mostly hidden in these poses |
| Tail base | 35% | 47% | the clicks sit ~17 px inside the body vs both models (convention) |

When confident, the distilled snout is closer to the click than SLEAP's (median 4.4 vs 7.3 px) and
never >30 px off (SLEAP: 2%).

**Against SLEAP on 10-08** (1441 frames): median distance snout 5.9 px, paws 1.8-3.2 px, tail base
9.2 px. Whole-session checks (10-06, 10-08): confident paws land on their own mask in 400/400 random
frames; left/right swaps ~0.1% of frames.

## How it was made

### Why not just shrink SLEAP
The SLEAP model (1405 labels; ConvNeXt-T centroid 87.5 M + Swin-S instance 108.6 M, most of it in
the decoders) runs ~2.5 frames/s here. Its own history shows size was not what made it good: on its
114-frame validation set, Swin-T vs Swin-S with ImageNet pretraining differ by -0.008 PCK@10 (95% CI
+-0.016), while pretraining itself was worth ~+0.05; ~80% of its compute is the centroid stage,
which this pipeline's localizer already does. So its keypoints were distilled into the existing
part network instead.

### Steps
1. **Teacher labels** (`sleap_teacher.py`, SLEAP env, ~1 h 45 min): the full SLEAP pipeline on 900
   random 20-frame runs of Leonidas 10-06 (17,280 frames; frame-accurate seeks checked against the
   masks), saved in resumable parts in `<session>/sleap/teacher/`.
2. **Crops** (`distill_frames.py`, ~10 min): each frame cut exactly as `run_video.py` cuts it for the
   network (768 px at its crop offset), keypoints in crop px with SLEAP's per-node score ->
   `PartSeg/distill_Leonidas_20261006_155113/` (17,068 frames with an instance).
3. **First student** (`train_seg.py --kp-distill ... --init models/partseg_20261015_014832_resnet34.pt`,
   24,000 iterations, ~52 min): a 6-channel heatmap head (3x3 conv-BN-ReLU + 1x1) on the last decoder
   features of the 7-class part network. Training samples: 40% teacher frames, the rest SAM-masked
   clips and the 969 hand-labelled SLEAP frames from 4 videos. Targets: human points (a node marked
   not visible = known absent) and teacher nodes scored >= 0.5 (the rest unknown, as are all
   clip frames); CenterNet focal loss on Gaussians (sigma 2.5 px at 512), weight 1; flips swap
   left/right. Result: snout gross errors on grooming frames (13% of confident snouts >30 px off).
4. **Hard frames** (`sleap_teacher.py --frames-json hard_frames.json --name teacher_hard`): 6,314
   more teacher frames from 266 grooming-like (slow body, fast forepaws) or curled (tail base not
   found) 2 s blocks of 10-06.
5. **Final model** (`train_seg.py`, both teacher sets, `--init` the first student, 12,000
   iterations, ~25 min): grooming snout gross errors 13% -> 0%; nothing else got worse.
   Steps 4-5 are one script, `PartSeg/distill_hard.sh`.

Evaluation tools: `eval_distill.py` (student vs SLEAP on a session's SLEAP predictions),
`sleap_click_project.py` (blind SLEAP labelling project from chosen frames) and `score_clicks.py`
(both models vs hand clicks). 10-08 was kept out of all training as the test session.

## Limitations and next steps

- Trained on one Rig2 animal (Leonidas) plus the hand-labelled sessions; check a new animal or rig
  with a few `eval_distill.py` / click frames before trusting it.
- Right front paw and tail base are the weakest points. The same loop fixes a weak point: mine
  frames where it is uncertain, label them with SLEAP, fine-tune (~1 h, unattended).
- Tail base: decide which convention you want (yours sits ~17 px further inside the body).
- The SLEAP models were trained at max width 2148 px; Rig2 frames are 2248 px, so sleap-nn rescales
  and pads every frame. This affected the teacher labels and any SLEAP inference on Rig2.

## Reviewing in SLEAP

`kp_to_slp.py RUN_DIR` (SLEAP env) writes `RUN_DIR/predictions.slp`: every frame's keypoints as a
SLEAP prediction (nodes scored < 0.2 left out, scores kept), plus 200 random mouse-in-view frames as
Labeling Suggestions. Open with `sleap-label RUN_DIR/predictions.slp`. Done for every Leonidas,
Pericles and Archimedes session (2026-10-21; ~1 min per 1 M-frame session).
