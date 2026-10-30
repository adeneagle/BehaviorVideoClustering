# BehaviorVideoClustering: SAM3 anatomical part-mask pilot

Question: can visually prompted SAM3 tracking give anatomically consistent,
temporally stable part masks (body = head + trunk, 4 paws) over whole
behaviour clips?

Run everything with the `sam3` conda env:

```bash
conda activate sam3
cd /home/aeagle/Aden/Code/BehaviorVideoClustering
```

1. `python extract_clips.py --video V.mkv --labels L.csv`
   picks 2 clips per labelled behaviour (6 s, centred on the longest bouts,
   >= 60 s apart) and writes 768 px mouse-centred crops to
   `Data/Behavior/PartSeg/<video>/clips/`. Crop offsets are kept in
   `meta.json`.
2. `python annotate.py` clicks prompts per part. You can go clip to clip with
   `n` and only click (fast), or press `p` to propagate and correct inside
   the GUI.
3. `python propagate.py` replays `prompts.json` for each clip that has no
   masks yet, then writes `masks.npz`, `overlay.mp4`, `parts.mp4` and
   `contact_sheet.jpg`.
4. Watch `overlay.mp4`. Fix clips in `annotate.py` (add correction clicks,
   then `p`), then use review mode (`r`) to flag wrong parts on every 30th
   frame.
5. `python evaluate.py` writes the metrics, L/R swap check and per-clip
   plots to `<video>/eval/`.

## Fast segmentation network (trained on the SAM masks)

6. `python train_seg.py` trains a ResNet34 U-Net (24000 iterations, ~40 min;
   ~320 crops/s in PyTorch, ~1090 in TensorRT). Clips are split whole into train / validation, and frames you
   flagged in review are left out. `report.txt` has per-part IoU against
   SAM on the held-out clips, plus the L/R swap rate. Use `--val-clips none`
   to train the final model on every clip.
7. `python predict_seg.py <video>/seg_runs/<run>` writes `masks_<run>.npz`
   and videos for every clip, labelled or not.
   `python evaluate.py --masks masks_<run>.npz` scores them like SAM's masks.

8. `python run_video.py VIDEO.mkv --labels L.csv` runs the packaged models
   over a whole video in one pass (`<video>/full_video/<model>/`).

## Whole-video inference (TensorRT, frames on the GPU)

`run_video.py` follows the lab's `Aden/code/video_processing/label_video.py`:

- **Models** live in `PartSeg/models/`: `loc2_final.pt` (the SAM2-distilled
  body localizer, from `PaulData/paul_all/models/` on the lab server) and
  `partseg_20261012_202544_resnet34.pt` (the final part network; the default
  is `seg_model.WEIGHTS`). `--model` takes another checkpoint or a
  `seg_runs/<run>` directory.
- **TensorRT** (`trt_engines.py`): both networks as FP16 engines, built on
  first use (~4 min) and cached in `PartSeg/models/trt_engines/`. The file
  name carries the weights' hash, the GPU architecture and the TensorRT
  version, so a new model or another machine rebuilds by itself.
  `--backend torch` is the PyTorch path.
- **Decoding** (`nvdec_reader.py`): NVDEC or cv2, both delivering the luma
  plane to the GPU, where the crop and both resizes happen. Both give the
  pixels of a plain `cv2.VideoCapture` read (checked on all 71317 frames of
  the first video). `--decoder auto` times both on the first frames and takes
  the faster one; the default is cv2, because on these .mkv files the NVDEC
  probe's time grows with the file (3 min idle for 9 GB, ~20 min for 90 GB,
  read over the network) and cv2 won every time anyway.
- **One pass**: the crop follows the localizer through the same smoothing
  that cut the training clips; the part network runs 40 frames behind the
  decoder instead of in a second pass.

On the RTX 3090 (2248x2064 video): 472 fps with the preview, 518 without;
the old two-pass PyTorch script did 200 fps in its second pass alone. The
part network is the limit (1090 crops/s in TensorRT, 320 in PyTorch). NVDEC
loses here: its one decode engine does 385 fps on these frames and drops
further while the same GPU runs the networks, and cv2 does ~680 fps without
the BGR conversion. Expect NVDEC to win on newer cards.

```bash
python trt_engines.py validate VIDEO   # TensorRT vs fp32 PyTorch on real frames
python trt_engines.py bench            # networks alone, PyTorch vs TensorRT
python nvdec_reader.py validate VIDEO  # every frame of both decoders vs cv2
python nvdec_reader.py bench VIDEO
```

Extra packages in the `sam3` env (numpy stays at 1.26):
`pip install tensorrt-cu12==10.16.1.11 onnx onnxscript PyNvVideoCodec==2.2.3`.
The lab server's `video_processing/.venv` already has them; there, pass
`--model`, `--loc` and `--out-dir` (the defaults are local paths).

`extract_clips.py` still follows the mouse by background subtraction.

## Behaviour embeddings (V-JEPA 2 on the part maps)

After `run_video.py`, with `R = <video>/full_video/<model>`:

9. `python parts_video.py R` writes `R/parts.mp4`: the label maps as solid
   part colours on black, 256 px. This replaces the camera video from here on.
10. `python embed_vjepa.py R` embeds that representation with V-JEPA 2
    (ViT-L, `facebook/vjepa2-vitl-fpc64-256`): 16 frames at 15 fps (~1 s),
    one window every 0.5 s, mean of the last-layer tokens. ~26 windows/s on
    the 3090. By default (`--canon window`) every window is rotated by ONE
    angle so the mouse faces up: heading is gone between windows, turning
    within a window stays visible. The heading is tail base -> body centroid
    (`python axis.py R` caches it per frame in `R/axis.npz`; needs the tail
    model), else the paw axis. Body-frame forward / lateral speed and turn
    rate are saved per window, since the crop removes translation. Output
    `R/vjepa_window/features.npz` (`--canon none`: `R/vjepa/`); `--sheet`
    writes `canon_examples.jpg`, frames as the encoder sees them.
11. `python cluster_embeddings.py R [--emb vjepa_window]`: PCA 50 -> UMAP
    10-d -> HDBSCAN (`--select leaf` by default; `eom` lumps the active
    cloud into one cluster once heading no longer splits it). Writes
    `R/<emb>/clusters/` (summary.txt, overview.png, contact_sheet.jpg,
    examples/cluster_<k>.mp4 from preview.mp4).
12. `python label_video.py R [--start-min A --end-min B]`: preview.mp4 with
    a third panel (the part map as encoded, rotated), the cluster of the
    nearest window with its forward speed and turn rate, and a +-30 s
    ethogram strip; H.264 in `R/<emb>/clusters/labeled.mp4` (~4 min for 90 min).

First run, `Archimedes_20260924_115808` (90 min, no behaviour labels), tail
model: heading is linearly decodable from the features with R^2 0.76
without canonicalisation and 0.04 with it; the share of clustered windows in
clusters that face one direction (R > 0.8) went from 87% to 31%. Leaf
clusters separate fast straight runs, turns in each direction and rest
postures, but leave ~48% of windows as noise.

`embed_vjepa.py` options added 2026-10-16: `--layers 12,16,20` also saves the
token mean of those encoder blocks (`feats_L<l>`); `--out-name`; `--input
parts|raw|masked[,...]` (raw = the camera crop, canonicalised the same way,
cut 1088 px wide so rotation fills the corners; masked = raw outside the
dilated part mask set to black; several inputs share one decoding pass and go
to `<out-name>_<input>/`); `--max-windows N` for quick checks.

`--backbone vjepa2_1-l` uses V-JEPA 2.1 ViT-L/16 (Meta, March 2026; 384 px,
distilled from ViT-G; trained with dense and deep self-supervision). Not in
`transformers`: the model code is the official repo, cloned at
`Aden/Code/third_party/vjepa2` (commit 204698b), and the weights are Meta's
`dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt` (5.2 GB, with
training state) reduced to the EMA encoder in
`PartSeg/models/vjepa2_1/vjepa2_1_vitl_encoder_only.pt`. `torch.hub.load`
does not work at that commit (`src/hub/backbones.py` sets the download URL to
`http://localhost:8300`). Its blocks 6 / 12 / 18 / 24 have their own trained
output norms; `--layers` takes a subset of 6,12,18 (24 is `feats`). Runs
under fp16 autocast (its RoPE is fp32), ~9 windows/s at 384 px on the 3090.

## Neural validation of the clusters (Rig2 sessions with spikes)

For sessions recorded on Rig2 with Neuropixels (`RealtimeData/Rig2/<animal>/
<session>/seg01`, aligned with `HsuPaper/scripts/align_rig2.py`, units from
`kilosort4/phy/cluster_rf.tsv`), spikes and frames come from
`HsuPaper/scripts/bonsai_loader.BonsaiSession` (ephys seconds; decoded frame
i -> alignment fit). Each window counts spikes over exactly the frames V-JEPA
saw. Units: `rf_label == good` by default (`--quality`).

- `neural_decode.py R --segment SEG`: XGBoost (GPU) decoding of the clusters
  from spikes, per area, from kinematics, kinematics + spikes, a spike lag scan;
  blocked 10-fold CV with 10 s buffers; chance from circular label shifts.
  `R/<emb>/neural/summary.txt`.
- `sweep_neural.py`: decodability of HDBSCAN / k-means partitions against
  matched-k k-means on kinematics and time-shifted labels.
- `tune_neural.py`: the tuning score. dR2 = R^2(spikes ~ kin + clusters) -
  R^2(spikes ~ kin) (ridge, blocked CV), each config over a min-cluster-size
  grid, as excess over kinematics k-means at the same k. `--kin-pos` adds
  arena position (quadratic + 8 x 8 grid) and image heading to kin;
  `--detrend-min N` removes slow firing drift first. `tune_<tag>/labels.npz`
  keeps every partition.
- `confirm_neural.py`: XGBoost bits from spikes, kinematics, both, and
  dbits = what spikes add beyond kinematics, for partitions in labels.npz.

Results (2026-10-16), Leonidas (EY9755) 20261006_155113 (tuned on; frames
black from min ~23 to 76, lights off) and 20261008_145014 (held out; 491
units, no dark period):

- Spikes predict the clusters well: 35 leaf clusters (part maps, last layer)
  decoded at 46% balanced accuracy (chance 3%), 2.73 bits/window; the lag
  scan peaks at 0 s (alignment confirmed). M5/6 ~ DS > VS ~ M2/3 at equal
  unit counts. Kinematics alone decode them better (55%); spikes add 0.29
  bits beyond kinematics.
- A raw neural score is a poor tuning target: it rises or falls with k
  depending on the score, and k-means on kinematics alone matches or beats
  the V-JEPA clusters at every k.
- Clustering knobs (PCA dims, UMAP dims / neighbours / min_dist, HDBSCAN
  min_samples / leaf vs eom) and window length (0.5 / 1 / 2 s) are flat on
  dR2 (seed noise ~0.0001).
- Encoder layer matters: blocks 16-20 beat the last layer on dR2 on both
  sessions (held out: 0.0165 vs 0.0130, three seeds), but XGBoost dbits does
  not move (0.27-0.35 vs 0.29-0.31 bits) and part of layer 16's held-out
  gain is slow drift (+27% -> +12% after 10-min detrending).
- Adding kinematics to the clustering features (kin_w 1.0) raised dR2 on
  10-06 but not held out, and lowered XGBoost dbits on both: the labels get
  more kinematic. Not adopted.
- Raw vs masked vs part maps: raw / masked features encode arena position
  (R^2 0.81-0.93 / 0.72-0.81 vs 0.31-0.47 for part maps) and leftover image
  heading. Position also explains neural variance on its own, so these were
  scored with `--kin-pos`. 10-06: all inputs about equal. Held out: part maps
  0.0065 / 0.0083 / 0.0089 (last / L16 / L20) vs raw <= 0.0046, masked
  <= 0.0026. Keep the part maps.
- With place in the baseline, spikes add only ~0.04-0.10 bits beyond
  kinematics + place for any input (part maps 0.29 -> 0.055 on 10-06): many
  clusters are partly "this posture in this place".
- V-JEPA 2.1 ViT-L (`--backbone vjepa2_1-l`, part maps, 384 px), last layer,
  against V-JEPA 2 (dR2 excess; XGBoost bits spikes add, min cluster size
  0.005):

  | | 10-06 dR2 kin / +place | 10-08 dR2 kin / +place | 10-06 bits kin / +place | 10-08 bits kin / +place |
  |---|---|---|---|---|
  | V-JEPA 2 last | 0.0016 / 0.0006 | 0.0130 / 0.0065 | 0.29 / 0.055 | 0.31 / 0.016 |
  | V-JEPA 2 L16 | 0.0020 / 0.0010 | 0.0165 / 0.0083 | 0.27 / - | 0.35 / - |
  | V-JEPA 2 L20 | 0.0020 / 0.0010 | 0.0162 / 0.0089 | - / 0.063 | - / -0.010 |
  | V-JEPA 2.1 last | 0.0022 / 0.0011 | 0.0162 / 0.0086 | 0.33 / 0.104 | 0.34 / 0.053 |

  2.1's last layer is best or tied everywhere, and the only one ahead on
  bits beyond kinematics + place in both sessions (V-JEPA 2's layer-20 gain
  on dR2 does not carry over to decoding there). Its own intermediate blocks
  (6 / 12 / 18) are worse than its last. It encodes no more arena position
  than V-JEPA 2 (R^2 0.35). Cost: ~3x slower (~9 vs ~25 windows/s).
- Defaults were not changed: `cluster_embeddings.py` still uses the last
  layer of the part-map embedding.

Bug fixed in the process: XGBoost early stopping on an inner split, without
refitting on the whole training fold, scored -9 bits whenever a cluster
occurs only late in a session; folds now refit, and `base_score` is fixed so
a class absent from training does not get an extreme intercept.

## Hand-labelled SLEAP keypoints (other sessions and animals)

The SLEAP project on the lab server (`sleap_kp.py` has the paths) holds
~1400 frames with human-placed snout / paw / tail-base keypoints from six
videos. They are the only labels that are independent of SAM and of the one
session the clips come from. Once:

a. `python extract_kp_frames.py` caches mouse-centred crops and keypoints in
   `PartSeg/sleap_gt/`, and flags frames whose left/right labels are mirrored
   (`lr_suspect`; `lr_suspect_frames.jpg` shows them).
b. `python prompt_from_keypoints.py --body-from <video>/seg_runs/<clip-only run>`
   makes training labels for those frames: paws from SAM3 clicked on the
   keypoints, body from the earlier network where it is confident.

Then `train_seg.py` mixes them in automatically (half of each batch), holds
out two whole videos, and reports how often a hand-placed paw keypoint gets
the right paw label there. `python eval_keypoints.py <run>` gives that score
for any model, with a sheet of the worst frames. For the final model:
`python train_seg.py --val-clips none --kp-val-videos none`.

What was tried on the held-out sessions (2026-10-12; right-paw rate / all
paws right in a frame; two ResNet34 seeds differ by ~0.4 / ~2 points):

| setup | right paw | all paws |
|---|---|---|
| clips only (no SLEAP frames), 6000 it | 88.8% | 64.6% |
| + hand-labelled SLEAP frames, 6000 it | 92.2% | 72.5% |
| same, 12000 it | 94.2% | 79.7% |
| same, 24000 it (the default now) | 95.0% | 82.5% |
| + tail class, 24000 it (2026-10-15) | 95.3% | 83.4% |
| ResNet50, 12000 it | 94.3% | 80.0% |
| ResNet34 at 768 px, 6000 it | 90.8% | 70.4% |
| + 2409 frames from SLEAP predictions (`extract_pred_frames.py`, `--kp-extra`), 12000 it | 93.6% | 77.6% |

So: train long; a bigger ResNet, a larger input and same-animal prediction
frames do not help. New animals / sessions with hand labels should.

### Tail class and body axis (2026-10-15)

The network has a 7th class, `tail` (from the clips' tail sink masks). A
clip frame without a tail mask, and every SLEAP frame, has only "background
or tail" as its background target, so unprompted tails are never taught as
background. `axis.py` turns a label map into a tail base (where the tail
meets the body, tolerating the few px of background the network often leaves
there) and the axis tail base -> body centroid. `eval_keypoints.py` scores
both against the human `tail_base` / snout keypoints for tail models.

On the two held-out sessions (`seg_runs/20261015_014659_resnet34`):
tail base found in 93.7% of frames with a visible tail_base keypoint, median
7.2 px from it, 90% within 25 px. Heading error against tail_base -> snout:

| axis | defined | median | 90th pct | > 45 deg |
|---|---|---|---|---|
| tail base -> body centroid | 89.3% | 7.1 | 21.3 | 0.5% |
| hindpaw -> forepaw midpoints | 86.9% | 11.4 | 43.9 | 9.9% |

Packaged final model (all clips but 002149, all six SLEAP sessions):
`PartSeg/models/partseg_20261015_014832_resnet34.pt`. `seg_model.WEIGHTS`
still points at the 6-class model; pass `--model` to use the tail one.
`run_video.py` takes either. The tail is not drawn in the solid render, so
`embed_vjepa.py` input is the same with both.

Before training, check that `evaluate.py` shows no L/R disagreement between
clips. A clip clicked with L/R the other way round teaches the network that
left and right are arbitrary.

`prompts.json` is a replayable event log, so prompts generated from SLEAP
keypoints only need to write the same format and then go through step 3
(for tracked clips; the hand-labelled frames above are single frames).

## Behaviour-only scoring of the clusters (no spikes)

Spikes are kept for the final, held-out validation; clusterings are compared
on behaviour alone (added 2026-10-18).

1. `python part_tracks.py R` writes `R/part_tracks.npz`: per-frame centroids
   of body and paws (full-res px), part areas, heading. These measurements
   come from the label maps only, so they are independent of V-JEPA and of
   the clustering (~5 min per session).
2. Embed with `embed_vjepa.py` (`--centre raw`, the default since 2026-10-18; e.g. `--backbone vjepa2_1-l
   --layers 6,12,18 --out-name vjepa21_rc`). The default `--centre crop`
   centres each frame on run_video.py's crop, which follows a centred
   smoothing of the track that reads ~0.7 s ahead. The part maps then show
   where the mouse is about to go, and at a clipped crop (frame edge) where
   it is in the arena.
3. `python score_behaviour.py R --fit-run R_OTHER --emb vjepa21_rc`: the
   partitions are fitted on another session and assigned to this one. Score =
   held-out log-likelihood (bits per window) of the mouse's next movements
   (posture 0.13-1.07 s ahead; summaries over 0.7-2 s and 2-5 s) given the
   window's state, BEYOND a history baseline (window kinematics, posture at
   the window's end and its change). The script's docstring has the
   details. It compares every partition with k-means on the history
   features at equal state entropy (`excess`), with a time-shifted and a
   bout-permuted copy of itself, and with the continuous embedding
   (`ceiling`). `--parent NAME` puts a parent partition into the baseline, to
   score subtypes within it. Output: `R/<emb>/behaviour/score_<tag>/`.
4. `python abx_behaviour.py R/<emb>/behaviour/score_<tag>`: cross-session
   ABX: do two windows with the same state, one from each session, move more
   alike (DTW on the part trajectories, speed tercile matched) than windows
   with different states? A diagnostic only, because a partition of those
   trajectories themselves would win it.

First result, 10-08, partitions fitted on the same session
(`score_leakcheck`). With `--centre crop`, the V-JEPA partitions' gain at
0.13-1 s about halves once re-centred (km_emb_k80 at 0.53 s: 0.172 -> 0.064
bits), and the continuous ceiling drops 0.32 -> 0.24 bits; at 0.7-2 s the
ceiling is unchanged (0.51 / 0.50). Re-centred, kinematic k-means wins
the short horizons, and V-JEPA partitions win 0.7-5 s at equal entropy
(leaf HDBSCAN f=0.005: +0.15 bits excess).

Cross-session (`vjepa21_rc/behaviour/score_cross` on 10-08, fitted on 10-06,
all re-centred): the time-shifted and bout-permuted nulls score ~0 bits.
Kinematic k-means wins 0.13-0.27 s; it can exceed the "ceiling" there, which
is linear in the embedding. HDBSCAN on V-JEPA wins 0.53 s-5 s at equal
entropy, but only by ~0.02-0.04 bits (excess, add form; ceiling 0.44 bits
at 0.7-2 s). Frozen vs refitted AMI ~0.45-0.6 for every method. ABX error
(0.5 = chance): km_kin 0.27-0.34, hdb 0.35-0.36, km_emb 0.36-0.47. As
expected, the partition of the measured kinematics wins the measured-space
diagnostic.

## Action segmentation with TW-FINCH (added 2026-10-29)

`python tw_finch.py R [--emb vjepa21_rc]` runs TW-FINCH (Sarfraz et al.,
CVPR 2021; temporally weighted first-neighbour clustering: distance = cosine
distance x |time difference|) on the window features (standardised, PCA 50).
Checked against a dense transcription of the authors' code on synthetic
videos: identical hierarchy and exact-K merges. ~45 s per 4 h session on the
GPU. Output `R/<emb>/twfinch/`: labels.npz, segments.npz, summary.txt,
overview.png. Three kinds of partition:

- `tw_k<K>`: TW-FINCH itself at K clusters. Over a whole session the time
  weight wins and the clusters are long blocks of the session (10-08, K=20:
  median bout 88 s, ~3 bouts per cluster), not recurring behaviours.
- `twseg_l<level>_k<K>`: TW-FINCH as a segmenter. The runs of one hierarchy
  level are segments (10-08 levels 0-3: median 1.5 / 4.5 / 14 / 39 s), and
  k-means on the segment means gives K recurring types.
- `shufseg_l<level>_k<K>`: control, the same segment lengths in random
  order, then the same k-means.

Leak-free check (`summary.txt`, homog): share of window kinematics + posture
variance (part_tracks, not V-JEPA) explained by segment identity. TW-FINCH
beats shuffled lengths at every level on both sessions, by 0.035-0.042 at
levels 1-2 (10-08: 0.525 vs 0.490, 0.352 vs 0.311; 10-06: +0.042, +0.038;
20+ SDs of the shuffles). The boundaries fall where behaviour changes, but
the gain is small.

`score_behaviour.py --labels R/<emb>/twfinch/labels.npz` (10-08, fitted on
itself, `score_twfinch`): a segment's label uses windows up to a segment
ahead, so the future-movement scores are LEAKY. Even shuffled segments beat
HDBSCAN at 0.7-5 s (shufseg_l2_k20 0.256 vs hdb_f0.005 0.232 bits; best
twseg_l2_k20 0.349, ~90% of the continuous ceiling). Compare twseg with
shufseg only: at <= 1 s TW-FINCH is ahead in 15/20 (level, K) pairs by
<= 0.036 bits; at 0.7-5 s, levels 0-1 go either way (+-0.06, varies with K),
level 2 is ahead in 4/5 and level 3 in 5/5 (+0.08-0.19). tw_k* score ~0.

`label_video.py R --emb vjepa21_rc --labels R/vjepa21_rc/twfinch/labels.npz
--key twseg_l1_k20,twseg_l2_k20,win_km_k20` writes
`twfinch/labeled_twseg_l1_k20.mp4`: the first key labels the panel, each key
gets an ethogram row. label_video.py now finds the window nearest in time
(it used to count hops from the first window, which shifted labels after
every dropped window) and shows dropped windows as "no window".

## Sticky HMM on the window features (added 2026-10-29)

`python sticky_hmm.py R [--fit-run R_OTHER] [--ks 10,20,40] [--kappas 0,1e4]
[--tempers 1,0.3,0.1]`: Gaussian HMM (full covariance, shrunk towards the
global one) on the standardised PCA-50 features, MAP-EM from a k-means
start, with `kappa` extra pseudo-counts on self-transitions (the MoSeq idea).
Written here (numba forward-backward / Viterbi; checked against brute-force
enumeration and on synthetic sticky HMMs). ~5-25 s per model. Two label sets
per model: the Viterbi path (uses the whole session) and `_causal`, the
argmax of the forward-filtered posterior (windows up to this one only), which
score_behaviour.py can compare with km_emb / hdb on equal terms. Output
`R/<emb>/stickyhmm/` (or `stickyhmm_cross/` with `--fit-run`).

Stickiness alone barely works in 50-d: the per-window evidence swamps the
prior (kappa 1e5 only takes K=20 bouts from 1.8 to 2.7 s mean).
`--temper t` multiplies the emission log-likelihoods by t (consecutive
windows overlap, so they are not independent evidence); t = 0.1 gives
3.5-9 s median Viterbi bouts (causal 2.5-5 s). PCA 10 alone helps little.

Scores (10-08, `score_stickyhmm` / `score_stickyhmm_cross`, excess bits at
0.7-5 s over kinematic k-means at equal entropy):

- Fitted on 10-08 itself: the best causal HMM (k20, kappa 0, t 0.1; 2.5 s
  median bouts) 0.184 vs best HDBSCAN 0.171 (0.5 s bouts) and km_emb_k20
  0.051; several causal HMMs match or beat HDBSCAN. Viterbi labels reach
  0.376 but see the future.
- Fitted on 10-06, applied to 10-08 (frozen): the gain mostly goes. Best
  causal HMM 0.051 (k40, kappa 1e4, t 0.3) vs hdb_f0.01 0.050 and
  km_emb_k20 0.040; the t = 0.1 models fall to ~0 (0.002-0.022). Probably a
  shift of the features between sessions (not checked).
- At <= 1 s, kinematic k-means still wins (excess_short <= 0), as for
  every V-JEPA partition.

Video: `videos/Leonidas_20261008_stickyhmm_k20_t0.1_115-130min.mp4`
(`label_video.py --labels R/vjepa21_rc/stickyhmm/labels.npz --key
hmm_k20_kap0_t0.1,hmm_k20_kap0_t0.1_causal,hmm_k20_kap10000_t0.1`; causal
labels use the same state numbers as the Viterbi path).

## Keypoints distilled from the lab's SLEAP model (added 2026-10-19)

The SLEAP top-down model (1405 labels: ConvNeXt-T centroid + Swin-S instance)
is ~200 M parameters and runs at ~2.5 frames/s on the 3090. Its search history
and validation predictions show that ImageNet pretraining mattered (+0.05
PCK@10), Swin-T vs Swin-S did not (−0.008, 95% CI ±0.016 on 114 frames), and
~80% of the compute is the centroid stage, which this pipeline's localizer
already does. So the part network learns the six SLEAP nodes from it:

1. `~/.local/share/uv/tools/sleap/bin/python sleap_teacher.py R` (SLEAP env)
   runs the teacher on ~900 random 20-frame runs of the session (~1 h 45 min),
   in resumable parts in `<session>/sleap/teacher/`.
2. `python distill_frames.py R` cuts those frames the way run_video.py does
   (768 px at x0, y0) into `PartSeg/distill_<session>/` (sleap_gt's layout +
   per-node teacher scores).
3. `python train_seg.py --kp-distill PartSeg/distill_<session> --init
   PartSeg/models/partseg_20261015_014832_resnet34.pt --exclude-clips
   002149_left-turn`: adds a 6-channel heatmap head (seg_model.KP_NAMES).
   Targets: hand-placed keypoints (not visible = absent) and teacher nodes
   scored >= 0.5 (the rest unknown); CenterNet focal loss. `forward(x)` still
   returns the segmentation only, so run_video.py, TensorRT and the eval
   scripts are unchanged; `forward(x, keypoints=True)` adds the heatmaps and
   `seg_model.decode_keypoints()` reads them.
4. `python eval_distill.py CKPT R_HELDOUT` compares student and teacher on a
   session neither trained on (the teacher's full-pipeline .slp files in
   `<session>/sleap/`), per node and per file (e.g. grooming bouts).

First student (2026-10-19): `seg_runs/20261019_130918_resnet34` (from the
7-class network, 24000 iterations, 17,068 teacher frames from 10-06; run as
a capped `systemd-run --user` unit, ~52 min, 7.9-9.6 GB RAM). On the held-out
human-labelled videos it places 82-95% of points within 10 px (snout 82%,
median 4.2 px; paws 2.8-3.4 px), the paw label is right 98.2% of the time
(was ~95%), part IoU 0.868, 240 fps at batch 1. Against the teacher on 10-08
(never seen by either): median distance 1.9-3.2 px for paws, 5.9 px snout
(4-8 px in grooming bouts, 91-93% within 20 px). Its confidences run lower
than the teacher's: thresholds of ~0.15-0.3 match the teacher's presence
rate (agreement 76-89%), so don't reuse SLEAP's 0.5. Weakest: the tail base
(64% within 10 px; weak presence signal) and points in the tail-wrapped
grooming bout at 107.5 min.

`run_video.py --model <keypoint run>` writes the keypoints too: frames.csv
gets kp_<node>_x / _y (full-res image px) and kp_<node>_score for the six
SLEAP nodes, decoded on the GPU (the TensorRT engine has a second output,
kp_logits). Checked on 10-08 frames 363000-363299: identical to the PyTorch
evaluation (per-node median distance to SLEAP within 0.2 px), 249 fps.
Models without a keypoint head run exactly as before.

Hand clicks for grooming snouts (2026-10-19): `sleap_click_project.py`
(SLEAP env) builds `<session>/sleap/groom_clicks.slp`, a blind labelling
project: 60 suggested frames of 10-08 in shuffled order (40 spread over the
three grooming bouts, the 10 frames of the 107.5 min bout where teacher and
student disagree most, 10 from other grooming-like moments;
`groom_clicks_frames.json`), no predictions shown. `groom_clicks_teacher.slp`
holds the teacher on the same frames. After labelling,
`python score_clicks.py groom_clicks.slp groom_clicks_teacher.slp CKPT R
--groups groom_clicks_frames.json` scores both models per frame group.

Hand-click scores (60 grooming frames of 10-08, `groom_clicks_summary.txt`):
snout within 20 px SLEAP 77% vs student 62%; when confident, SLEAP is >30 px
off 2% of the time (median 7.3 px), the student 13% (median 5.2 px; closer
than SLEAP in 73% of the 41 frames where both are confident). Left paws: the
student is better (left hind 98% vs 65% within 10 px); right forepaw: SLEAP
is better. The right hind paw is mostly hidden in these poses and the tail
base is clicked ~17 px further inside the body than either model puts it,
so those two rows measure label convention / guesses more than accuracy.

Second student (2026-10-19): `seg_runs/20261019_185549_resnet34`, fine-tuned
12000 iterations from the first on its 17,068 frames plus 6,314 teacher frames
mined from 10-06 grooming-like / curled (tail base missing) 2 s blocks
(`<session>/sleap/hard_frames.json`, `sleap_teacher.py --frames-json ... --name
teacher_hard`, `distill_frames.py --teacher teacher_hard`; whole chain in
`PartSeg/distill_hard.sh`, run as a capped systemd user unit, ~70 min). On the
60 hand-clicked grooming frames: snout within 20 px 62% -> 73% (SLEAP 77%), and
snouts >30 px off when confident 13% -> 0% (median 4.4 px when confident; the
10 disagreement frames 50% -> 80% within 20 px). Paws unchanged; the tail base
is confident less often (50%). On 10-08 vs SLEAP: snout within 20 px 93.6%
(was 91.9%), paws unchanged, tail base 56% within 10 px (was 64%). Held-out
human labels: PCK@10 89.9%, part IoU 0.867.
Note: the SLEAP models were trained at max width 2148 px; Rig2 frames are
2248 px, so sleap-nn rescales and pads every frame (its warning).

Packaged default since 2026-10-19: `models/partseg_20261019_185549_resnet34_kp.pt`
(seg_model.WEIGHTS; 7 classes + keypoints). TensorRT fp16 on the 3090: the
network alone 1083 crops/s at batch 64 (PyTorch bf16 366), same as the
segmentation-only network; run_video.py end to end on 10-06 (5 min, cv2
decoding): 451 fps with the preview, 555 without, i.e. a 3.4 h, 60 fps
session in ~25 min. The old defaults are kept in models/ (pass --model).

Whole sessions with the keypoint default (2026-10-20, `PartSeg/run_kp_sessions.sh`
as a capped user unit): 10-06 484 fps (25 min), 10-08 475 fps (30 min), in
`<session>/full_video/partseg_20261019_185549_resnet34_kp/`. Crops identical to
the earlier runs. Confident (>= 0.2): snout 75-80%, paws 79-98%, tail base
70-77%; confident paws land on their own mask in 400/400 random frames; on
10-08 vs SLEAP's 1441 frames the medians match the evaluation (snout 5.9 px).
Paw jumps > 50 px between frames (~2%) are mostly real swing during running
(81-91% while the body moves > 5 px/frame); L/R swaps ~0.1% of frames.
