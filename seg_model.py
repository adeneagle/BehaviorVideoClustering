"""Small, fast part-segmentation network trained on the SAM3 part masks.

A timm encoder (ImageNet-pretrained) with a light U-Net decoder that predicts
one class per pixel at full input resolution. Classes are contiguous indices:

    0 background, 1 body, 2 forepaw_L, 3 forepaw_R, 4 hindpaw_L, 5 hindpaw_R, 6 tail

The tail gives the body axis (tail base -> body centroid). It is an optional
sink part in the SAM clips, so wherever a frame has no tail mask (not
prompted yet, or SAM lost it) its background is only "background or tail"
(BG_OR_TAIL), never a false "not tail". The implant is folded into
background. Neither is drawn in the solid render.

Networks trained before 2026-10-15 have the first six classes only; their
checkpoints carry their own class list.

Keypoints (2026-10-19, distilled from the lab's SLEAP model, see distill_frames.py): a network
built with n_kp > 0 also has a heatmap head on the last decoder features, one channel per SLEAP
node (KP_NAMES; sigmoid = confidence that the point is there). forward(x) still returns only the
segmentation logits, so every existing consumer works unchanged; forward(x, keypoints=True)
returns (seg logits, keypoint logits) and decode_keypoints() turns the latter into points.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import ANATOMICAL, DEFAULT_OUT_ROOT, PART_BY_ID, part_id

# the packaged model run_video.py and trt_engines.py use by default (since 2026-10-19): a copy of
# seg_runs/20261019_185549_resnet34/best.pt: 7 classes (with tail) + the six SLEAP keypoints, distilled
# from the lab's SLEAP model (README "Keypoints distilled"). Earlier defaults, still in models/:
# partseg_20261012_202544_resnet34.pt (6 classes), partseg_20261015_014832_resnet34.pt (7 classes)
WEIGHTS = DEFAULT_OUT_ROOT / "models" / "partseg_20261019_185549_resnet34_kp.pt"

# class index <-> SAM obj id (label value in masks.npz)
CLASS_OBJ_IDS = [0] + [p.obj_id for p in ANATOMICAL] + [part_id("tail")]
CLASS_NAMES = ["background"] + [p.name for p in ANATOMICAL] + ["tail"]
N_CLASSES = len(CLASS_NAMES)
TAIL = CLASS_NAMES.index("tail")
IGNORE = 255
# "not a paw": the target for the non-paw pixels of frames labelled only by
# paw keypoints (see prompt_from_keypoints.py)
NOTPAW = 254
# "background or tail": background in frames with no tail label
BG_OR_TAIL = 253
# partial targets -> the classes they allow
PARTIAL = {NOTPAW: [0, 1, TAIL], BG_OR_TAIL: [0, TAIL]}
assert CLASS_NAMES[:2] == ["background", "body"]

# SLEAP node order (sleap_kp.NODES); left/right are the animal's own
KP_NAMES = ("snout", "left_front_paw", "right_front_paw", "left_hind_paw", "right_hind_paw", "tail_base")
KP_MIRROR = [KP_NAMES.index(n.replace("left", "@").replace("right", "left").replace("@", "right")) for n in KP_NAMES]


def obj_to_class_lut():
    lut = torch.zeros(256, dtype=torch.uint8)  # unknown ids (implant, retired) -> background
    for c, o in enumerate(CLASS_OBJ_IDS):
        lut[o] = c
    for code in (IGNORE, *PARTIAL):
        lut[code] = code
    return lut.numpy()


def mirror_class_lut():
    """Class permutation for a horizontally flipped image: the animal's left
    paw becomes its right paw. Flipping without this swap would teach the
    model that left and right are interchangeable."""
    lut = list(range(256))
    names = CLASS_NAMES
    for i, n in enumerate(names):
        for a, b in (("_L", "_R"), ("_R", "_L")):
            if n.endswith(a):
                lut[i] = names.index(n[: -len(a)] + b)
    return torch.tensor(lut, dtype=torch.uint8).numpy()


class ConvBlock(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )


class PartSegNet(nn.Module):
    def __init__(self, encoder="resnet34", n_classes=N_CLASSES, pretrained=True, dec_ch=(256, 128, 64, 32, 16),
                 img_size=512, n_kp=0):
        super().__init__()
        import timm

        # Swin encoders are built for one input size (img_size must be a
        # multiple of 256 so every stage divides into 8x8 windows) and return
        # channels-last features.
        self.channels_last = encoder.startswith("swin")
        kw = {}
        if self.channels_last:
            kw = dict(img_size=img_size, **({} if encoder.startswith("swinv2") else dict(window_size=8)))
        self.encoder = timm.create_model(encoder, pretrained=pretrained, features_only=True, **kw)
        enc_ch = self.encoder.feature_info.channels()[::-1]  # deepest first
        self.blocks = nn.ModuleList()
        cin = enc_ch[0]
        skips = enc_ch[1:] + [0] * (len(dec_ch) - len(enc_ch) + 1)
        for skip, cout in zip(skips, dec_ch):
            self.blocks.append(ConvBlock(cin + skip, cout))
            cin = cout
        self.head = nn.Conv2d(cin, n_classes, 1)
        self.n_kp = n_kp
        if n_kp:
            self.kp_head = nn.Sequential(nn.Conv2d(cin, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32),
                                         nn.ReLU(inplace=True), nn.Conv2d(32, n_kp, 1))
            nn.init.constant_(self.kp_head[-1].bias, -4.6)  # sigmoid ~0.01 everywhere at the start (CenterNet)
        self.reductions = self.encoder.feature_info.reduction()

    def forward(self, x, keypoints=False):
        H, W = x.shape[-2:]
        feats = self.encoder(x)[::-1]
        if self.channels_last:
            feats = [f.permute(0, 3, 1, 2) for f in feats]
        y = feats[0]
        for i, blk in enumerate(self.blocks):
            skip = feats[i + 1] if i + 1 < len(feats) else None
            size = skip.shape[-2:] if skip is not None else (y.shape[-2] * 2, y.shape[-1] * 2)
            y = F.interpolate(y, size=size, mode="bilinear", align_corners=False)
            if skip is not None:
                y = torch.cat([y, skip], 1)
            y = blk(y)
        feat = y
        y = self.head(feat)
        if y.shape[-2:] != (H, W):
            y = F.interpolate(y, size=(H, W), mode="bilinear", align_corners=False)
        if not keypoints:
            return y
        k = self.kp_head(feat)
        if k.shape[-2:] != (H, W):
            k = F.interpolate(k, size=(H, W), mode="bilinear", align_corners=False)
        return y, k


def decode_keypoints(kp_logits, radius=2):
    """[B,K,H,W] keypoint logits -> xy [B,K,2] (network px, sub-pixel: probability-weighted
    centroid in a (2r+1)^2 window around the peak) and score [B,K] (peak sigmoid)."""
    B, K, H, W = kp_logits.shape
    p = kp_logits.float().sigmoid()
    score, idx = p.flatten(2).max(2)
    cy, cx = idx // W, idx % W
    off = torch.arange(-radius, radius + 1, device=p.device)
    ys = (cy[..., None, None] + off[:, None]).clamp(0, H - 1)
    xs = (cx[..., None, None] + off[None, :]).clamp(0, W - 1)
    flat = p.flatten(2)
    w = torch.gather(flat, 2, (ys * W + xs).flatten(2)).view(B, K, 2 * radius + 1, 2 * radius + 1)
    w = w / w.sum((2, 3), keepdim=True).clamp(min=1e-9)
    x = (w * xs.float()).sum((2, 3))
    y = (w * ys.float()).sum((2, 3))
    return torch.stack([x, y], -1), score


IMG_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMG_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def to_input(gray_u8):
    """[B,H,W] uint8 gray (any device) -> normalised [B,3,H,W] float."""
    x = gray_u8.float().div_(255).unsqueeze(1).expand(-1, 3, -1, -1)
    return (x - IMG_MEAN.to(x.device)) / IMG_STD.to(x.device)


def save_checkpoint(path, model, cfg, metrics=None):
    torch.save({"state_dict": model.state_dict(), "cfg": cfg, "metrics": metrics,
                "class_names": CLASS_NAMES, "class_obj_ids": CLASS_OBJ_IDS,
                "kp_names": list(KP_NAMES[: model.n_kp])}, path)


def load_checkpoint(path, device="cuda"):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = PartSegNet(ck["cfg"]["encoder"], len(ck["class_names"]), pretrained=False, img_size=ck["cfg"]["size"],
                       n_kp=len(ck.get("kp_names", [])))
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), ck


assert all(o in PART_BY_ID for o in CLASS_OBJ_IDS[1:])
