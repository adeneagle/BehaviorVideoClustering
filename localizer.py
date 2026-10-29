"""Body-centroid localizer distilled from SAM2 (body only, tail eroded), with a presence head.

Copied from the lab server (Aden/code/video_processing/localizer.py and
localizer2.py); weights are PaulData/paul_all/models/loc2_final.pt there,
cached locally in <out_root>/models/.

Input: uint8 gray frame downscaled (INTER_AREA) to 352x384 (H x W) from the
full frame. Output: (x, y) in small-image pixels, the heatmap logits, and a
presence logit (sigmoid >= 0.5: a mouse is there). ResNet18 through layer3
with a layer2 skip -> stride-8 heatmap -> soft-argmax.
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import DEFAULT_OUT_ROOT

H, W = 352, 384
WEIGHTS = DEFAULT_OUT_ROOT / "models" / "loc2_final.pt"
SERVER_WEIGHTS = Path("/mnt/howlin/mnt/data/samba/yttri/Aden/PaulData/paul_all/models/loc2_final.pt")


class Localizer2(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import resnet18

        r = resnet18(weights=None)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)
        self.l1, self.l2, self.l3 = r.layer1, r.layer2, r.layer3
        self.lat3 = nn.Conv2d(256, 128, 1)
        self.head = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
                                  nn.Conv2d(64, 1, 1))
        self.pres = nn.Sequential(nn.Linear(256 * 2 + 128, 64), nn.ReLU(True), nn.Linear(64, 1))
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x_u8):  # (B,H,W) uint8
        x = (x_u8.float() / 255.0).unsqueeze(1).expand(-1, 3, -1, -1)
        x = (x - self.mean) / self.std
        f2 = self.l2(self.l1(self.stem(x)))
        f3 = self.l3(f2)
        f = f2 + F.interpolate(self.lat3(f3), size=f2.shape[-2:], mode="bilinear", align_corners=False)
        lg = self.head(f)[:, 0]
        B, h, w = lg.shape
        p = F.softmax(lg.flatten(1), 1).view(B, h, w)
        gx = (torch.arange(w, device=lg.device).float() + 0.5) * 8.0
        gy = (torch.arange(h, device=lg.device).float() + 0.5) * 8.0
        xy = torch.stack([(p.sum(1) * gx).sum(1), (p.sum(2) * gy).sum(1)], 1)
        feat = torch.cat([f3.mean((2, 3)), f3.amax((2, 3)), f.amax((2, 3))], 1)
        return xy, lg, self.pres(feat)[:, 0]


def load_localizer(path=None, device="cuda"):
    path = Path(path) if path else (WEIGHTS if WEIGHTS.exists() else SERVER_WEIGHTS)
    model = Localizer2()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model.to(device).eval()


@torch.no_grad()
def locate(model, small_u8):
    """[B,352,384] uint8 tensor -> (xy [B,2] in small-image px, presence prob [B]), numpy."""
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=small_u8.is_cuda):
        xy, _, pl = model(small_u8)
    return xy.float().cpu().numpy(), torch.sigmoid(pl.float()).cpu().numpy()
