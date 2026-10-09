"""Compact PyTorch implementations of conditional U-Net and visual encoders.

Architecture follows the pinned official DP/DP3/FlowPolicy reference files in
vendor/baseline_reference; see their MIT licenses and provenance.json.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class ResidualImageBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
            nn.GroupNorm(8, cout), nn.ReLU(), nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.GroupNorm(8, cout))
        self.skip = nn.Conv2d(cin, cout, 1, stride, bias=False) if cin != cout or stride != 1 else nn.Identity()

    def forward(self, x):
        return F.relu(self.body(x)+self.skip(x))


class ObservationEncoder(nn.Module):
    def __init__(self, modality, width=128):
        super().__init__()
        self.modality = modality
        if modality == 'rgb':
            self.visual = nn.Sequential(nn.Conv2d(3, 32, 5, 2, 2), nn.GroupNorm(8, 32), nn.ReLU(),
                ResidualImageBlock(32, 64, 2), ResidualImageBlock(64, 128, 2),
                ResidualImageBlock(128, 128), nn.AdaptiveAvgPool2d((3, 4)), nn.Flatten(), nn.Linear(1536, width))
        else:
            self.visual = nn.Sequential(nn.Linear(3, 64), nn.LayerNorm(64), nn.ReLU(),
                nn.Linear(64, 128), nn.LayerNorm(128), nn.ReLU(), nn.Linear(128, 256))
            self.point_projection = nn.Sequential(nn.Linear(256, width), nn.LayerNorm(width))
        self.state = nn.Sequential(nn.Linear(34, 64), nn.Mish(), nn.Linear(64, 64))
        self.output_dim = 2*(width+64)

    def forward(self, batch):
        x = batch['rgb' if self.modality == 'rgb' else 'points']
        b, slots = x.shape[:2]
        if self.modality == 'rgb':
            visual = self.visual(x.flatten(0, 1).permute(0, 3, 1, 2).float()/255.)
        else:
            points = x.flatten(0, 1).float()
            points = (points-points.new_tensor([.65, 0, .22]))/points.new_tensor([.55, .55, .5])
            visual = self.point_projection(self.visual(points).amax(1))
        state = self.state(batch['state'].flatten(0, 1).float())
        return torch.cat((visual, state), -1).reshape(b, slots, -1).flatten(1)


class TimeEmbedding(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        self.width = width
        self.layers = nn.Sequential(nn.Linear(width, width*4), nn.Mish(), nn.Linear(width*4, width))

    def forward(self, t):
        f = torch.exp(torch.arange(self.width//2, device=t.device)*(-math.log(10000)/(self.width//2-1)))
        h = t.float()[:, None]*f[None]
        return self.layers(torch.cat((h.sin(), h.cos()), -1))


class ConditionalBlock(nn.Module):
    def __init__(self, cin, cout, cond):
        super().__init__()
        self.first = nn.Sequential(nn.Conv1d(cin, cout, 5, padding=2), nn.GroupNorm(8, cout), nn.Mish())
        self.second = nn.Sequential(nn.Conv1d(cout, cout, 5, padding=2), nn.GroupNorm(8, cout), nn.Mish())
        self.film = nn.Sequential(nn.Mish(), nn.Linear(cond, 2*cout))
        self.skip = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, condition):
        scale, shift = self.film(condition).chunk(2, -1)
        h = self.first(x)*scale[..., None]+shift[..., None]
        return self.second(h)+self.skip(x)


class ConditionalUNet(nn.Module):
    """Three-resolution FiLM U-Net; terminal padding admits a 30-step chunk."""
    def __init__(self, condition_dim, widths=(64, 128, 256)):
        super().__init__()
        self.time = TimeEmbedding()
        cond = condition_dim+64
        self.down = nn.ModuleList()
        cin = 7
        for cout in widths:
            self.down.append(nn.ModuleList([ConditionalBlock(cin, cout, cond), ConditionalBlock(cout, cout, cond)]))
            cin = cout
        self.middle = nn.ModuleList([ConditionalBlock(cin, cin, cond) for _ in range(2)])
        self.up = nn.ModuleList()
        for cout in reversed(widths[:-1]):
            self.up.append(nn.ModuleList([ConditionalBlock(cin+cout, cout, cond), ConditionalBlock(cout, cout, cond)]))
            cin = cout
        self.out = nn.Sequential(nn.Conv1d(cin, cin, 5, padding=2), nn.GroupNorm(8, cin), nn.Mish(), nn.Conv1d(cin, 7, 1))

    def forward(self, actions, t, condition):
        horizon = actions.shape[1]
        x = F.pad(actions.transpose(1, 2), (0, (-horizon)%4), mode='replicate')
        cond = torch.cat((self.time(t), condition), -1)
        skips = []
        for i, blocks in enumerate(self.down):
            for block in blocks:
                x = block(x, cond)
            if i < len(self.down)-1:
                skips.append(x)
                x = F.avg_pool1d(x, 2)
        for block in self.middle:
            x = block(x, cond)
        for blocks, skip in zip(self.up, reversed(skips)):
            x = torch.cat((F.interpolate(x, size=skip.shape[-1], mode='nearest'), skip), 1)
            for block in blocks:
                x = block(x, cond)
        return self.out(x).transpose(1, 2)[:, :horizon]
