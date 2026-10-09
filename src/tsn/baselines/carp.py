"""CARP adaptation: per-joint residual VQ, then masked next-scale prediction.

The original method uses independent action-dimension tokenizers, multi-scale
residual quantization, a frozen tokenizer during policy fitting, and same-scale
bidirectional attention. These are retained; temporal convolutions and a small
Transformer replace the original 2D tokenizer/large adaLN Transformer.
"""
import torch
from torch import nn
from torch.nn import functional as F


class ActionTokenizer(nn.Module):
    scales = (1, 2, 4, 8)

    def __init__(self, vocabulary=256, latent_dim=8):
        super().__init__()
        self.vocabulary, self.latent_dim = vocabulary, latent_dim
        self.encoder = nn.Sequential(nn.Conv1d(7, 112, 5, 2, 2, groups=7), nn.Mish(),
            nn.Conv1d(112, 56, 5, 2, 2, groups=7))
        self.decoder = nn.Sequential(nn.Conv1d(56, 112, 5, padding=2, groups=7), nn.Mish(),
            nn.Conv1d(112, 7, 5, padding=2, groups=7))
        self.codebook = nn.Parameter(torch.randn(7, vocabulary, latent_dim)*.05)
        self.phi = nn.ModuleList([nn.Conv1d(56, 56, 3, padding=1, groups=7) for _ in self.scales])

    def embed(self, indices):
        # indices: (B, scale_length, joint); embeddings: (B, joint*latent, scale_length)
        joints = torch.arange(7, device=indices.device)[None, None]
        return self.codebook[joints, indices].permute(0, 2, 3, 1).flatten(1, 2)

    def contribution(self, indices, scale_index):
        h = F.interpolate(self.embed(indices), size=8, mode='linear', align_corners=False)
        return .5*h+.5*self.phi[scale_index](h)

    def encode(self, actions):
        x = F.pad(actions.transpose(1, 2), (0, 2), mode='replicate')
        z = self.encoder(x).float()
        residual, accumulated = z.detach().clone(), torch.zeros_like(z)
        ids, contexts, losses = [], [], []
        for i, length in enumerate(self.scales):
            contexts.append(F.adaptive_avg_pool1d(accumulated.detach(), length).transpose(1, 2))
            query = F.adaptive_avg_pool1d(residual, length).reshape(len(z), 7, 8, length).permute(0, 1, 3, 2)
            similarity = torch.einsum('bjld,jvd->bjlv', F.normalize(query, dim=-1), F.normalize(self.codebook.detach(), dim=-1))
            indices = similarity.argmax(-1).transpose(1, 2)
            contribution = self.contribution(indices, i)
            accumulated = accumulated+contribution
            residual = residual-contribution.detach()
            ids.append(indices)
            losses.append(.25*F.mse_loss(z, accumulated.detach())+F.mse_loss(accumulated, z.detach()))
        straight = z+(accumulated-z).detach()
        return straight, ids, contexts, torch.stack(losses).mean()

    def decode(self, latent):
        return self.decoder(F.interpolate(latent, size=32, mode='linear', align_corners=False)).transpose(1, 2)[:, :30]

    def loss(self, actions):
        latent, _, _, vq = self.encode(actions)
        reconstruction = self.decode(latent)
        return F.mse_loss(reconstruction, actions)+vq


class NextScalePolicy(nn.Module):
    def __init__(self, condition_dim, tokenizer, width=192, layers=4):
        super().__init__()
        self.tokenizer = tokenizer
        self.input = nn.Linear(8, width)
        self.condition = nn.Linear(condition_dim, width)
        self.position = nn.Parameter(torch.randn(1, 105, width)*.02)
        encoder = nn.TransformerEncoderLayer(width, 4, width*4, .1, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder, layers, enable_nested_tensor=False)
        self.output = nn.Linear(width, tokenizer.vocabulary)
        level = torch.cat([torch.full((length*7,), i) for i, length in enumerate(tokenizer.scales)])
        self.register_buffer('attention_mask', level[:, None] < level[None, :])

    def logits(self, contexts, condition):
        tokens = torch.cat([h.reshape(len(h), -1, 7, 8).flatten(1, 2) for h in contexts], 1)
        length = tokens.shape[1]
        x = self.input(tokens)+self.position[:, :length]+self.condition(condition)[:, None]
        x = self.transformer(x, mask=self.attention_mask[:length, :length])
        return self.output(x)

    def loss(self, actions, condition):
        with torch.no_grad():
            _, ids, contexts, _ = self.tokenizer.encode(actions)
        logits = self.logits(contexts, condition)
        target = torch.cat([x.flatten(1) for x in ids], 1)
        return F.cross_entropy(logits.flatten(0, 1), target.flatten())

    def sample(self, condition, generator=None):
        accumulated = condition.new_zeros(len(condition), 56, 8)
        contexts = []
        for i, length in enumerate(self.tokenizer.scales):
            contexts.append(F.adaptive_avg_pool1d(accumulated, length).transpose(1, 2))
            logits = self.logits(contexts, condition)[:, -length*7:]
            # Upstream sampling is stochastic; TSN uses a seeded categorical draw.
            probability = logits.float().softmax(-1)
            indices = torch.multinomial(probability.flatten(0, 1), 1, generator=generator).reshape(len(condition), length, 7)
            accumulated = accumulated+self.tokenizer.contribution(indices, i)
        return self.tokenizer.decode(accumulated)
