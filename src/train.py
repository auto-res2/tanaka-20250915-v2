import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from diffusers import StableDiffusionPipeline
import timm
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import json
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ViTDraftG(nn.Module):
    """Global Draft network using ViT-tiny for coarse predictions"""
    def __init__(self, in_channels: int = 4, time_embed_dim: int = 256):
        super().__init__()
        # Pre-trained ViT-tiny backbone
        self.vit_backbone = timm.create_model(
            'vit_tiny_patch16_224.augreg_in21k', pretrained=True, num_classes=0
        )
        vit_dim = 192  # ViT-tiny hidden dim

        # Adapt 64×64 latents → 32×32 tokens @ stride 16
        self.input_proj = nn.Conv2d(in_channels, vit_dim, kernel_size=16, stride=16)

        # Simple sinusoidal-like time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, vit_dim),
        )

        # Heads
        self.eps_head = nn.Sequential(
            nn.Linear(vit_dim, vit_dim * 2),
            nn.SiLU(),
            nn.Linear(vit_dim * 2, in_channels * 32 * 32),
        )
        self.logvar_head = nn.Sequential(
            nn.Linear(vit_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1 * 4 * 4),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        B, C, H, W = x.shape
        # Down-sample to 32×32 for ViT
        if H > 32:
            x_d = F.interpolate(x, size=(32, 32), mode="bilinear")
        else:
            x_d = x
        tokens = self.input_proj(x_d).flatten(2).transpose(1, 2)  # (B, N, D)
        tokens = tokens + self.time_embed(t.unsqueeze(-1)).unsqueeze(1)
        feats = self.vit_backbone.forward_features(tokens)  # (B, N, D)
        global_feat = feats.mean(1)
        eps = self.eps_head(global_feat).view(B, C, 32, 32)
        if H > 32:
            eps = F.interpolate(eps, size=(H, W), mode="bilinear")
        logvar = self.logvar_head(global_feat).view(B, 1, 4, 4)
        return eps, logvar

class LightweightUNetDraftL(nn.Module):
    """Local Draft network – lightweight UNet"""
    def __init__(self, in_channels: int = 4, base_channels: int = 128, time_embed_dim: int = 256):
        super().__init__()
        # Time embedding
        self.time_emb = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        # Blocks
        self.enc1 = self._blk(in_channels, base_channels)
        self.enc2 = self._blk(base_channels, base_channels * 2)
        self.enc3 = self._blk(base_channels * 2, base_channels * 4)
        self.enc4 = self._blk(base_channels * 4, base_channels * 4)
        self.mid = self._blk(base_channels * 4, base_channels * 4)
        self.dec4 = self._blk(base_channels * 8, base_channels * 4)
        self.dec3 = self._blk(base_channels * 6, base_channels * 2)
        self.dec2 = self._blk(base_channels * 3, base_channels)
        self.dec1 = self._blk(base_channels * 2, base_channels)
        self.out_proj = nn.Conv2d(base_channels, in_channels, 3, padding=1)
        # Per-level time projections
        self.t_proj = nn.ModuleList([
            nn.Linear(time_embed_dim, base_channels),
            nn.Linear(time_embed_dim, base_channels * 2),
            nn.Linear(time_embed_dim, base_channels * 4),
            nn.Linear(time_embed_dim, base_channels * 4),
        ])

    @staticmethod
    def _blk(inp, out):
        return nn.Sequential(
            nn.Conv2d(inp, out, 3, padding=1),
            nn.GroupNorm(8, out),
            nn.SiLU(),
            nn.Conv2d(out, out, 3, padding=1),
            nn.GroupNorm(8, out),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        t_emb = self.time_emb(t.unsqueeze(-1))
        e1 = self.enc1(x) + self.t_proj[0](t_emb).unsqueeze(-1).unsqueeze(-1)
        e2 = self.enc2(F.avg_pool2d(e1, 2)) + self.t_proj[1](t_emb).unsqueeze(-1).unsqueeze(-1)
        e3 = self.enc3(F.avg_pool2d(e2, 2)) + self.t_proj[2](t_emb).unsqueeze(-1).unsqueeze(-1)
        e4 = self.enc4(F.avg_pool2d(e3, 2)) + self.t_proj[3](t_emb).unsqueeze(-1).unsqueeze(-1)
        m = self.mid(e4)
        d4 = self.dec4(torch.cat([m, e4], 1))
        d3 = self.dec3(torch.cat([F.interpolate(d4, scale_factor=2, mode="bilinear"), e3], 1))
        d2 = self.dec2(torch.cat([F.interpolate(d3, scale_factor=2, mode="bilinear"), e2], 1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, scale_factor=2, mode="bilinear"), e1], 1))
        return self.out_proj(d1), None

class PAMSSModel(nn.Module):
    """Dual-path drafts + gating + teacher"""
    def __init__(self, device: str = 'cuda'):
        super().__init__()
        self.device = device
        self.draft_g = ViTDraftG().to(device)
        self.draft_l = LightweightUNetDraftL().to(device)
        self.gate = nn.Sequential(
            nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, 1), nn.Sigmoid()
        ).to(device)
        # Teacher pipeline (Stable Diffusion v1.5)
        self.teacher = StableDiffusionPipeline.from_pretrained(
            'stable-diffusion-v1-5/stable-diffusion-v1-5', torch_dtype=torch.float16,
            use_auth_token=os.getenv('HF_TOKEN')
        ).to(device)
        self.teacher_unet = self.teacher.unet
        self.vae = self.teacher.vae
        self.text_encoder = self.teacher.text_encoder
        self.tokenizer = self.teacher.tokenizer
        import clip
        self.clip, _ = clip.load('ViT-B/32', device=device)
        # Hyper-params
        self.kappa = 0.25
        self.delta = 0.01

    def compute_saliency(self, x_t: torch.Tensor, txt: torch.Tensor):
        x_t.requires_grad_(True)
        img = self.vae.decode(x_t / 0.18215).sample
        img_r = F.interpolate(img, size=(224, 224), mode='bilinear')
        im_feat = self.clip.encode_image(img_r)
        loss = F.cosine_similarity(im_feat, txt).sum()
        loss.backward(retain_graph=True)
        w = x_t.grad.mean((2, 3), keepdim=True)
        cam = (w * x_t).sum(1, keepdim=True).relu()
        return cam

    def fuse(self, eps_g: torch.Tensor, eps_l: torch.Tensor, t: torch.Tensor):
        g = self.gate(t.unsqueeze(-1)).unsqueeze(-1).unsqueeze(-1)
        return g * eps_g + (1 - g) * eps_l

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        txt: torch.Tensor,
        use_saliency: bool = True,
        use_early_exit: bool = True,
    ):
        eps_g, logvar = self.draft_g(x_t, t)
        eps_l, _ = self.draft_l(x_t, t)
        eps = self.fuse(eps_g, eps_l, t)
        if use_saliency:
            sal = self.compute_saliency(x_t, txt)
            m_low = sal.mean((2, 3)) < self.kappa
        else:
            m_low = torch.zeros(x_t.size(0), dtype=torch.bool, device=self.device)
        if (~m_low).any():
            with torch.no_grad():
                eps_teacher = self.teacher_unet(
                    x_t[~m_low], t[~m_low], encoder_hidden_states=txt[~m_low]
                ).sample
            eps[~m_low] = eps_teacher
        return eps, logvar, m_low

def train_pamss(model: PAMSSModel, train_loader: DataLoader, val_loader: DataLoader, cfg: Dict, device: str = 'cuda'):
    opt = torch.optim.AdamW([
        {'params': model.draft_g.parameters()},
        {'params': model.draft_l.parameters()},
        {'params': model.gate.parameters(), 'lr': cfg['learning_rate'] * 0.1},
    ], lr=cfg['learning_rate'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg['num_epochs'])
    history = {'train_losses': [], 'val_losses': [], 'teacher_call_rates': []}
    for ep in range(cfg['num_epochs']):
        model.train()
        t_loss, calls = 0.0, 0.0
        for batch in tqdm(train_loader, desc=f'Train {ep+1}/{cfg["num_epochs"]}'):
            img = batch['images'].to(device)
            prom = batch['prompts']
            txt_in = model.tokenizer(prom, padding=True, truncation=True, return_tensors='pt').to(device)
            txt_emb = model.text_encoder(txt_in.input_ids)[0]
            noise = torch.randn_like(img)
            t = torch.rand(img.size(0), device=device)
            x_t = img + noise * t[:, None, None, None]
            eps, _, m = model(x_t, t, txt_emb)
            loss = F.mse_loss(eps, noise)
            opt.zero_grad(); loss.backward(); opt.step()
            t_loss += loss.item(); calls += (~m).float().mean().item()
        sched.step()
        model.eval(); v_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                img = batch['images'].to(device)
                prom = batch['prompts']
                txt = model.tokenizer(prom, padding=True, truncation=True, return_tensors='pt').to(device)
                txt_emb = model.text_encoder(txt.input_ids)[0]
                noise = torch.randn_like(img)
                t = torch.rand(img.size(0), device=device)
                x_t = img + noise * t[:, None, None, None]
                eps, _, _ = model(x_t, t, txt_emb)
                v_loss += F.mse_loss(eps, noise).item()
        history['train_losses'].append(t_loss / len(train_loader))
        history['val_losses'].append(v_loss / len(val_loader))
        history['teacher_call_rates'].append(calls / len(train_loader))
        logger.info(
            f"Epoch {ep+1}: train {history['train_losses'][-1]:.4f}  val {history['val_losses'][-1]:.4f}  teacher {history['teacher_call_rates'][-1]:.2%}"
        )
    return model, history

def save_model(model: PAMSSModel, path: str):
    p = Path(path); p.mkdir(parents=True, exist_ok=True)
    torch.save(model.draft_g.state_dict(), p / 'draft_g.pth')
    torch.save(model.draft_l.state_dict(), p / 'draft_l.pth')
    torch.save(model.gate.state_dict(), p / 'gate.pth')
    logger.info(f'Model saved to {p}')