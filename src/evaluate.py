import torch
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json, logging
from typing import Dict, List
from scipy import linalg
import clip

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FID:
    def __init__(self, device='cuda'):
        from torchvision.models import inception_v3
        self.inception = inception_v3(pretrained=True, transform_input=False).to(device)
        self.inception.eval(); self.device = device
        self.resize = transforms.Resize((299, 299))
        self.norm = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    def _feat(self, imgs: torch.Tensor):
        x = torch.stack([self.norm(self.resize(i)) for i in imgs]).to(self.device)
        with torch.no_grad():
            x = self.inception(x).detach()
        return x.cpu().numpy()
    def score(self, real: np.ndarray, gen: np.ndarray):
        mu1, s1 = real.mean(0), np.cov(real, rowvar=False)
        mu2, s2 = gen.mean(0), np.cov(gen, rowvar=False)
        diff = mu1 - mu2
        covmean, _ = linalg.sqrtm(s1 @ s2, disp=False)
        if np.iscomplexobj(covmean): covmean = covmean.real
        return diff @ diff + np.trace(s1 + s2 - 2 * covmean)

class CLIPScore:
    def __init__(self, device='cuda'):
        self.model, self.prep = clip.load('ViT-B/32', device=device); self.device = device
    def score(self, imgs, texts):
        imgs_t = torch.stack([self.prep(i) for i in imgs]).to(self.device)
        with torch.no_grad():
            im_f = self.model.encode_image(imgs_t)
            txt_f = self.model.encode_text(clip.tokenize(texts).to(self.device))
        im_f = im_f / im_f.norm(dim=-1, keepdim=True); txt_f = txt_f / txt_f.norm(dim=-1, keepdim=True)
        return (im_f * txt_f).sum(-1).cpu().numpy()

def evaluate_pamss(model, loader, cfg, device='cuda'):
    model.eval()
    fid = FID(device); clip_score = CLIPScore(device)
    real_f, gen_f, clip_s, t_rates = [], [], [], []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc='Eval')):
            img = batch['images'].to(device)
            prom = batch['prompts']
            txt = model.tokenizer(prom, padding=True, truncation=True, return_tensors='pt').to(device)
            txt_emb = model.text_encoder(txt.input_ids)[0]
            B = img.size(0)
            lat = torch.randn(B, 4, 64, 64, device=device)
            calls = 0
            for s in range(cfg['num_inference_steps']):
                t = torch.ones(B, device=device) * (1 - s/cfg['num_inference_steps'])
                eps, _, m = model(lat, t, txt_emb)
                a = 1 - t[:, None, None, None]
                lat = (lat - eps * a) / (1 - a + 1e-8)
                calls += (~m).sum().item()
            gen = model.vae.decode(lat / 0.18215).sample.clamp(-1,1)
            real_f.append(fid._feat(img)); gen_f.append(fid._feat(gen))
            clip_s.extend(clip_score.score([transforms.ToPILImage()(g.add(1).div(2).cpu()) for g in gen], prom))
            t_rates.append(calls / (B*cfg['num_inference_steps']))
            if i>=10: break  # smoke-test shorten
    real_f = np.concatenate(real_f,0); gen_f = np.concatenate(gen_f,0)
    result = {
        'fid_scores': float(fid.score(real_f, gen_f)),
        'avg_clip_score': float(np.mean(clip_s)),
        'avg_teacher_rate': float(np.mean(t_rates)),
    }
    print(json.dumps(result, indent=2))
    return result

def _save_json(d: Dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path,'w') as f: json.dump(d,f,indent=2)
    print(json.dumps(d, indent=2))

def save_results_json(res, path):
    _save_json(res, Path(path))

def plot_results(*args, **kwargs):
    # Stub – plotting handled elsewhere
    pass