import torch, random, json, numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
import logging, requests, zipfile, io
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SyntheticDataset(Dataset):
    def __init__(self, prompts, size=64):
        self.prompts = prompts; self.size = size
    def __len__(self): return len(self.prompts)
    def __getitem__(self, idx):
        return {'images': torch.randn(4, self.size, self.size), 'prompts': self.prompts[idx]}

def FaceTextPromptsDataset(n_face=100, n_text=100):
    face = [f'portrait photo of person {i}' for i in range(n_face)]
    text = [f'a sign saying TEXT {i}' for i in range(n_text)]
    return SyntheticDataset(face+text)

def create_data_loaders(cfg):
    if cfg.get('use_full_data', False):
        prompts = [f'laion prompt {i}' for i in range(cfg.get('laion_samples',1000))]
        ds = SyntheticDataset(prompts)
    else:
        ds = FaceTextPromptsDataset(100,100)
    l = len(ds); tr=int(0.7*l); va=int(0.15*l); te=l-tr-va
    tr_ds, va_ds, te_ds = random_split(ds, [tr,va,te])
    kw = dict(batch_size=cfg.get('batch_size',4), num_workers=cfg.get('num_workers',2), pin_memory=True)
    return DataLoader(tr_ds, shuffle=True, **kw), DataLoader(va_ds, **kw), DataLoader(te_ds, **kw)

def download_pretrained_models():
    logger.info('Pretrained models downloaded automatically by HF'); return True