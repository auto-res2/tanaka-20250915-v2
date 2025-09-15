import argparse, yaml, torch, logging, sys, time, json
from pathlib import Path
from .train import PAMSSModel, train_pamss, save_model
from .evaluate import evaluate_pamss, plot_results, save_results_json
from .preprocess import create_data_loaders, download_pretrained_models

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

def _load_cfg(p):
    with open(p,'r') as f: return yaml.safe_load(f)

def _run(cfg, name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    download_pretrained_models()
    tr, va, te = create_data_loaders(cfg)
    model = PAMSSModel(device=device)
    if cfg.get('do_training', True):
        model, hist = train_pamss(model, tr, va, cfg, device)
        save_model(model, cfg['save_dir'])
        save_results_json(hist, Path(cfg['save_dir'])/ 'training.json')
    res = evaluate_pamss(model, te, cfg, device)
    save_results_json(res, Path(cfg['save_dir'])/ 'eval.json')
    plot_results(res, Path(cfg['save_dir'])/ 'figs')
    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke-test', action='store_true'); ap.add_argument('--full-experiment', action='store_true'); ap.add_argument('--config')
    ar = ap.parse_args()
    if ar.config:
        cfg_p, name = ar.config, 'custom'
    elif ar.smoke_test:
        cfg_p, name = 'config/smoke_test.yaml', 'smoke'
    elif ar.full_experiment:
        cfg_p, name = 'config/full_experiment.yaml', 'full'
    else:
        print('Specify --smoke-test or --full-experiment'); sys.exit(1)
    cfg = _load_cfg(cfg_p)
    _run(cfg, name)

if __name__ == '__main__':
    main()