import argparse
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from algorithms.algorithms import get_algorithm_class
from configs.data_model_configs import get_dataset_class
from configs.params import get_hparams_class
from models.models import SpatioTemporalBackbone


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


def fix_randomness(seed: int):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _load_data_loader_module(project_root: Path):
    """
    Load utils/data_loader.py explicitly.
    The repository has both `utils.py` and `utils/`, so regular `import utils.data_loader`
    is ambiguous in Python.
    """
    import importlib.util

    module_path = project_root / "utils" / "data_loader.py"
    spec = importlib.util.spec_from_file_location("terse_data_loader", str(module_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_inertial_signals(har_root: Path, split: str) -> np.ndarray:
    signal_dir = har_root / split / "Inertial Signals"
    channel_files = [
        f"body_acc_x_{split}.txt",
        f"body_acc_y_{split}.txt",
        f"body_acc_z_{split}.txt",
        f"body_gyro_x_{split}.txt",
        f"body_gyro_y_{split}.txt",
        f"body_gyro_z_{split}.txt",
        f"total_acc_x_{split}.txt",
        f"total_acc_y_{split}.txt",
        f"total_acc_z_{split}.txt",
    ]

    channels = [np.loadtxt(signal_dir / f, dtype=np.float32) for f in channel_files]
    return np.stack(channels, axis=1)  # [N, 9, 128]


def _prepare_har_pt(raw_har_dir: Path, out_dir: Path, seed: int = 42, train_ratio: float = 0.8):
    """Create per-subject train/test .pt files expected by utils/data_loader.py."""
    out_dir.mkdir(parents=True, exist_ok=True)

    train_x = _read_inertial_signals(raw_har_dir, "train")
    test_x = _read_inertial_signals(raw_har_dir, "test")
    x_all = np.concatenate([train_x, test_x], axis=0)

    y_train = np.loadtxt(raw_har_dir / "train" / "y_train.txt", dtype=np.int64)
    y_test = np.loadtxt(raw_har_dir / "test" / "y_test.txt", dtype=np.int64)
    y_all = np.concatenate([y_train, y_test], axis=0) - 1  # to 0-based labels

    s_train = np.loadtxt(raw_har_dir / "train" / "subject_train.txt", dtype=np.int64)
    s_test = np.loadtxt(raw_har_dir / "test" / "subject_test.txt", dtype=np.int64)
    s_all = np.concatenate([s_train, s_test], axis=0)

    rng = np.random.default_rng(seed)
    subject_ids = sorted(np.unique(s_all).tolist())

    for sid in subject_ids:
        idx = np.where(s_all == sid)[0]
        rng.shuffle(idx)
        split_idx = max(1, int(len(idx) * train_ratio))
        if split_idx >= len(idx):
            split_idx = len(idx) - 1

        tr_idx, te_idx = idx[:split_idx], idx[split_idx:]
        if len(te_idx) == 0:
            te_idx = tr_idx[-1:]
            tr_idx = tr_idx[:-1]

        train_payload = {
            "samples": torch.from_numpy(x_all[tr_idx]).float(),
            "labels": torch.from_numpy(y_all[tr_idx]).long(),
        }
        test_payload = {
            "samples": torch.from_numpy(x_all[te_idx]).float(),
            "labels": torch.from_numpy(y_all[te_idx]).long(),
        }

        torch.save(train_payload, out_dir / f"train_{sid}.pt")
        torch.save(test_payload, out_dir / f"test_{sid}.pt")


@torch.no_grad()
def evaluate(algorithm, dataloader, device):
    algorithm.eval()
    preds, labels = [], []
    for x, y, _ in dataloader:
        x = x.float().to(device)
        y = y.long().to(device)
        _, feat = algorithm.feature_extractor(x)
        out = algorithm.classifier(feat)
        preds.append(out.argmax(dim=1).cpu())
        labels.append(y.cpu())

    pred = torch.cat(preds)
    true = torch.cat(labels)
    return (pred == true).float().mean().item() * 100.0


def build_logger():
    logger = logging.getLogger("terse-runner")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    return logger


def run(args):
    project_root = Path(__file__).resolve().parent
    data_loader_mod = _load_data_loader_module(project_root)

    fix_randomness(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    logger = build_logger()

    dataset_cfg = get_dataset_class(args.dataset)()
    hparams_cfg = get_hparams_class(args.dataset)()
    train_params = dict(hparams_cfg.train_params)
    alg_hparams = dict(hparams_cfg.alg_hparams[args.algorithm])
    hparams = {**train_params, **alg_hparams}
    hparams["num_epochs"] = args.epochs

    if args.dataset == "HAR":
        raw_har_dir = project_root / "data" / "UCI HAR Dataset"
        generated_dir = project_root / "data" / "processed_har_pt"
        if args.force_prepare or not (generated_dir / f"train_{args.source}.pt").exists():
            logger.info("Preparing HAR subject-wise .pt files ...")
            _prepare_har_pt(raw_har_dir, generated_dir, seed=args.seed)
        data_dir = generated_dir
    else:
        data_dir = Path(args.data_dir)

    source_loader = data_loader_mod.data_generator(str(data_dir), str(args.source), "train", dataset_cfg, hparams)
    target_train_loader = data_loader_mod.data_generator(str(data_dir), str(args.target), "train", dataset_cfg, hparams)
    target_test_loader = data_loader_mod.data_generator(str(data_dir), str(args.target), "test", dataset_cfg, hparams)

    algorithm_class = get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(SpatioTemporalBackbone, dataset_cfg, hparams, device).to(device)

    logger.info(f"Device: {device}")
    logger.info(f"Dataset: {args.dataset}, Algorithm: {args.algorithm}")
    logger.info(f"Source subject: {args.source}, Target subject: {args.target}")

    logger.info("\n=== Stage 1: Source pretraining ===")
    pre_avg = defaultdict(AverageMeter)
    pre_avg["Src_cls_loss"] = AverageMeter()
    src_state = algorithm.pretrain(source_loader, pre_avg, logger)
    algorithm.network.load_state_dict(src_state)

    pre_acc = evaluate(algorithm, target_test_loader, device)
    logger.info(f"Target test accuracy after pretraining: {pre_acc:.2f}%")

    logger.info("\n=== Stage 2: Source-free target adaptation ===")
    adapt_avg = defaultdict(AverageMeter)
    adapt_avg["entropy_loss"] = AverageMeter()
    last_state, best_state = algorithm.update(target_train_loader, adapt_avg, logger)

    chosen_state = best_state if args.use_best else last_state
    algorithm.network.load_state_dict(chosen_state)

    final_acc = evaluate(algorithm, target_test_loader, device)
    logger.info(f"Final target test accuracy: {final_acc:.2f}%")


def parse_args():
    parser = argparse.ArgumentParser(description="Train/evaluate TERSE from a single command.")
    parser.add_argument("--dataset", type=str, default="HAR", choices=["HAR", "WISDM", "EEG_EDF_ORI"])
    parser.add_argument("--algorithm", type=str, default="TERSE")
    parser.add_argument("--source", type=int, default=2, help="Source domain/subject id")
    parser.add_argument("--target", type=int, default=11, help="Target domain/subject id")
    parser.add_argument("--epochs", type=int, default=10, help="Epochs used for both stages")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    parser.add_argument("--use-best", action="store_true", help="Use best checkpoint returned by adaptation")
    parser.add_argument("--force-prepare", action="store_true", help="Regenerate HAR .pt files")
    parser.add_argument("--data-dir", type=str, default="data", help="Data dir for non-HAR datasets")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
