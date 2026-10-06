"""Step 2 - train the LSTM (Adam, lr 0.005, 256 hidden units as in the paper) with early stopping."""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from musicgen.data import load_processed, make_windows
from musicgen.models import MusicLSTM
from musicgen.plotting import plot_loss


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_epoch(model, loader, device, opt=None):
    model.train(opt is not None); tot = n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(opt is not None):
            loss, _, _ = model.loss(x, y)
        if opt is not None:
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        tot += loss.item() * len(x); n += len(x)
    return tot / n


def train(out="outputs", epochs=40, lr=0.005, hidden=256, batch=64, patience=8, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    D = load_processed(out); device = get_device(); out = Path(out)
    print(f"Device: {device}")
    Xtr, Ytr = make_windows(D["splits"]["train"]); Xva, Yva = make_windows(D["splits"]["val"], stride=64)
    tl = DataLoader(TensorDataset(Xtr, Ytr), batch_size=batch, shuffle=True)
    vl = DataLoader(TensorDataset(Xva, Yva), batch_size=256)
    print(f"train windows {len(Xtr)}, val windows {len(Xva)}")
    model = MusicLSTM(len(D["chord_vocab"]), hidden=hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    hist = {"train": [], "val": []}; best, best_state, bad = 1e9, None, 0; t0 = time.time()
    for ep in range(1, epochs + 1):
        tr = run_epoch(model, tl, device, opt); va = run_epoch(model, vl, device)
        hist["train"].append(tr); hist["val"].append(va)
        flag = ""
        if va < best:
            best, bad, flag = va, 0, " *"; hist["best_epoch"] = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        print(f"epoch {ep:3d}  train {tr:.4f}  val {va:.4f}{flag}  ({time.time() - t0:.0f}s)")
        if bad >= patience:
            print("early stopping"); break
    model.load_state_dict(best_state)
    hist.update(best_val=best, epochs_run=len(hist["train"]), train_seconds=time.time() - t0, device=str(device))
    torch.save({"state_dict": best_state, "cfg": model.cfg, "chord_vocab": D["chord_vocab"],
                "starts": D["starts"]}, out / "lstm.pt")
    (out / "results").mkdir(exist_ok=True)
    (out / "results" / "history.json").write_text(json.dumps(hist, indent=2))
    plot_loss(hist, out / "results" / "loss_curve.png")
    return hist


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs"); ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=0.005); ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--batch", type=int, default=64); ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    train(a.out, a.epochs, a.lr, a.hidden, a.batch, a.patience, a.seed)
