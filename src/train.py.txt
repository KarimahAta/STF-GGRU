import os, argparse, yaml
import numpy as np
import torch
import torch.nn as nn

from stf_ggru import STF_GGRU
from utils import s0, d0, l0, j0

def yml(p):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def mk_win(X, Y, th=12, ph=1):
    xs, ys = [], []
    T = X.shape[0]
    for i in range(th, T - ph + 1):
        xs.append(X[i-th:i])        # [th,N,F]
        ys.append(Y[i+ph-1])        # [N]
    return np.stack(xs, 0), np.stack(ys, 0)

def ld_npz(p):
    z = np.load(p)
    return z["x"].astype(np.float32), z["y"].astype(np.float32)

def run(cfg, f):
    s0(int(cfg.get("seed", 1)))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    l0(f"dev={dev}")

    d = cfg["d"]; m = cfg["m"]; t = cfg["t"]; w = cfg["w"]
    outd = cfg.get("out", "results/run")
    d0(outd)

    # data
    p = os.path.join(d["path"], cfg.get("npz", "data.npz"))
    if not os.path.exists(p):
        raise FileNotFoundError(f"missing npz: {p}")

    X, Y = ld_npz(p)                           # X:[T,N,F], Y:[T,N]
    xs, ys = mk_win(X, Y, th=int(w["h"]), ph=int(w["p"]))

    B = xs.shape[0]
    r1 = int(0.7 * B); r2 = int(0.85 * B)

    x_tr, y_tr = xs[:r1], ys[:r1]
    x_va, y_va = xs[r1:r2], ys[r1:r2]
    x_te, y_te = xs[r2:], ys[r2:]

    # torch
    x_tr = torch.tensor(x_tr, device=dev)
    y_tr = torch.tensor(y_tr, device=dev)
    x_va = torch.tensor(x_va, device=dev)
    y_va = torch.tensor(y_va, device=dev)
    x_te = torch.tensor(x_te, device=dev)
    y_te = torch.tensor(y_te, device=dev)

    fi = x_tr.shape[-1]
    net = STF_GGRU(
        f=fi, h=int(m["h"]), k=int(m["k"]), w=float(m["a"]), dp=float(m["dp"]),
        f_t=f["t"], f_s=f["s"], f_i=f["i"], f_k=f["k"], f_c=f["c"]
    ).to(dev)

    opt = torch.optim.Adam(net.parameters(), lr=float(t["lr"]), weight_decay=float(t.get("wd", 0.0)))
    loss = nn.MSELoss()

    ep = int(t["ep"]); bs = int(t["bs"]); es = int(t.get("es", 15))
    best = 1e18; bad = 0
    ck = os.path.join(outd, "best.pt")

    def val():
        net.eval()
        with torch.no_grad():
            yh = net(x_va)
            v = loss(yh, y_va).item()
        net.train()
        return v

    for e in range(1, ep + 1):
        ix = torch.randperm(x_tr.size(0), device=dev)
        Ls = []

        for i in range(0, x_tr.size(0), bs):
            j = ix[i:i+bs]
            xb = x_tr[j]; yb = y_tr[j]
            opt.zero_grad()
            yh = net(xb)
            l = loss(yh, yb)
            l.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            Ls.append(l.item())

        lv = val()
        lt = float(np.mean(Ls))
        l0(f"ep={e:03d} tr={lt:.4f} va={lv:.4f}")

        if lv < best:
            best = lv; bad = 0
            torch.save({"net": net.state_dict(), "cfg": cfg, "f": f}, ck)
        else:
            bad += 1
            if bad >= es:
                l0("early_stop")
                break

    # test
    z = torch.load(ck, map_location=dev)
    net.load_state_dict(z["net"])
    net.eval()
    with torch.no_grad():
        yh = net(x_te)

    y0 = y_te.detach().cpu().numpy()
    y1 = yh.detach().cpu().numpy()

    rm = float(np.sqrt(np.mean((y0 - y1) ** 2)))
    ma = float(np.mean(np.abs(y0 - y1)))
    mp = float(np.mean(np.abs((y0 - y1) / (y0 + 1e-8))) * 100.0)

    res = {"RMSE": rm, "MAE": ma, "MAPE": mp, "best_va": float(best), "flags": f}
    j0(os.path.join(outd, "metrics.json"), res)
    l0(f"TEST rmse={rm:.4f} mae={ma:.4f} mape={mp:.2f}%")
    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, default="configs/pemsd4.yaml")
    ap.add_argument("--out", type=str, default="results/base")
    ap.add_argument("--npz", type=str, default="data.npz")

    ap.add_argument("--no_t", action="store_true")
    ap.add_argument("--no_s", action="store_true")
    ap.add_argument("--no_i", action="store_true")
    ap.add_argument("--no_k", action="store_true")
    ap.add_argument("--no_c", action="store_true")

    a = ap.parse_args()
    cfg = yml(a.cfg)
    cfg["out"] = a.out
    cfg["npz"] = a.npz

    f = {
        "t": 0 if a.no_t else 1,
        "s": 0 if a.no_s else 1,
        "i": 0 if a.no_i else 1,
        "k": 0 if a.no_k else 1,
        "c": 0 if a.no_c else 1,
    }

    run(cfg, f)

if __name__ == "__main__":
    main()
