import argparse
import json
import numpy as np
import torch

from stf_ggru import STF_GGRU

def mk_win(X, Y, th=12, ph=1):
    xs, ys = [], []
    T = X.shape[0]
    for i in range(th, T - ph + 1):
        xs.append(X[i-th:i])      # [th,N,F]
        ys.append(Y[i+ph-1])      # [N]
    return np.stack(xs, 0), np.stack(ys, 0)

def ld_npz(p):
    z = np.load(p)
    return z["x"].astype(np.float32), z["y"].astype(np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", type=str, required=True)
    ap.add_argument("--npz", type=str, required=True)
    ap.add_argument("--th", type=int, default=12)
    ap.add_argument("--ph", type=int, default=1)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"

    X, Y = ld_npz(args.npz)
    xs, ys = mk_win(X, Y, th=args.th, ph=args.ph)

    r2 = int(0.85 * xs.shape[0])
    x_te = torch.tensor(xs[r2:], device=dev)
    y_te = torch.tensor(ys[r2:], device=dev)

    ck = torch.load(args.pt, map_location=dev)
    cfg = ck.get("cfg", {})
    f = ck.get("f", {"t":1,"s":1,"i":1,"k":1,"c":1})

    m = cfg.get("m", {"h":64,"k":8,"a":0.5,"dp":0.3})
    fi = x_te.shape[-1]

    net = STF_GGRU(
        f=fi,
        h=int(m["h"]),
        k=int(m["k"]),
        w=float(m["a"]),
        dp=float(m["dp"]),
        f_t=f["t"], f_s=f["s"], f_i=f["i"], f_k=f["k"], f_c=f["c"]
    ).to(dev)

    net.load_state_dict(ck["net"])
    net.eval()

    with torch.no_grad():
        yh = net(x_te)

    y0 = y_te.detach().cpu().numpy()
    y1 = yh.detach().cpu().numpy()

    rm = float(np.sqrt(np.mean((y0 - y1) ** 2)))
    ma = float(np.mean(np.abs(y0 - y1)))
    mp = float(np.mean(np.abs((y0 - y1) / (y0 + 1e-8))) * 100.0)

    out = {"RMSE": rm, "MAE": ma, "MAPE": mp, "flags": f}
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
