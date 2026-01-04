import os, json, time, random
import numpy as np

def s0(s=1):
    random.seed(s)
    np.random.seed(s)

def d0(p):
    if p and (not os.path.exists(p)):
        os.makedirs(p, exist_ok=True)

def j0(p, o):
    d0(os.path.dirname(p))
    with open(p, "w", encoding="utf-8") as f:
        json.dump(o, f, indent=2)

def r0(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def t0():
    return time.strftime("%Y-%m-%d %H:%M:%S")

def l0(x):
    print(f"[{t0()}] {x}")

def z0(x, eps=1e-8):
    x = np.asarray(x, dtype=np.float32)
    m = x.mean(axis=0, keepdims=True)
    s = x.std(axis=0, keepdims=True) + eps
    return (x - m) / s, m, s

def rmse(y, yh, eps=1e-8):
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    yh = np.asarray(yh, dtype=np.float32).reshape(-1)
    return float(np.sqrt(np.mean((y - yh) ** 2) + eps))

def mae(y, yh):
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    yh = np.asarray(yh, dtype=np.float32).reshape(-1)
    return float(np.mean(np.abs(y - yh)))

def mape(y, yh, eps=1e-8):
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    yh = np.asarray(yh, dtype=np.float32).reshape(-1)
    return float(np.mean(np.abs((y - yh) / (y + eps))) * 100.0)

def w0(p, s):
    d0(os.path.dirname(p))
    with open(p, "w", encoding="utf-8") as f:
        f.write(s)

def rd0(p):
    with open(p, "r", encoding="utf-8") as f:
        return f.read()
