import os
import argparse
import numpy as np
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=str, required=True)
    ap.add_argument("--out", dest="out", type=str, required=True)
    ap.add_argument("--N", type=int, default=307)   # sensors
    args = ap.parse_args()

    df = pd.read_csv(args.inp)

    # ترتيب الأعمدة
    df = df.sort_values(["time", "entity_id"])

    # استخراج القيم
    times = df["time"].unique()
    sensors = sorted(df["entity_id"].unique())

    T = len(times)
    N = args.N
    F = 3  # flow, occupancy, speed

    sid_map = {s: i for i, s in enumerate(sensors)}
    tid_map = {t: i for i, t in enumerate(times)}

    X = np.zeros((T, N, F), dtype=np.float32)
    M = np.zeros((T, N), dtype=np.float32)

    for _, r in df.iterrows():
        t = tid_map[r["time"]]
        s = sid_map[r["entity_id"]]
        if s >= N:
            continue

        X[t, s, 0] = r["traffic_flow"]
        X[t, s, 1] = r["traffic_occupancy"]
        X[t, s, 2] = r["traffic_speed"]
        M[t, s] = 1.0

    # forward fill للقيم الناقصة
    for s in range(N):
        last = None
        for t in range(T):
            if M[t, s] > 0:
                last = X[t, s].copy()
            elif last is not None:
                X[t, s] = last

    Y = X[:, :, 0]  # flow target

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, x=X, y=Y)

    print("DONE")
    print("x:", X.shape, "y:", Y.shape)
    print("saved to:", args.out)

if __name__ == "__main__":
    main()
