import argparse
import os
import json
import pandas as pd

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exact-metrics-dir", required=True)
    p.add_argument("--dkl-metrics-dir",   required=True)
    p.add_argument("--output-dir",    required=True)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    exact = json.load(open(os.path.join(args.exact_metrics_dir, "metrics.json")))
    dkl   = json.load(open(os.path.join(args.dkl_metrics_dir, "metrics.json")))

    df = pd.DataFrame([exact,dkl], index=["ExactGP", "DKL"])
    csv = os.path.join(args.output_dir,"comparison.csv")
    df.to_csv(csv)
    print(f"[evaluate] saved → {csv}")

if __name__=="__main__":
    main()
