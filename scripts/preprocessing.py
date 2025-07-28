import argparse
import os
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir",  required=True,
                   help="Directory containing X.npy, y.npy")
    p.add_argument("--output-dir", required=True,
                   help="Where to write standardization params as scaler.pkl and input/output tensor as .pt files")
    p.add_argument("--test-size", type=float, default=0.15)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    X = np.load(os.path.join(args.input_dir,"X.npy"))
    y = np.load(os.path.join(args.input_dir,"y.npy"))

    # 1. Standardization
    dataset = np.hstack((X, y))
    scaler = StandardScaler()
    scaled_dataset = scaler.fit_transform(dataset)
    X_scaled, y_scaled = scaled_dataset[:,:-1], scaled_dataset[:,-1]

    # 2. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=args.test_size, shuffle=True)
    y_train = y_train.squeeze()
    y_test = y_test.squeeze()
    
    # 3. Convert to Tensor
    joblib.dump(scaler, os.path.join(args.output_dir,"scaler.pkl"))
    torch.save(torch.from_numpy(X_train).float(),
               os.path.join(args.output_dir,"train_X.pt"))
    torch.save(torch.from_numpy(y_train).float(),
               os.path.join(args.output_dir,"train_y.pt"))
    torch.save(torch.from_numpy(X_test).float(),
               os.path.join(args.output_dir,"test_X.pt"))
    torch.save(torch.from_numpy(y_test).float(),
               os.path.join(args.output_dir,"test_y.pt"))

    print(f"[preprocess] wrote tensors + scaler → {args.output_dir}")

if __name__=="__main__":
    main()




