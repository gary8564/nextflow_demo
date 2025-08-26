import argparse
import os
import numpy as np
import h5py
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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
    
    # 3. Save to HDF5 (language-agnostic format)
    hdf5_file = os.path.join(args.output_dir, "data.h5")
    with h5py.File(hdf5_file, 'w') as f:
        # Save datasets
        f.create_dataset('train_X', data=X_train.astype(np.float32))
        f.create_dataset('train_y', data=y_train.astype(np.float32))
        f.create_dataset('test_X', data=X_test.astype(np.float32))
        f.create_dataset('test_y', data=y_test.astype(np.float32))
        
        # Save standardization parameters
        f.create_dataset('scaler_mean', data=scaler.mean_.astype(np.float32))
        f.create_dataset('scaler_scale', data=scaler.scale_.astype(np.float32))
        
        # Save the dataset info as attributes
        f.attrs['n_samples_train'] = len(X_train)
        f.attrs['n_samples_test'] = len(X_test)
        f.attrs['n_features'] = X_train.shape[1]

    print(f"[preprocess] wrote HDF5 data → {hdf5_file}")

if __name__=="__main__":
    main()




