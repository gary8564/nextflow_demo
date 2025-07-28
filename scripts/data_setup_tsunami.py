import argparse
import os
import numpy as np

def load_tsunami(data_root_folder):
    """Load tsunami files from the given root folder."""
    init_water_level = np.loadtxt(os.path.join(data_root_folder,
        "Initial_water_level_distibutions/deform4all.txt"), delimiter=",")
    X = init_water_level.copy()
    time_patterns = ["0100", "0250", "0500", "1200", "2400"]
    for pattern in time_patterns:
        tsunami_water_level = np.loadtxt(
                                os.path.join(data_root_folder,
                                    f"tsunami_water_level_disributions/z40{pattern}all_1d.txt"
                                ), delimiter=","
                              )
        X = np.hstack((X, tsunami_water_level))
    Y_all = np.loadtxt(os.path.join(data_root_folder,
        "Inundation_distibutions/zmax5depth.txt"), delimiter=",")
    # pick a representative station (median of per-sample argmax)
    loc = int(np.median(np.argmax(Y_all, axis=1)))
    y = Y_all[:, loc][:, None]
    return X, y

def main():
    # Parse arguments
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", required=True,
                   help="Directory containing extracted tsunami data")
    p.add_argument("--output-dir", required=True,
                   help="Store processed X.npy, y.npy")
    args = p.parse_args()
    
    # Check if input directory exists
    if not os.path.exists(args.input_dir):
        raise ValueError(f"Input directory does not exist: {args.input_dir}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load tsunami data
    X, y = load_tsunami(args.input_dir)
    
    # Save data
    np.save(os.path.join(args.output_dir, "X.npy"), X)
    np.save(os.path.join(args.output_dir, "y.npy"), y)
    print(f"""[data_setup_tsunami] saved 
          input → {args.output_dir}/X.npy (shape: {X.shape})
          output → {args.output_dir}/y.npy (shape: {y.shape})""")

if __name__ == "__main__":
    main() 