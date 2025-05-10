import os
import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Directory containing stats_*.npz files")
    args = parser.parse_args()

    DATA_DIR = args.data
    OUTPUT_FILE = os.path.join(DATA_DIR, "stats.npz")

    # List all the stats_*.npz files
    stat_files = [f for f in os.listdir(DATA_DIR) if f.startswith("stats_") and f.endswith(".npz")]

    if not stat_files:
        print("No stats_*.npz files found.")
        return

    total_mean = None
    total_mean2 = None
    count = 0

    for file in stat_files:
        path = os.path.join(DATA_DIR, file)
        data = np.load(path)

        MEAN = data["MEAN"]
        MEAN2 = data["MEAN2"]

        if total_mean is None:
            total_mean = np.zeros_like(MEAN)
            total_mean2 = np.zeros_like(MEAN2)

        total_mean += MEAN
        total_mean2 += MEAN2
        count += 1

    # Average over all classes
    final_mean = total_mean / count
    final_mean2 = total_mean2 / count

    np.savez(OUTPUT_FILE, MEAN=final_mean, MEAN2=final_mean2)
    print(f"Saved combined stats to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
