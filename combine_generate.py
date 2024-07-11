import json
import pandas as pd
import argparse

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='out/data/Aura-SPPO-Iter1')
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--numgpu", type=int, default=1)
    return parser.parse_args()

def main():
    args = parse_arguments()

    for j in range(args.pairs):
        results = []
        for i in range(args.numgpu):
            file_path = f"{args.output_dir}/responses_{i}_{j}.json"
            print(f'Reading from {file_path}')
            with open(file_path) as f:
                gen = json.load(f)
                results += gen

        output_path = f"{args.output_dir}/responses_{j}.json"
        print(f'Saved to {output_path}')
        with open(output_path, "w") as f:
            json.dump(results, f)

if __name__ == "__main__":
    main()
