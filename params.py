import argparse
import time
import os

parser = argparse.ArgumentParser(description="Script for running LTspice simulations.")

parser.add_argument("--name", type=str, default="run", help="Provide a test name")
parser.add_argument("--no_cuda", action="store_true", default=False, help="Enables CUDA training")
parser.add_argument("--ltspice", type=str, default=os.path.join("C:", "LTspice"), help="Installed folder of LTspice")
parser.add_argument("--semiconductor", type=str, default="complete", help="single, star, complete")
parser.add_argument("--mutual", type=str, default="equiv", help="equiv or none")
parser.add_argument("--seed", type=int, default=15, help="Random seed.")
parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train")
parser.add_argument("--batch", type=int, default=3000, help="Batch size")
parser.add_argument("--train_test", type=float, default=0.3, help="train:test ratio e.g., 0.3 means 0.7:train, 0.3:test")

args = parser.parse_args()

# Generate a unique name with the current date and time
args.name = f"{args.name}_{time.strftime('%d_%m_%Y')}_{time.strftime('%H:%M:%S')}"

# Create dataset name based on semiconductor and mutual arguments
args.dataset = f"dataset_{args.semiconductor}__mutual_{args.mutual}"

if __name__ == '__main__':
    print(args)