import os
import torch
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == '__main__':
    import importlib

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1" #prevents "CUDA error: unspecified launch failure" and is recommended for some illegal memory access errors #increases train time by ~15%

    os.system("python attack_comparison.py --batchsize=1 --verbose=False")

    os.system("python attack_comparison.py --batchsize=10 --verbose=False")

    os.system("python hyperparameter_sweep.py")