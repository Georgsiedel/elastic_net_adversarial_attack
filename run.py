import os
import torch
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == '__main__':


    import importlib
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1" #prevents "CUDA error: unspecified launch failure" and is recommended for some illegal memory access errors #increases train time by ~15%

    os.system("python attack_comparison.py --max_batchsize=50 --verbose=True --samplesize_accuracy=100 --samplesize_attack=50 --dataset=cifar10 --model=MainiAVG --epsilon_l1=50 --epsilon_l0_pixel=50 --max_iterations=100 --attack_types pgd_l0 sigma_zero GSE_attack sparse_rs_blackbox exp_attack_l1 SLIDE")
#