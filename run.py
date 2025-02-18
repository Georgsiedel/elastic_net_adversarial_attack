import os
import torch
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == '__main__':
    import importlib

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1" #prevents "CUDA error: unspecified launch failure" and is recommended for some illegal memory access errors #increases train time by ~5-15%

    os.system("python hyperparameter_sweep.py --dataset=cifar10 --samplesize_accuracy=1000 --samplesize_attack=100 --model=corruption_robust " \
    "--hyperparameter=max_iterations_sweep --hyperparameter_range 10 50 100 200 300 --attack_type=exp_attack_l1_blackbox --epsilon_l1=12 ")
    
    os.system("python hyperparameter_sweep.py --batchsize=100 --dataset=cifar10 --samplesize_accuracy=1000 --samplesize_attack=100 --model=corruption_robust " \
    "--hyperparameter=max_iterations_sweep --hyperparameter_range 1000 5000 10000 20000 30000 --attack_type=sparse_rs_blackbox --epsilon_l1=12 ")

    os.system("python hyperparameter_sweep.py --dataset=cifar10 --model=MainiAVG --hyperparameter=quantile " \
    "--hyperparameter_range 0.999 --attack_type=exp_attack_l1 --epsilon_l1=12 ")

    os.system("python hyperparameter_sweep.py --dataset=cifar10 --model=MainiAVG --hyperparameter=beta " \
    "--hyperparameter_range 5.0 6.0 7.0 8.0 10.0 15.0 20.0 --attack_type=exp_attack_l1 --epsilon_l1=12 ")

    os.system("python hyperparameter_sweep.py --dataset=imagenet --model=standard --hyperparameter=beta " \
    "--hyperparameter_range 2.5 --attack_type=exp_attack_l1 --epsilon_l1=12 ")