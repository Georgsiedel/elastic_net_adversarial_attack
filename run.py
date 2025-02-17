import os
import torch
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == '__main__':
    import importlib

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1" #prevents "CUDA error: unspecified launch failure" and is recommended for some illegal memory access errors #increases train time by ~5-15%
    
    os.system("python hyperparameter_sweep.py --dataset=imagenet --model=Salman2020Do_R50 --hyperparameter=beta " \
          "--hyperparameter_range 6.0 20.0 --attack_type=exp_attack_l1 --epsilon_l1=12")

    os.system("python hyperparameter_sweep.py --dataset=imagenet --model=standard --hyperparameter=beta " \
    "--hyperparameter_range 3.0 5.0 6.0 7.0 8.0 10.0 15.0 20.0 1.0 2.0 --attack_type=exp_attack_l1 --epsilon_l1=12 ")

    os.system("python hyperparameter_sweep.py --dataset=imagenet --model=standard --hyperparameter=quantile " \
    "--hyperparameter_range 0.0 0.9 0.99 0.999 --attack_type=exp_attack_l1 --epsilon_l1=12 ")
