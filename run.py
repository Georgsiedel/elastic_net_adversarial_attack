import os
import torch
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == '__main__':

    import importlib
    #os.environ["CUDA_LAUNCH_BLOCKING"] = "1" #prevents "CUDA error: unspecified launch failure" and is recommended for some illegal memory access errors #increases train time by ~15%

    #os.system("python hyperparameter_sweep.py --hyperparameter=learning_rate --hyperparameter_range 0.75 1.5 --beta=0.05 --model=corruption_robust --epsilon_l1=4")

    #os.system("python attack_comparison.py --model=standard --learning_rate=1.0 --beta=0.01 --epsilon_l1=2 --attack_types brendel_bethge exp_attack_l1")

    os.system("python attack_comparison.py --dataset=imagenet --model=standard --learning_rate=1.25 --attack_types brendel_bethge")

    os.system("python attack_comparison.py --dataset=imagenet --model=Salman2020Do_R50 --beta=7.0 --learning_rate=2.25 --attack_types custom_apgd AutoAttack brendel_bethge")

    os.system("python attack_comparison.py --dataset=imagenet --model=ViT_revisiting --epsilon_l1=50 --beta=15.0 --learning_rate=2.25 --max_batchsize=20")
