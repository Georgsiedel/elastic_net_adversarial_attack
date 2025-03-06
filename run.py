import os
import torch
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == '__main__':

    import importlib
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1" #prevents "CUDA error: unspecified launch failure" and is recommended for some illegal memory access errors #increases train time by ~15%
    
    #os.system("python hyperparameter_sweep.py --hyperparameter=beta --hyperparameter_range 0.5 1.5 2.5 3.0 5.0 7.0 10.0 20.0 --model=ConvNext_iso_CvSt_revisiting --dataset=imagenet --epsilon_l1=50")
    
    os.system("python attack_comparison.py --dataset=imagenet --model=standard --beta=3.0 --learning_rate=1.25 --attack_types apgd_art brendel_bethge exp_attack_l1")

    os.system("python attack_comparison.py --dataset=imagenet --model=Salman2020Do_R50 --beta=7.0 --learning_rate=2.25 --attack_types apgd_art custom_apgd AutoAttack brendel_bethge exp_attack_l1")
    
    os.system("python attack_comparison.py --dataset=imagenet --model=ConvNext_iso_CvSt_revisiting --epsilon_l1=50 --beta=5.0 --learning_rate=1.5")

    os.system("python attack_comparison.py --dataset=imagenet --model=ViT_revisiting --epsilon_l1=50 --beta=15.0 --learning_rate=2.25")
