import os
import torch
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == '__main__':

    import importlib
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1" #prevents "CUDA error: unspecified launch failure" and is recommended for some illegal memory access errors #increases train time by ~15%

    os.system("python attack_comparison.py --learning_rate=0.5 --beta=1.0 --track_distance=True --dataset=imagenet --model=ConvNext_iso_CvSt_revisiting --epsilon_l1=100 --max_iterations=100 --attack_types exp_attack")
    #os.system("python attack_comparison.py --learning_rate=0.5 --beta=1.0 --track_distance=True --dataset=imagenet --model=ConvNext_iso_CvSt_revisiting --epsilon_l1=100 --max_iterations=300 --attack_types exp_attack")

    os.system("python attack_comparison.py --learning_rate=0.2 --beta=2.0 --track_distance=True --dataset=imagenet --model=ConvNext_iso_CvSt_revisiting --epsilon_l1=75 --max_iterations=100 --attack_types exp_attack_L1_rule_higher_beta")
    #os.system("python attack_comparison.py --learning_rate=0.2 --beta=2.0 --track_distance=True --dataset=imagenet --model=ConvNext_iso_CvSt_revisiting --epsilon_l1=75 --max_iterations=300 --attack_types exp_attack_L1_rule_higher_beta")
