import os
import torch
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == '__main__':

    import importlib
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1" #prevents "CUDA error: unspecified launch failure" and is recommended for some illegal memory access errors #increases train time by ~15%


    os.system("python hyperparameter_sweep.py --track_c=True --hyperparameter_range1 0.2 0.5 1.0 --hyperparameter_range2 0.5 5.0 --dataset=imagenet --model=DVCE_R50 --attack_type exp_attack --epsilon_l1=255 --max_iterations=100")

    os.system("python hyperparameter_sweep.py --track_c=True --hyperparameter_range1 0.2 0.5 1.0 --hyperparameter_range2 0.5 1.0 2.0 --dataset=imagenet --model=ConvNext_iso_CvSt_revisiting --attack_type exp_attack_L1_rule_higher_beta --epsilon_l1=75 --max_iterations=100")
    os.system("python hyperparameter_sweep.py --track_c=True --hyperparameter_range1 0.2 0.5 1.0 --hyperparameter_range2 0.5 1.0 2.0 --dataset=imagenet --model=DVCE_R50 --attack_type exp_attack_L1_rule_higher_beta --epsilon_l1=255 --max_iterations=100")

    os.system("python attack_comparison.py --learning_rate=0.2 --beta=1.0 --dataset=imagenet --model=standard --epsilon_l1=12 --max_iterations=100 --attack_types exp_attack_l1_linf")
    os.system("python attack_comparison.py --learning_rate=0.2 --beta=1.0 --dataset=imagenet --model=standard --epsilon_l1=12 --max_iterations=300 --attack_types exp_attack_l1_linf")
    os.system("python attack_comparison.py --learning_rate=0.2 --beta=1.0 --dataset=imagenet --model=standard --epsilon_l1=25 --max_iterations=100 --attack_types exp_attack_l1_linf")
    os.system("python attack_comparison.py --learning_rate=0.2 --beta=1.0 --dataset=imagenet --model=standard --epsilon_l1=25 --max_iterations=300 --attack_types exp_attack_l1_linf")
    os.system("python attack_comparison.py --learning_rate=0.2 --beta=1.0 --dataset=imagenet --model=standard --epsilon_l1=50 --max_iterations=100 --attack_types exp_attack_l1_linf")
    os.system("python attack_comparison.py --learning_rate=0.2 --beta=1.0 --dataset=imagenet --model=standard --epsilon_l1=50 --max_iterations=300 --attack_types exp_attack_l1_linf")
    os.system("python attack_comparison.py --learning_rate=0.2 --beta=1.0 --dataset=imagenet --model=standard --epsilon_l1=75 --max_iterations=100 --attack_types exp_attack_l1_linf")
    os.system("python attack_comparison.py --learning_rate=0.2 --beta=1.0 --dataset=imagenet --model=standard --epsilon_l1=75 --max_iterations=300 --attack_types exp_attack_l1_linf")

    os.system("python attack_comparison.py --learning_rate=0.1 --beta=0.2 --dataset=imagenet --model=standard --epsilon_l1=100 --max_iterations=100 --attack_types exp_attack")
    os.system("python attack_comparison.py --learning_rate=0.1 --beta=0.2 --dataset=imagenet --model=standard --epsilon_l1=100 --max_iterations=300 --attack_types exp_attack")
