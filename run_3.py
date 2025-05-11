import os
import torch
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == '__main__':

    import importlib
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1" #prevents "CUDA error: unspecified launch failure" and is recommended for some illegal memory access errors #increases train time by ~15%

    os.system("python attack_comparison.py --max_batchsize=200 --learning_rate=5.0 --beta=1.0 --dataset=imagenet --model=Salman2020Do_R50 --epsilon_l1=25 --max_iterations=300 --attack_types exp_attack_l1")
    os.system("python attack_comparison.py --max_batchsize=200 --learning_rate=5.0 --beta=1.0 --dataset=imagenet --model=Salman2020Do_R50 --epsilon_l1=50 --max_iterations=100 --attack_types exp_attack_l1")
    os.system("python attack_comparison.py --max_batchsize=200 --learning_rate=5.0 --beta=1.0 --dataset=imagenet --model=Salman2020Do_R50 --epsilon_l1=50 --max_iterations=300 --attack_types exp_attack_l1")
    os.system("python attack_comparison.py --max_batchsize=200 --learning_rate=5.0 --beta=1.0 --dataset=imagenet --model=Salman2020Do_R50 --epsilon_l1=75 --max_iterations=100 --attack_types exp_attack_l1")
    os.system("python attack_comparison.py --max_batchsize=200 --learning_rate=5.0 --beta=1.0 --dataset=imagenet --model=Salman2020Do_R50 --epsilon_l1=75 --max_iterations=300 --attack_types exp_attack_l1")

    os.system("python attack_comparison.py --max_batchsize=200 --learning_rate=0.5 --beta=2.0 --track_distance=True --dataset=imagenet --model=Salman2020Do_R50 --epsilon_l1=100 --max_iterations=100 --attack_types exp_attack_L1_rule_higher_beta")
    #os.system("python attack_comparison.py --learning_rate=0.5 --beta=2.0 --track_distance=True --dataset=imagenet --model=Salman2020Do_R50 --epsilon_l1=100 --max_iterations=300 --attack_types exp_attack_L1_rule_higher_beta")

    os.system("python attack_comparison.py --max_batchsize=200 --learning_rate=0.1 --beta=0.2 --track_distance=True --dataset=imagenet --model=standard --epsilon_l1=100 --max_iterations=100 --attack_types exp_attack")
    #os.system("python attack_comparison.py --learning_rate=0.1 --beta=0.2 --track_distance=True --dataset=imagenet --model=standard --epsilon_l1=100 --max_iterations=300 --attack_types exp_attack")
