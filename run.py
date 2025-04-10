import os
import torch
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == '__main__':

    import importlib
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1" #prevents "CUDA error: unspecified launch failure" and is recommended for some illegal memory access errors #increases train time by ~15%
    
    os.system("python hyperparameter_sweep.py --hyperparameter=learning_rate --hyperparameter_range 0.1 0.25 0.5 1.0 2.0 --dataset=cifar10 --model=standard --attack_type exp_attack_l1_blackbox --epsilon_l1=2")
    os.system("python hyperparameter_sweep.py --hyperparameter=learning_rate --hyperparameter_range 0.1 0.25 0.5 1.0 2.0 --dataset=cifar10 --model=corruption_robust --attack_type exp_attack_l1_blackbox --epsilon_l1=4")
    os.system("python hyperparameter_sweep.py --hyperparameter=learning_rate --hyperparameter_range 0.1 0.25 0.5 1.0 2.0 --dataset=cifar10 --model=CroceL1 --attack_type exp_attack_l1_blackbox --epsilon_l1=12")
    os.system("python hyperparameter_sweep.py --hyperparameter=learning_rate --hyperparameter_range 0.1 0.25 0.5 1.0 2.0 --dataset=cifar10 --model=MainiAVG --attack_type exp_attack_l1_blackbox --epsilon_l1=12")    
    os.system("python hyperparameter_sweep.py --hyperparameter=learning_rate --hyperparameter_range 0.1 0.25 0.5 1.0 2.0 --dataset=imagenet --model=standard --attack_type exp_attack_l1_blackbox --epsilon_l1=12")
    os.system("python hyperparameter_sweep.py --hyperparameter=learning_rate --hyperparameter_range 0.1 0.25 0.5 1.0 2.0 --dataset=imagenet --model=ConvNext_iso_CvSt_revisiting --attack_type exp_attack_l1_blackbox --epsilon_l1=75")

    os.system("python attack_comparison.py --max_batchsize=25 --dataset=imagenet --model=ViT_revisiting --epsilon_l1=255 --max_iterations=300 --attack_types ead_fb ead_fb_L1_rule_higher_beta")
    os.system("python attack_comparison.py --max_batchsize=25 --dataset=imagenet --model=ViT_revisiting --epsilon_l1=255 --max_iterations=500")
    os.system("python attack_comparison.py --verbose=True --max_iterations=30 --dataset=imagenet --model=ConvNext_iso_CvSt_revisiting --epsilon_l1=75 --samplesize_attack=100 --attack_types pointwise_blackbox square_l1_blackbox sparse_rs_custom_L1_blackbox geoda_blackbox")

    os.system("python attack_comparison.py --max_batchsize=100 --max_iterations=100 --dataset=cifar10 --model=standard --epsilon_l1=2 --attack_types exp_attack exp_attack_L1_rule_higher_beta exp_attack_l1 exp_attack_l1_linf")
    os.system("python attack_comparison.py --max_batchsize=100 --max_iterations=100 --dataset=cifar10 --model=standard --epsilon_l1=4 --attack_types exp_attack exp_attack_L1_rule_higher_beta exp_attack_l1 exp_attack_l1_linf")
    os.system("python attack_comparison.py --max_batchsize=100 --max_iterations=100 --dataset=cifar10 --model=standard --epsilon_l1=12 --attack_types exp_attack exp_attack_L1_rule_higher_beta exp_attack_l1 exp_attack_l1_linf")
    os.system("python attack_comparison.py --max_batchsize=100 --max_iterations=100 --dataset=cifar10 --model=standard --epsilon_l1=25 --attack_types exp_attack exp_attack_L1_rule_higher_beta exp_attack_l1 exp_attack_l1_linf")

    os.system("python attack_comparison.py --max_batchsize=100 --max_iterations=300 --dataset=cifar10 --model=standard --epsilon_l1=2 --attack_types exp_attack exp_attack_L1_rule_higher_beta exp_attack_l1 exp_attack_l1_linf")
    os.system("python attack_comparison.py --max_batchsize=100 --max_iterations=300 --dataset=cifar10 --model=standard --epsilon_l1=4 --attack_types exp_attack exp_attack_L1_rule_higher_beta exp_attack_l1 exp_attack_l1_linf")
    os.system("python attack_comparison.py --max_batchsize=100 --max_iterations=300 --dataset=cifar10 --model=standard --epsilon_l1=12 --attack_types exp_attack exp_attack_L1_rule_higher_beta exp_attack_l1 exp_attack_l1_linf")
    os.system("python attack_comparison.py --max_batchsize=100 --max_iterations=300 --dataset=cifar10 --model=standard --epsilon_l1=25 --attack_types exp_attack exp_attack_L1_rule_higher_beta exp_attack_l1 exp_attack_l1_linf")

    os.system("python attack_comparison.py --max_batchsize=100 --max_iterations=500 --dataset=cifar10 --model=standard --epsilon_l1=2 --attack_types exp_attack exp_attack_L1_rule_higher_beta exp_attack_l1 exp_attack_l1_linf")
    os.system("python attack_comparison.py --max_batchsize=100 --max_iterations=500 --dataset=cifar10 --model=standard --epsilon_l1=4 --attack_types exp_attack exp_attack_L1_rule_higher_beta exp_attack_l1 exp_attack_l1_linf")
    os.system("python attack_comparison.py --max_batchsize=100 --max_iterations=500 --dataset=cifar10 --model=standard --epsilon_l1=12 --attack_types exp_attack exp_attack_L1_rule_higher_beta exp_attack_l1 exp_attack_l1_linf")
    os.system("python attack_comparison.py --max_batchsize=100 --max_iterations=500 --dataset=cifar10 --model=standard --epsilon_l1=25 --attack_types exp_attack exp_attack_L1_rule_higher_beta exp_attack_l1 exp_attack_l1_linf")

    os.system("python attack_comparison.py --max_batchsize=100 --max_iterations=100 --dataset=cifar10 --model=MainiAVG --epsilon_l1=2 --attack_types exp_attack exp_attack_L1_rule_higher_beta exp_attack_l1 exp_attack_l1_linf")
    os.system("python attack_comparison.py --max_batchsize=100 --max_iterations=100 --dataset=cifar10 --model=MainiAVG --epsilon_l1=4 --attack_types exp_attack exp_attack_L1_rule_higher_beta exp_attack_l1 exp_attack_l1_linf")
    os.system("python attack_comparison.py --max_batchsize=100 --max_iterations=100 --dataset=cifar10 --model=MainiAVG --epsilon_l1=12 --attack_types exp_attack exp_attack_L1_rule_higher_beta exp_attack_l1 exp_attack_l1_linf")
    os.system("python attack_comparison.py --max_batchsize=100 --max_iterations=100 --dataset=cifar10 --model=MainiAVG --epsilon_l1=25 --attack_types exp_attack exp_attack_L1_rule_higher_beta exp_attack_l1 exp_attack_l1_linf")

    os.system("python attack_comparison.py --max_batchsize=100 --max_iterations=300 --dataset=cifar10 --model=MainiAVG --epsilon_l1=2 --attack_types exp_attack exp_attack_L1_rule_higher_beta exp_attack_l1 exp_attack_l1_linf")
    os.system("python attack_comparison.py --max_batchsize=100 --max_iterations=300 --dataset=cifar10 --model=MainiAVG --epsilon_l1=4 --attack_types exp_attack exp_attack_L1_rule_higher_beta exp_attack_l1 exp_attack_l1_linf")
    os.system("python attack_comparison.py --max_batchsize=100 --max_iterations=300 --dataset=cifar10 --model=MainiAVG --epsilon_l1=12 --attack_types exp_attack exp_attack_L1_rule_higher_beta exp_attack_l1 exp_attack_l1_linf")
    os.system("python attack_comparison.py --max_batchsize=100 --max_iterations=300 --dataset=cifar10 --model=MainiAVG --epsilon_l1=25 --attack_types exp_attack exp_attack_L1_rule_higher_beta exp_attack_l1 exp_attack_l1_linf")

    os.system("python attack_comparison.py --max_batchsize=100 --max_iterations=500 --dataset=cifar10 --model=MainiAVG --epsilon_l1=2 --attack_types exp_attack exp_attack_L1_rule_higher_beta exp_attack_l1 exp_attack_l1_linf")
    os.system("python attack_comparison.py --max_batchsize=100 --max_iterations=500 --dataset=cifar10 --model=MainiAVG --epsilon_l1=4 --attack_types exp_attack exp_attack_L1_rule_higher_beta exp_attack_l1 exp_attack_l1_linf")
    os.system("python attack_comparison.py --max_batchsize=100 --max_iterations=500 --dataset=cifar10 --model=MainiAVG --epsilon_l1=12 --attack_types exp_attack exp_attack_L1_rule_higher_beta exp_attack_l1 exp_attack_l1_linf")
    os.system("python attack_comparison.py --max_batchsize=100 --max_iterations=500 --dataset=cifar10 --model=MainiAVG --epsilon_l1=25 --attack_types exp_attack exp_attack_L1_rule_higher_beta exp_attack_l1 exp_attack_l1_linf")

    os.system("python attack_comparison.py --max_batchsize=100 --max_iterations=100 --dataset=cifar10 --model=CroceL1 --epsilon_l1=2 --attack_types exp_attack exp_attack_L1_rule_higher_beta exp_attack_l1 exp_attack_l1_linf")
    os.system("python attack_comparison.py --max_batchsize=100 --max_iterations=100 --dataset=cifar10 --model=CroceL1 --epsilon_l1=4 --attack_types exp_attack exp_attack_L1_rule_higher_beta exp_attack_l1 exp_attack_l1_linf")
    os.system("python attack_comparison.py --max_batchsize=100 --max_iterations=100 --dataset=cifar10 --model=CroceL1 --epsilon_l1=12 --attack_types exp_attack exp_attack_L1_rule_higher_beta exp_attack_l1 exp_attack_l1_linf")
    os.system("python attack_comparison.py --max_batchsize=100 --max_iterations=100 --dataset=cifar10 --model=CroceL1 --epsilon_l1=25 --attack_types exp_attack exp_attack_L1_rule_higher_beta exp_attack_l1 exp_attack_l1_linf")

    os.system("python attack_comparison.py --max_batchsize=100 --max_iterations=300 --dataset=cifar10 --model=CroceL1 --epsilon_l1=2 --attack_types exp_attack exp_attack_L1_rule_higher_beta exp_attack_l1 exp_attack_l1_linf")
    os.system("python attack_comparison.py --max_batchsize=100 --max_iterations=300 --dataset=cifar10 --model=CroceL1 --epsilon_l1=4 --attack_types exp_attack exp_attack_L1_rule_higher_beta exp_attack_l1 exp_attack_l1_linf")
    os.system("python attack_comparison.py --max_batchsize=100 --max_iterations=300 --dataset=cifar10 --model=CroceL1 --epsilon_l1=12 --attack_types exp_attack exp_attack_L1_rule_higher_beta exp_attack_l1 exp_attack_l1_linf")
    os.system("python attack_comparison.py --max_batchsize=100 --max_iterations=300 --dataset=cifar10 --model=CroceL1 --epsilon_l1=25 --attack_types exp_attack exp_attack_L1_rule_higher_beta exp_attack_l1 exp_attack_l1_linf")

    os.system("python attack_comparison.py --max_batchsize=100 --max_iterations=500 --dataset=cifar10 --model=CroceL1 --epsilon_l1=2 --attack_types exp_attack exp_attack_L1_rule_higher_beta exp_attack_l1 exp_attack_l1_linf")
    os.system("python attack_comparison.py --max_batchsize=100 --max_iterations=500 --dataset=cifar10 --model=CroceL1 --epsilon_l1=4 --attack_types exp_attack exp_attack_L1_rule_higher_beta exp_attack_l1 exp_attack_l1_linf")
    os.system("python attack_comparison.py --max_batchsize=100 --max_iterations=500 --dataset=cifar10 --model=CroceL1 --epsilon_l1=12 --attack_types exp_attack exp_attack_L1_rule_higher_beta exp_attack_l1 exp_attack_l1_linf")
    os.system("python attack_comparison.py --max_batchsize=100 --max_iterations=500 --dataset=cifar10 --model=CroceL1 --epsilon_l1=25 --attack_types exp_attack exp_attack_L1_rule_higher_beta exp_attack_l1 exp_attack_l1_linf")

    os.system("python attack_comparison.py --max_iterations=100 --dataset=imagenet --model=ConvNext_iso_CvSt_revisiting --epsilon_l1=12 --attack_types exp_attack exp_attack_L1_rule_higher_beta exp_attack_l1 exp_attack_l1_linf")
    os.system("python attack_comparison.py --max_iterations=100 --dataset=imagenet --model=ConvNext_iso_CvSt_revisiting --epsilon_l1=25 --attack_types exp_attack exp_attack_L1_rule_higher_beta exp_attack_l1 exp_attack_l1_linf")
    os.system("python attack_comparison.py --max_iterations=100 --dataset=imagenet --model=ConvNext_iso_CvSt_revisiting --epsilon_l1=50 --attack_types exp_attack exp_attack_L1_rule_higher_beta exp_attack_l1 exp_attack_l1_linf")
    os.system("python attack_comparison.py --max_iterations=100 --dataset=imagenet --model=ConvNext_iso_CvSt_revisiting --epsilon_l1=75 --attack_types exp_attack exp_attack_L1_rule_higher_beta exp_attack_l1 exp_attack_l1_linf")
    os.system("python attack_comparison.py --max_iterations=100 --dataset=imagenet --model=ConvNext_iso_CvSt_revisiting --epsilon_l1=255 --attack_types exp_attack exp_attack_L1_rule_higher_beta exp_attack_l1 exp_attack_l1_linf")

    os.system("python attack_comparison.py --max_batchsize=100 --max_iterations=300 --dataset=cifar10 --model=CroceL1 --epsilon_l1=2 --attack_types exp_attack exp_attack_L1_rule_higher_beta exp_attack_l1 exp_attack_l1_linf")
    os.system("python attack_comparison.py --max_batchsize=100 --max_iterations=300 --dataset=cifar10 --model=CroceL1 --epsilon_l1=4 --attack_types exp_attack exp_attack_L1_rule_higher_beta exp_attack_l1 exp_attack_l1_linf")
    os.system("python attack_comparison.py --max_batchsize=100 --max_iterations=300 --dataset=cifar10 --model=CroceL1 --epsilon_l1=12 --attack_types exp_attack exp_attack_L1_rule_higher_beta exp_attack_l1 exp_attack_l1_linf")
    os.system("python attack_comparison.py --max_batchsize=100 --max_iterations=300 --dataset=cifar10 --model=CroceL1 --epsilon_l1=25 --attack_types exp_attack exp_attack_L1_rule_higher_beta exp_attack_l1 exp_attack_l1_linf")

    os.system("python attack_comparison.py --max_batchsize=100 --max_iterations=500 --dataset=cifar10 --model=CroceL1 --epsilon_l1=2 --attack_types exp_attack exp_attack_L1_rule_higher_beta exp_attack_l1 exp_attack_l1_linf")
    os.system("python attack_comparison.py --max_batchsize=100 --max_iterations=500 --dataset=cifar10 --model=CroceL1 --epsilon_l1=4 --attack_types exp_attack exp_attack_L1_rule_higher_beta exp_attack_l1 exp_attack_l1_linf")
    os.system("python attack_comparison.py --max_batchsize=100 --max_iterations=500 --dataset=cifar10 --model=CroceL1 --epsilon_l1=12 --attack_types exp_attack exp_attack_L1_rule_higher_beta exp_attack_l1 exp_attack_l1_linf")
    os.system("python attack_comparison.py --max_batchsize=100 --max_iterations=500 --dataset=cifar10 --model=CroceL1 --epsilon_l1=25 --attack_types exp_attack exp_attack_L1_rule_higher_beta exp_attack_l1 exp_attack_l1_linf")

