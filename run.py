import os
import torch
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == '__main__':

    import importlib
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1" #prevents "CUDA error: unspecified launch failure" and is recommended for some illegal memory access errors #increases train time by ~15%

    os.system("python hyperparameter_sweep.py --hyperparameter=learning_rate --hyperparameter_range 0.025 0.1 0.25 1.0 1.5 2.0 3.0 5.0 --dataset=imagenet --model=standard --attack_type exp_attack_l1 --epsilon_l1=12 --max_iterations=100")
    os.system("python hyperparameter_sweep.py --hyperparameter=beta --hyperparameter_range 0.025 0.1 0.25 1.0 2.0 3.0 5.0 10.0 --dataset=imagenet --model=standard --attack_type exp_attack_l1 --epsilon_l1=12 --max_iterations=100")

    os.system("python hyperparameter_sweep.py --track_c=True --hyperparameter=learning_rate --hyperparameter_range 0.1 --dataset=imagenet --model=standard --attack_type ead_fb --epsilon_l1=12 --max_iterations=100")
    os.system("python hyperparameter_sweep.py --track_c=True --hyperparameter=learning_rate --hyperparameter_range 0.1 --dataset=imagenet --model=ConvNext_iso_CvSt_revisiting --attack_type ead_fb --epsilon_l1=12 --max_iterations=100")
    os.system("python hyperparameter_sweep.py --track_c=True --hyperparameter=learning_rate --hyperparameter_range 0.1 --dataset=imagenet --model=DVCE_R50 --attack_type ead_fb --epsilon_l1=12 --max_iterations=100")
    os.system("python hyperparameter_sweep.py --track_c=True --hyperparameter=learning_rate --hyperparameter_range 0.1 --dataset=imagenet --model=Salman2020Do_R50_Linf --attack_type ead_fb --epsilon_l1=12 --max_iterations=100")
    os.system("python hyperparameter_sweep.py --track_c=True --hyperparameter=learning_rate --hyperparameter_range 0.1 --dataset=cifar10 --model=standard --attack_type ead_fb --epsilon_l1=12 --max_iterations=100")
    os.system("python hyperparameter_sweep.py --track_c=True --hyperparameter=learning_rate --hyperparameter_range 0.1 --dataset=cifar10 --model=corruption_robust --attack_type ead_fb --epsilon_l1=12 --max_iterations=100")
    os.system("python hyperparameter_sweep.py --track_c=True --hyperparameter=learning_rate --hyperparameter_range 0.1 --dataset=cifar10 --model=MainiAVG --attack_type ead_fb --epsilon_l1=12 --max_iterations=100")
    os.system("python hyperparameter_sweep.py --track_c=True --hyperparameter=learning_rate --hyperparameter_range 0.1 --dataset=cifar10 --model=CroceL1 --attack_type ead_fb --epsilon_l1=12 --max_iterations=100")

    os.system("python hyperparameter_sweep.py --track_c=True --hyperparameter=learning_rate --hyperparameter_range 0.1 --dataset=imagenet --model=standard --attack_type ead_fb_L1_rule_higher_beta --epsilon_l1=12 --max_iterations=100")
    os.system("python hyperparameter_sweep.py --track_c=True --hyperparameter=learning_rate --hyperparameter_range 0.1 --dataset=imagenet --model=ConvNext_iso_CvSt_revisiting --attack_type ead_fb_L1_rule_higher_beta --epsilon_l1=12 --max_iterations=100")
    os.system("python hyperparameter_sweep.py --track_c=True --hyperparameter=learning_rate --hyperparameter_range 0.1 --dataset=imagenet --model=DVCE_R50 --attack_type ead_fb_L1_rule_higher_beta --epsilon_l1=12 --max_iterations=100")
    os.system("python hyperparameter_sweep.py --track_c=True --hyperparameter=learning_rate --hyperparameter_range 0.1 --dataset=imagenet --model=Salman2020Do_R50_Linf --attack_type ead_fb_L1_rule_higher_beta --epsilon_l1=12 --max_iterations=100")
    os.system("python hyperparameter_sweep.py --track_c=True --hyperparameter=learning_rate --hyperparameter_range 0.1 --dataset=cifar10 --model=standard --attack_type ead_fb_L1_rule_higher_beta --epsilon_l1=12 --max_iterations=100")
    os.system("python hyperparameter_sweep.py --track_c=True --hyperparameter=learning_rate --hyperparameter_range 0.1 --dataset=cifar10 --model=corruption_robust --attack_type ead_fb_L1_rule_higher_beta --epsilon_l1=12 --max_iterations=100")
    os.system("python hyperparameter_sweep.py --track_c=True --hyperparameter=learning_rate --hyperparameter_range 0.1 --dataset=cifar10 --model=MainiAVG --attack_type ead_fb_L1_rule_higher_beta --epsilon_l1=12 --max_iterations=100")
    os.system("python hyperparameter_sweep.py --track_c=True --hyperparameter=learning_rate --hyperparameter_range 0.1 --dataset=cifar10 --model=CroceL1 --attack_type ead_fb_L1_rule_higher_beta --epsilon_l1=12 --max_iterations=100")

    os.system("python hyperparameter_sweep.py --track_c=True --hyperparameter=learning_rate --hyperparameter_range 0.1 --dataset=imagenet --model=standard --attack_type ead_fb --epsilon_l1=12 --max_iterations=300")
    os.system("python hyperparameter_sweep.py --track_c=True --hyperparameter=learning_rate --hyperparameter_range 0.1 --dataset=imagenet --model=ConvNext_iso_CvSt_revisiting --attack_type ead_fb --epsilon_l1=12 --max_iterations=300")
    os.system("python hyperparameter_sweep.py --track_c=True --hyperparameter=learning_rate --hyperparameter_range 0.1 --dataset=imagenet --model=DVCE_R50 --attack_type ead_fb --epsilon_l1=12 --max_iterations=300")
    os.system("python hyperparameter_sweep.py --track_c=True --hyperparameter=learning_rate --hyperparameter_range 0.1 --dataset=imagenet --model=Salman2020Do_R50_Linf --attack_type ead_fb --epsilon_l1=12 --max_iterations=300")
    os.system("python hyperparameter_sweep.py --track_c=True --hyperparameter=learning_rate --hyperparameter_range 0.1 --dataset=cifar10 --model=standard --attack_type ead_fb --epsilon_l1=12 --max_iterations=300")
    os.system("python hyperparameter_sweep.py --track_c=True --hyperparameter=learning_rate --hyperparameter_range 0.1 --dataset=cifar10 --model=corruption_robust --attack_type ead_fb --epsilon_l1=12 --max_iterations=300")
    os.system("python hyperparameter_sweep.py --track_c=True --hyperparameter=learning_rate --hyperparameter_range 0.1 --dataset=cifar10 --model=MainiAVG --attack_type ead_fb --epsilon_l1=12 --max_iterations=300")
    os.system("python hyperparameter_sweep.py --track_c=True --hyperparameter=learning_rate --hyperparameter_range 0.1 --dataset=cifar10 --model=CroceL1 --attack_type ead_fb --epsilon_l1=12 --max_iterations=300")

    os.system("python hyperparameter_sweep.py --track_c=True --hyperparameter=learning_rate --hyperparameter_range 0.1 --dataset=imagenet --model=standard --attack_type ead_fb_L1_rule_higher_beta --epsilon_l1=12 --max_iterations=300")
    os.system("python hyperparameter_sweep.py --track_c=True --hyperparameter=learning_rate --hyperparameter_range 0.1 --dataset=imagenet --model=ConvNext_iso_CvSt_revisiting --attack_type ead_fb_L1_rule_higher_beta --epsilon_l1=12 --max_iterations=300")
    os.system("python hyperparameter_sweep.py --track_c=True --hyperparameter=learning_rate --hyperparameter_range 0.1 --dataset=imagenet --model=DVCE_R50 --attack_type ead_fb_L1_rule_higher_beta --epsilon_l1=12 --max_iterations=300")
    os.system("python hyperparameter_sweep.py --track_c=True --hyperparameter=learning_rate --hyperparameter_range 0.1 --dataset=imagenet --model=Salman2020Do_R50_Linf --attack_type ead_fb_L1_rule_higher_beta --epsilon_l1=12 --max_iterations=300")
    os.system("python hyperparameter_sweep.py --track_c=True --hyperparameter=learning_rate --hyperparameter_range 0.1 --dataset=cifar10 --model=standard --attack_type ead_fb_L1_rule_higher_beta --epsilon_l1=12 --max_iterations=300")
    os.system("python hyperparameter_sweep.py --track_c=True --hyperparameter=learning_rate --hyperparameter_range 0.1 --dataset=cifar10 --model=corruption_robust --attack_type ead_fb_L1_rule_higher_beta --epsilon_l1=12 --max_iterations=300")
    os.system("python hyperparameter_sweep.py --track_c=True --hyperparameter=learning_rate --hyperparameter_range 0.1 --dataset=cifar10 --model=MainiAVG --attack_type ead_fb_L1_rule_higher_beta --epsilon_l1=12 --max_iterations=300")
    os.system("python hyperparameter_sweep.py --track_c=True --hyperparameter=learning_rate --hyperparameter_range 0.1 --dataset=cifar10 --model=CroceL1 --attack_type ead_fb_L1_rule_higher_beta --epsilon_l1=12 --max_iterations=300")

    os.system("python attack_comparison.py --verbose=True --max_iterations=30 --dataset=imagenet --model=ConvNext_iso_CvSt_revisiting --epsilon_l1=75 --samplesize_attack=100 --attack_types pointwise_blackbox square_l1_blackbox sparse_rs_custom_L1_blackbox geoda_blackbox")
  
    os.system("python hyperparameter_sweep.py --samplesize_attack=100 --hyperparameter=learning_rate --hyperparameter_range 0.1 0.25 1.0 2.5 --dataset=cifar10 --model=MainiAVG --attack_type exp_attack_l1_blackbox --epsilon_l1=25 --max_iterations=300")
    os.system("python hyperparameter_sweep.py --samplesize_attack=100 --hyperparameter=beta --hyperparameter_range 0.001 0.01 0.1 1.0 5.0 --dataset=cifar10 --model=MainiAVG --attack_type exp_attack_l1_blackbox --epsilon_l1=25 --max_iterations=300")

    os.system("python hyperparameter_sweep.py --track_c=True --hyperparameter=learning_rate --hyperparameter_range 0.1 0.25 1.0 2.5 --dataset=cifar10 --model=MainiAVG --attack_type exp_attack --epsilon_l1=12 --max_iterations=100")
    os.system("python hyperparameter_sweep.py --track_c=True --hyperparameter=beta --hyperparameter_range 0.001 0.01 0.1 1.0 5.0 --dataset=cifar10 --model=MainiAVG --attack_type exp_attack --epsilon_l1=12 --max_iterations=100")

    os.system("python hyperparameter_sweep.py --samplesize_attack=100 --hyperparameter=learning_rate --hyperparameter_range 0.1 0.25 1.0 2.5 --dataset=cifar10 --model=standard --attack_type exp_attack_l1_blackbox --epsilon_l1=4 --max_iterations=300")
    os.system("python hyperparameter_sweep.py --samplesize_attack=100 --hyperparameter=beta --hyperparameter_range 0.001 0.01 0.1 1.0 5.0 --dataset=cifar10 --model=standard --attack_type exp_attack_l1_blackbox --epsilon_l1=4 --max_iterations=300")

    os.system("python hyperparameter_sweep.py --track_c=True --hyperparameter=learning_rate --hyperparameter_range 0.1 0.25 1.0 2.5 --dataset=cifar10 --model=standard --attack_type exp_attack --epsilon_l1=2 --max_iterations=100")
    os.system("python hyperparameter_sweep.py --track_c=True --hyperparameter=beta --hyperparameter_range 0.001 0.01 0.1 1.0 5.0 --dataset=cifar10 --model=standard --attack_type exp_attack --epsilon_l1=2 --max_iterations=100")

    os.system("python hyperparameter_sweep.py --samplesize_attack=100 --hyperparameter=learning_rate --hyperparameter_range 0.1 0.25 1.0 2.5 --dataset=imagenet --model=standard --attack_type exp_attack_l1_blackbox --epsilon_l1=25 --max_iterations=300")
    os.system("python hyperparameter_sweep.py --samplesize_attack=100 --hyperparameter=beta --hyperparameter_range 0.001 0.01 0.1 1.0 5.0 --dataset=imagenet --model=standard --attack_type exp_attack_l1_blackbox --epsilon_l1=25 --max_iterations=300")

    os.system("python hyperparameter_sweep.py --track_c=True --hyperparameter=learning_rate --hyperparameter_range 0.1 0.25 1.0 2.5 --dataset=imagenet --model=standard --attack_type exp_attack --epsilon_l1=12 --max_iterations=100")
    os.system("python hyperparameter_sweep.py --track_c=True --hyperparameter=beta --hyperparameter_range 0.001 0.01 0.1 1.0 5.0 --dataset=imagenet --model=standard --attack_type exp_attack --epsilon_l1=12 --max_iterations=100")

    os.system("python hyperparameter_sweep.py --samplesize_attack=100 --hyperparameter=learning_rate --hyperparameter_range 0.1 0.25 1.0 2.5 --dataset=imagenet --model=ConvNext_iso_CvSt_revisiting --attack_type exp_attack_l1_blackbox --epsilon_l1=75 --max_iterations=300")
    os.system("python hyperparameter_sweep.py --samplesize_attack=100 --hyperparameter=beta --hyperparameter_range 0.001 0.01 0.1 1.0 5.0 --dataset=imagenet --model=ConvNext_iso_CvSt_revisiting --attack_type exp_attack_l1_blackbox --epsilon_l1=75 --max_iterations=300")

    os.system("python hyperparameter_sweep.py --track_c=True --hyperparameter=learning_rate --hyperparameter_range 0.1 0.25 1.0 2.5 --dataset=imagenet --model=ConvNext_iso_CvSt_revisiting --attack_type exp_attack --epsilon_l1=50 --max_iterations=100")
    os.system("python hyperparameter_sweep.py --track_c=True --hyperparameter=beta --hyperparameter_range 0.001 0.01 0.1 1.0 5.0 --dataset=imagenet --model=ConvNext_iso_CvSt_revisiting --attack_type exp_attack --epsilon_l1=50 --max_iterations=100")

    os.system("python attack_comparison.py --track_distance=True --track_c=True --beta=7.0 --learning_rate=2.25 --max_iterations=100 --dataset=imagenet --model=Salman2020Do_R50 --epsilon_l1=12 --attack_types exp_attack_l1 exp_attack_l1_linf ead_fb ead_fb_L1_rule_higher_beta")
    os.system("python attack_comparison.py --beta=7.0 --learning_rate=2.25 --max_iterations=100 --dataset=imagenet --model=Salman2020Do_R50 --epsilon_l1=25 --attack_types exp_attack_l1 exp_attack_l1_linf")
    os.system("python attack_comparison.py --beta=7.0 --learning_rate=2.25 --max_iterations=100 --dataset=imagenet --model=Salman2020Do_R50 --epsilon_l1=50 --attack_types exp_attack_l1 exp_attack_l1_linf")
    os.system("python attack_comparison.py --beta=7.0 --learning_rate=2.25 --max_iterations=100 --dataset=imagenet --model=Salman2020Do_R50 --epsilon_l1=75 --attack_types exp_attack_l1 exp_attack_l1_linf")

    os.system("python attack_comparison.py --track_distance=True --track_c=True --beta=7.0 --learning_rate=2.25 --max_iterations=300 --dataset=imagenet --model=Salman2020Do_R50 --epsilon_l1=12 --attack_types exp_attack_l1 exp_attack_l1_linf ead_fb ead_fb_L1_rule_higher_beta")
    os.system("python attack_comparison.py --beta=7.0 --learning_rate=2.25 --max_iterations=300 --dataset=imagenet --model=Salman2020Do_R50 --epsilon_l1=25 --attack_types exp_attack_l1 exp_attack_l1_linf")
    os.system("python attack_comparison.py --beta=7.0 --learning_rate=2.25 --max_iterations=300 --dataset=imagenet --model=Salman2020Do_R50 --epsilon_l1=50 --attack_types exp_attack_l1 exp_attack_l1_linf")
    os.system("python attack_comparison.py --beta=7.0 --learning_rate=2.25 --max_iterations=300 --dataset=imagenet --model=Salman2020Do_R50 --epsilon_l1=75 --attack_types exp_attack_l1 exp_attack_l1_linf")

    os.system("python attack_comparison.py --track_distance=True --track_c=True --beta=7.0 --learning_rate=2.25 --max_iterations=500 --dataset=imagenet --model=Salman2020Do_R50 --epsilon_l1=12 --attack_types exp_attack_l1 exp_attack_l1_linf ead_fb ead_fb_L1_rule_higher_beta")
    os.system("python attack_comparison.py --beta=7.0 --learning_rate=2.25 --max_iterations=500 --dataset=imagenet --model=Salman2020Do_R50 --epsilon_l1=25 --attack_types exp_attack_l1 exp_attack_l1_linf")
    os.system("python attack_comparison.py --beta=7.0 --learning_rate=2.25 --max_iterations=500 --dataset=imagenet --model=Salman2020Do_R50 --epsilon_l1=50 --attack_types exp_attack_l1 exp_attack_l1_linf")
    os.system("python attack_comparison.py --beta=7.0 --learning_rate=2.25 --max_iterations=500 --dataset=imagenet --model=Salman2020Do_R50 --epsilon_l1=75 --attack_types exp_attack_l1 exp_attack_l1_linf")
