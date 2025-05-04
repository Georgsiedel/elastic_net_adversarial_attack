import os
import torch
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == '__main__':

    import importlib
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1" #prevents "CUDA error: unspecified launch failure" and is recommended for some illegal memory access errors #increases train time by ~15%

    os.system("python hyperparameter_sweep.py --track_c=True --hyperparameter_range1 0.1 0.2 0.5 --hyperparameter_range2 0.5 1.0 --dataset=imagenet --model=standard --attack_type exp_attack --epsilon_l1=12 --max_iterations=100")

    os.system("python hyperparameter_sweep.py --track_c=True --hyperparameter_range1 0.1 0.2 0.5 --hyperparameter_range2 0.2 0.5 1.0 --dataset=imagenet --model=vgg19 --attack_type exp_attack --epsilon_l1=12 --max_iterations=100")

    os.system("python hyperparameter_sweep.py --track_c=True --hyperparameter_range1 0.5 1.0 2.0 --hyperparameter_range2 1.0 2.0 5.0 --dataset=imagenet --model=ConvNext_iso_CvSt_revisiting --attack_type exp_attack --epsilon_l1=75 --max_iterations=100")

    os.system("python hyperparameter_sweep.py --track_c=True --hyperparameter_range1 0.5 1.0 2.0 --hyperparameter_range2 1.0 2.0 5.0 --dataset=imagenet --model=DVCE_R50 --attack_type exp_attack --epsilon_l1=255 --max_iterations=100")

    os.system("python hyperparameter_sweep.py --track_c=True --hyperparameter_range1 0.5 1.0 2.0 --hyperparameter_range2 1.0 2.0 5.0 --dataset=imagenet --model=Salman2020Do_R50 --attack_type exp_attack --epsilon_l1=12 --max_iterations=100")


    os.system("python attack_comparison.py --verbose=True --max_iterations=30 --dataset=imagenet --model=ConvNext_iso_CvSt_revisiting --epsilon_l1=75 --samplesize_attack=100 --attack_types pointwise_blackbox square_l1_blackbox sparse_rs_custom_L1_blackbox geoda_blackbox")
