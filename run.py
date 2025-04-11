import os
import torch
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == '__main__':

    import importlib
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1" #prevents "CUDA error: unspecified launch failure" and is recommended for some illegal memory access errors #increases train time by ~15%

    #os.system("python attack_comparison.py --max_batchsize=100 --beta=0.01 --learning_rate=1.0 --max_iterations=100 --dataset=cifar10 --model=standard --epsilon_l1=25 --attack_types exp_attack_l1 exp_attack_l1_linf exp_attack exp_attack_L1_rule_higher_beta")

    os.system("python attack_comparison.py --track_distance=True --max_batchsize=100 --beta=0.5 --learning_rate=2.0 --max_iterations=100 --dataset=cifar10 --model=MainiAVG --epsilon_l1=12 --attack_types exp_attack_l1 exp_attack_l1_linf exp_attack exp_attack_L1_rule_higher_beta")
    os.system("python attack_comparison.py --track_distance=True--max_batchsize=100 --beta=0.01 --learning_rate=1.0 --max_iterations=100 --dataset=cifar10 --model=standard --epsilon_l1=4.0 --attack_types exp_attack_l1 exp_attack_l1_linf exp_attack exp_attack_L1_rule_higher_beta")

    os.system("python attack_comparison.py --track_distance=True --max_batchsize=100 --beta=0.5 --learning_rate=2.0 --max_iterations=100 --dataset=cifar10 --model=CroceL1 --epsilon_l1=12 --attack_types exp_attack_l1 exp_attack_l1_linf exp_attack exp_attack_L1_rule_higher_beta")

    os.system("python attack_comparison.py --beta=2.0 --learning_rate=1.25 --max_iterations=100 --dataset=imagenet --model=standard --epsilon_l1=12 --attack_types exp_attack_l1 exp_attack_l1_linf exp_attack exp_attack_L1_rule_higher_beta")
    os.system("python attack_comparison.py --beta=2.0 --learning_rate=1.25 --max_iterations=100 --dataset=imagenet --model=standard --epsilon_l1=25 --attack_types exp_attack_l1 exp_attack_l1_linf")
    os.system("python attack_comparison.py --beta=2.0 --learning_rate=1.25 --max_iterations=100 --dataset=imagenet --model=standard --epsilon_l1=50 --attack_types exp_attack_l1 exp_attack_l1_linf")
    os.system("python attack_comparison.py --beta=2.0 --learning_rate=1.25 --max_iterations=100 --dataset=imagenet --model=standard --epsilon_l1=75 --attack_types exp_attack_l1 exp_attack_l1_linf")

    os.system("python attack_comparison.py --beta=2.0 --learning_rate=1.25 --max_iterations=300 --dataset=imagenet --model=standard --epsilon_l1=12 --attack_types exp_attack_l1 exp_attack_l1_linf")
    os.system("python attack_comparison.py --beta=2.0 --learning_rate=1.25 --max_iterations=300 --dataset=imagenet --model=standard --epsilon_l1=25 --attack_types exp_attack_l1 exp_attack_l1_linf")
    os.system("python attack_comparison.py --beta=2.0 --learning_rate=1.25 --max_iterations=300 --dataset=imagenet --model=standard --epsilon_l1=50 --attack_types exp_attack_l1 exp_attack_l1_linf")
    os.system("python attack_comparison.py --beta=2.0 --learning_rate=1.25 --max_iterations=300 --dataset=imagenet --model=standard --epsilon_l1=75 --attack_types exp_attack_l1 exp_attack_l1_linf")
    
    os.system("python attack_comparison.py --beta=2.0 --learning_rate=1.25 --max_iterations=500 --dataset=imagenet --model=standard --epsilon_l1=12 --attack_types exp_attack_l1 exp_attack_l1_linf")
    os.system("python attack_comparison.py --beta=2.0 --learning_rate=1.25 --max_iterations=500 --dataset=imagenet --model=standard --epsilon_l1=25 --attack_types exp_attack_l1 exp_attack_l1_linf")
    os.system("python attack_comparison.py --beta=2.0 --learning_rate=1.25 --max_iterations=500 --dataset=imagenet --model=standard --epsilon_l1=50 --attack_types exp_attack_l1 exp_attack_l1_linf")
    os.system("python attack_comparison.py --beta=2.0 --learning_rate=1.25 --max_iterations=500 --dataset=imagenet --model=standard --epsilon_l1=75 --attack_types exp_attack_l1 exp_attack_l1_linf")

    os.system("python attack_comparison.py --beta=5.0 --learning_rate=1.5 --max_iterations=100 --dataset=imagenet --model=ConvNext_iso_CvSt_revisiting --epsilon_l1=12 --attack_types exp_attack_l1 exp_attack_l1_linf")
    os.system("python attack_comparison.py --beta=5.0 --learning_rate=1.5 --max_iterations=100 --dataset=imagenet --model=ConvNext_iso_CvSt_revisiting --epsilon_l1=25 --attack_types exp_attack_l1 exp_attack_l1_linf")
    os.system("python attack_comparison.py --beta=5.0 --learning_rate=1.5 --max_iterations=100 --dataset=imagenet --model=ConvNext_iso_CvSt_revisiting --epsilon_l1=50 --attack_types exp_attack_l1 exp_attack_l1_linf")
    os.system("python attack_comparison.py --beta=5.0 --learning_rate=1.5 --max_iterations=100 --dataset=imagenet --model=ConvNext_iso_CvSt_revisiting --epsilon_l1=75 --attack_types exp_attack_l1 exp_attack_l1_linf exp_attack exp_attack_L1_rule_higher_beta")

    os.system("python attack_comparison.py --beta=5.0 --learning_rate=1.5 --max_iterations=300 --dataset=imagenet --model=ConvNext_iso_CvSt_revisiting --epsilon_l1=12 --attack_types exp_attack_l1 exp_attack_l1_linf")
    os.system("python attack_comparison.py --beta=5.0 --learning_rate=1.5 --max_iterations=300 --dataset=imagenet --model=ConvNext_iso_CvSt_revisiting --epsilon_l1=25 --attack_types exp_attack_l1 exp_attack_l1_linf")
    os.system("python attack_comparison.py --beta=5.0 --learning_rate=1.5 --max_iterations=300 --dataset=imagenet --model=ConvNext_iso_CvSt_revisiting --epsilon_l1=50 --attack_types exp_attack_l1 exp_attack_l1_linf")
    os.system("python attack_comparison.py --beta=5.0 --learning_rate=1.5 --max_iterations=300 --dataset=imagenet --model=ConvNext_iso_CvSt_revisiting --epsilon_l1=75 --attack_types exp_attack_l1 exp_attack_l1_linf")
    
    os.system("python attack_comparison.py --beta=5.0 --learning_rate=1.5 --max_iterations=500 --dataset=imagenet --model=ConvNext_iso_CvSt_revisiting --epsilon_l1=12 --attack_types exp_attack_l1 exp_attack_l1_linf")
    os.system("python attack_comparison.py --beta=5.0 --learning_rate=1.5 --max_iterations=500 --dataset=imagenet --model=ConvNext_iso_CvSt_revisiting --epsilon_l1=25 --attack_types exp_attack_l1 exp_attack_l1_linf")
    os.system("python attack_comparison.py --beta=5.0 --learning_rate=1.5 --max_iterations=500 --dataset=imagenet --model=ConvNext_iso_CvSt_revisiting --epsilon_l1=50 --attack_types exp_attack_l1 exp_attack_l1_linf")
    os.system("python attack_comparison.py --beta=5.0 --learning_rate=1.5 --max_iterations=500 --dataset=imagenet --model=ConvNext_iso_CvSt_revisiting --epsilon_l1=75 --attack_types exp_attack_l1 exp_attack_l1_linf")

    os.system("python attack_comparison.py --beta=2.0 --learning_rate=2.0 --max_iterations=100 --dataset=imagenet --model=DVCE_R50 --epsilon_l1=12 --attack_types exp_attack_l1 exp_attack_l1_linf")
    os.system("python attack_comparison.py --beta=2.0 --learning_rate=2.0 --max_iterations=100 --dataset=imagenet --model=DVCE_R50 --epsilon_l1=25 --attack_types exp_attack_l1 exp_attack_l1_linf")
    os.system("python attack_comparison.py --beta=2.0 --learning_rate=2.0 --max_iterations=100 --dataset=imagenet --model=DVCE_R50 --epsilon_l1=50 --attack_types exp_attack_l1 exp_attack_l1_linf")
    os.system("python attack_comparison.py --beta=2.0 --learning_rate=2.0 --max_iterations=100 --dataset=imagenet --model=DVCE_R50 --epsilon_l1=75 --attack_types exp_attack_l1 exp_attack_l1_linf")
    os.system("python attack_comparison.py --beta=2.0 --learning_rate=2.0 --max_iterations=100 --dataset=imagenet --model=DVCE_R50 --epsilon_l1=255 --attack_types exp_attack_l1 exp_attack_l1_linf exp_attack exp_attack_L1_rule_higher_beta")

    os.system("python attack_comparison.py --beta=2.0 --learning_rate=2.0 --max_iterations=300 --dataset=imagenet --model=DVCE_R50 --epsilon_l1=12 --attack_types exp_attack_l1 exp_attack_l1_linf")
    os.system("python attack_comparison.py --beta=2.0 --learning_rate=2.0 --max_iterations=300 --dataset=imagenet --model=DVCE_R50 --epsilon_l1=25 --attack_types exp_attack_l1 exp_attack_l1_linf")
    os.system("python attack_comparison.py --beta=2.0 --learning_rate=2.0 --max_iterations=300 --dataset=imagenet --model=DVCE_R50 --epsilon_l1=50 --attack_types exp_attack_l1 exp_attack_l1_linf")
    os.system("python attack_comparison.py --beta=2.0 --learning_rate=2.0 --max_iterations=300 --dataset=imagenet --model=DVCE_R50 --epsilon_l1=75 --attack_types exp_attack_l1 exp_attack_l1_linf")
    os.system("python attack_comparison.py --beta=2.0 --learning_rate=2.0 --max_iterations=300 --dataset=imagenet --model=DVCE_R50 --epsilon_l1=255 --attack_types exp_attack_l1 exp_attack_l1_linf")

    os.system("python attack_comparison.py --beta=2.0 --learning_rate=2.0 --max_iterations=500 --dataset=imagenet --model=DVCE_R50 --epsilon_l1=12 --attack_types exp_attack_l1 exp_attack_l1_linf")
    os.system("python attack_comparison.py --beta=2.0 --learning_rate=2.0 --max_iterations=500 --dataset=imagenet --model=DVCE_R50 --epsilon_l1=25 --attack_types exp_attack_l1 exp_attack_l1_linf")
    os.system("python attack_comparison.py --beta=2.0 --learning_rate=2.0 --max_iterations=500 --dataset=imagenet --model=DVCE_R50 --epsilon_l1=50 --attack_types exp_attack_l1 exp_attack_l1_linf")
    os.system("python attack_comparison.py --beta=2.0 --learning_rate=2.0 --max_iterations=500 --dataset=imagenet --model=DVCE_R50 --epsilon_l1=75 --attack_types exp_attack_l1 exp_attack_l1_linf")
    os.system("python attack_comparison.py --beta=2.0 --learning_rate=2.0 --max_iterations=500 --dataset=imagenet --model=DVCE_R50 --epsilon_l1=255 --attack_types exp_attack_l1 exp_attack_l1_linf")

    os.system("python attack_comparison.py --max_batchsize=25 --dataset=imagenet --model=ViT_revisiting --epsilon_l1=255 --max_iterations=300 --attack_types ead_fb ead_fb_L1_rule_higher_beta")
    os.system("python attack_comparison.py --max_batchsize=25 --dataset=imagenet --model=ViT_revisiting --epsilon_l1=255 --max_iterations=500")
    os.system("python attack_comparison.py --verbose=True --max_iterations=30 --dataset=imagenet --model=ConvNext_iso_CvSt_revisiting --epsilon_l1=75 --samplesize_attack=100 --attack_types pointwise_blackbox square_l1_blackbox sparse_rs_custom_L1_blackbox geoda_blackbox")
