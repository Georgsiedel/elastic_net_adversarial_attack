import os
import torch
#os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
device = "cuda" if torch.cuda.is_available() else "cpu"
if __name__ == '__main__':
    import importlib
    os.system("python attack_comparison.py  --dataset_root ../datasets --dataset=cifar10 --max_batchsize=250 --model=standard  --max_iterations=1000 --epsilon_l1=24 --epsilon_l0=24 --attack_types exp_attack_sparse --learning_rate=1.0 --beta=0.01 --verbose=True")
