import argparse
import utils
import adversarial_attack.attack_utils as attack_utils
import json
import torch

def main(dataset, samplesize_accuracy, samplesize_attack, dataset_root, model, model_norm, hyperparameter, hyperparameter_range, 
         attack_type, epsilon_l1, epsilon_l2, eps_iter, norm, max_iterations, batchsize, save_images, verbose):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset
    xtest, ytest = utils.load_dataset(dataset=dataset, dataset_split=samplesize_accuracy, root=dataset_root)

    # Load model
    net, art_net, fb_net, alias = utils.get_model(dataset=dataset, modelname=model, norm=model_norm)

    # calculate accuracy, select a subset from the correctly classified images
    correct_map = utils.test_accuracy(net, xtest, ytest)
    xtest, ytest = utils.subset(correct_map, xtest, ytest, attack_samples=samplesize_attack)

    # Experiment setup
    Experiment = attack_utils.Experiment_class(
        art_net, fb_net, net, xtest, ytest, alias,
        epsilon_l1=epsilon_l1,
        epsilon_l2=epsilon_l2,
        eps_iter=eps_iter,
        norm=norm,
        max_iterations=max_iterations,
        batchsize=batchsize,
        save_images=save_images,
        verbose=verbose
    )

    # Hyperparameter sweep
    results_dict_hyperparameter_sweep = Experiment.hyperparameter_sweep(
        hyperparameter=hyperparameter,
        range=hyperparameter_range,
        attack_type=attack_type
    )

    json_file_path = f'./data/hyperparameter_sweep_{hyperparameter}_{attack_type}_{alias}_{samplesize_attack}samples_l1-epsilon-{epsilon_l1}.json'
    with open(json_file_path, 'w') as f:
        json.dump(results_dict_hyperparameter_sweep, f, indent=4)
    print(f'Evaluation results are saved under "{json_file_path}".')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter Sweep Script")
    parser.add_argument('--dataset', type=str, default='imagenet', choices=['cifar10', 'imagenet'],
                        help="Dataset to use")
    parser.add_argument('--samplesize_accuracy', type=int, default=10000, help="Split size for test accuracy evaluation")
    parser.add_argument('--samplesize_attack', type=int, default=500, help="Split size for attack evaluation")
    parser.add_argument('--dataset_root', type=str, default='../data', help="data folder relative root")
    parser.add_argument('--model', type=str, default='Salman2020Do_R50',
                        help="Model name (e.g., standard, ViT_revisiting, Salman2020Do_R50, corruption_robust, MainiAVG, etc.)")
    parser.add_argument('--model_norm', type=str, default='Linf',
                        help="Attack Norm the selected model was trained with. Only necessary if you load robustbench models")
    parser.add_argument('--hyperparameter', type=str, default='learning_rate', help="Hyperparameter to sweep")
    parser.add_argument('--hyperparameter_range', type=float, nargs='+', default=[1.0,3.0],#0.0001,0.001,0.01,0.1,1.0,0.025
                        help="Range of hyperparameter values (space-separated)")
    parser.add_argument('--attack_type', type=str, default='exp_attack_l1',
                        help="Type of attack for the hyperparameter sweep")
    parser.add_argument('--epsilon_l1', type=float, default=12, help="L1 norm epsilon (default: 12 for CIFAR10, 75 otherwise)")
    parser.add_argument('--epsilon_l2', type=float, default=0.5, help="L2 norm epsilon")
    parser.add_argument('--eps_iter', type=float, default=0.1, help="Step size for manual iterative attacks")
    parser.add_argument('--attack_norm', type=int, default=1, choices=[1, 2, float('inf')],
                        help="Attack norm type (1, 2, float('inf'))")
    parser.add_argument('--max_iterations', type=int, default=300, help="Maximum iterations for attacks")
    parser.add_argument('--batchsize', type=int, default=1, help="Batchsize to run every adversarial attack on")
    parser.add_argument('--save_images', type=int, default=1, help="Integer > 0: number of saved images per attack, 0: do not save)")
    parser.add_argument('--verbose', type=bool, default=False, help="Verbose output")

    args = parser.parse_args()
    main(
        args.dataset, args.samplesize_accuracy, args.samplesize_attack, args.dataset_root, args.model, args.model_norm, args.hyperparameter, 
        args.hyperparameter_range,  args.attack_type, args.epsilon_l1, args.epsilon_l2, args.eps_iter, args.attack_norm, args.max_iterations, 
        args.batchsize, args.save_images, args.verbose
    )