import argparse
import utils
import adversarial_attack.attack_utils as attack_utils
import json
import torch
import os
from itertools import product

def main(dataset, samplesize_accuracy, samplesize_attack, validation_run, dataset_root, model, model_norm, hyperparameter1, 
         hyperparameter2, range1, range2, attack_type, epsilon_l0_feature, epsilon_l0_pixel, epsilon_l1, epsilon_l2, eps_iter, 
         norm, max_iterations, max_batchsize, save_images, **kwargs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset
    xtest, ytest = utils.load_dataset(dataset=dataset, dataset_split=samplesize_accuracy, root=dataset_root)

    # Load model
    net, art_net, fb_net, alias = utils.get_model(dataset=dataset, modelname=model, norm=model_norm)

    # calculate accuracy, select a subset from the correctly classified images
    correct_map = utils.test_accuracy(net, xtest, ytest, batch_size=50)
    xtest, ytest = utils.subset(correct_map, xtest, ytest, attack_samples=samplesize_attack, valid=validation_run)

    # Experiment setup
    Experiment = attack_utils.Experiment_class(
        art_net, fb_net, net, xtest, ytest, alias,
        epsilon_l0_feature=epsilon_l0_feature,
        epsilon_l0_pixel=epsilon_l0_pixel,
        epsilon_l1=epsilon_l1,
        epsilon_l2=epsilon_l2,
        eps_iter=eps_iter,
        norm=norm,
        max_iterations=max_iterations,
        max_batchsize=max_batchsize,
        save_images=save_images
    )

    # Hyperparameter sweep
    #results_dict_hyperparameter_sweep = Experiment.hyperparameter_sweep(
   #     hyperparameter=hyperparameter,
    #    range=hyperparameter_range,
    #    attack_type=attack_type,
    #    **kwargs
    #)
#
#    json_file_path = f'./results/hyperparameter_sweep_{hyperparameter}_{attack_type}_{alias}_{samplesize_attack}samples_l1-epsilon-{epsilon_l1}_{max_iterations}_iters.json'
#    with open(json_file_path, 'w') as f:
#        json.dump(results_dict_hyperparameter_sweep, f, indent=4)
#    print(f'Evaluation results are saved under "{json_file_path}".')

        # Grid search over two hyperparameters
    grid_results = {}
    for val1, val2 in product(range1, range2):
        # set both hyperparameters in kwargs, overriding if present
        kwargs_copy = kwargs.copy()
        kwargs_copy[hyperparameter1] = val1
        kwargs_copy[hyperparameter2] = val2

        key = f"{hyperparameter1}_{val1}__{hyperparameter2}_{val2}"
        print(f"\t\t------------------ Running grid point: {hyperparameter1}={val1}, {hyperparameter2}={val2} -------------------\n")
        # Sweep over the first hyperparameter only (range of length 1)
        # effectively runs attack with both set
        results = Experiment.hyperparameter_sweep(
            attack_type=attack_type,
            **kwargs_copy
        )
        # results is a dict with one entry for val1; extract that
        grid_results[key] = results

    # Save full grid results
    out_path = f'./results/hyperparameter_grid_{hyperparameter1}_{hyperparameter2}_{attack_type}_{alias}_{samplesize_attack}samples-l1-epsilon-{epsilon_l1}_{max_iterations}_iters.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(grid_results, f, indent=4)
    print(f'Grid search results saved to {out_path}')

if __name__ == "__main__":
    #os.environ["CUDA_LAUNCH_BLOCKING"] = "1"#prevents "CUDA error: unspecified launch failure" and is recommended for some illegal memory access errors #increases train time by ~15%

    parser = argparse.ArgumentParser(description="Hyperparameter Sweep Script")
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'imagenet'],
                        help="Dataset to use")
    parser.add_argument('--samplesize_accuracy', type=int, default=10000, help="Split size for test accuracy evaluation")
    parser.add_argument('--samplesize_attack', type=int, default=500, help="Split size for attack evaluation")
    parser.add_argument('--validation_run', type=utils.str2bool, nargs='?', const=False, default=True, 
                        help="True for validation/tuning, False for testing. Selects attackset from the front or the back")
    parser.add_argument('--dataset_root', type=str, default='../data', help="data folder relative root")
    parser.add_argument('--model', type=str, default='standard',
                        help="Model name (e.g., standard, ViT_revisiting, ConvNext_iso_CvSt_revisiting, Salman2020Do_R50, corruption_robust, MainiAVG...)")
    parser.add_argument('--model_norm', type=str, default='Linf',
                        help="Attack Norm the selected model was trained with. Only necessary if you load robustbench models")
    parser.add_argument('--hyperparameter1', type=str, default='learning_rate', help="Hyperparameter to sweep")
    parser.add_argument('--hyperparameter_range1', type=float, nargs='+', default=[0.001,0.005,0.02,0.05],
                        help="Range of hyperparameter values (space-separated)")
    parser.add_argument('--hyperparameter2', type=str, default='beta', help="Hyperparameter to sweep")
    parser.add_argument('--hyperparameter_range2', type=float, nargs='+', default=[0.001,0.005,0.02,0.05],
                        help="Range of hyperparameter values (space-separated)")
    parser.add_argument('--attack_type', type=str, default='exp_attack_l1',
                        help="Type of attack for the hyperparameter sweep")
    parser.add_argument('--epsilon_l0_feature', type=int, default=25, help="L0 epsilon, translates to overall number of input features altered")
    parser.add_argument('--epsilon_l0_pixel', type=int, default=10, help="L0 epsilon, translates to overall number of pixels (grouped features along channels) altered")
    parser.add_argument('--epsilon_l1', type=float, default=2, help="L1 norm epsilon (default: 12 for CIFAR10, 75 otherwise)")
    parser.add_argument('--epsilon_l2', type=float, default=0.5, help="L2 norm epsilon")
    parser.add_argument('--eps_iter', type=float, default=0.1, help="Step size for manual iterative attacks")
    parser.add_argument('--learning_rate', type=float, help="Learning rate for exp attacks")
    parser.add_argument('--beta', type=float, help="Beta for exp_attack_l1, exp_attack, EAD")
    parser.add_argument('--attack_norm', type=int, default=1, choices=[1, 2, float('inf')],
                        help="Attack norm type (1, 2, float('inf'))")
    parser.add_argument('--max_iterations', type=int, default=100, help="Maximum iterations for attacks")
    parser.add_argument('--max_batchsize', type=int, default=50, help="Maximum Batchsize to run every adversarial attack on." \
                        "If attack is not optimized or not working with batches, will be returned by attacks.AdversarialAttacks class.")
    parser.add_argument('--save_images', type=int, default=1, help="Integer > 0: number of saved images per attack, 0: do not save)")
    parser.add_argument('--verbose', type=utils.str2bool, nargs='?', const=False, default=False, help="Verbose output")
    parser.add_argument('--track_c', type=utils.str2bool, nargs='?', const=False, default=False, help="Whether to track all images L1-distance")

    args = parser.parse_args()
    # Convert Namespace to dictionary and filter some arguments to kwargs
    filtered_kwargs = {"verbose", "beta", "learning_rate"}
    kwargs = {k: v for k, v in vars(args).items() if k in filtered_kwargs and v is not None}
   
    if args.track_c:
        dir = f"c_values_hyperparameter_sweep_{args.model}_{args.dataset}_{args.max_iterations}_iters"
    else:
        dir = None
    def add_argument_to_kwargs(kwargs, key, value):
        kwargs[key] = value
        return kwargs

    kwargs = add_argument_to_kwargs(kwargs, "track_c", dir)

    main(
        args.dataset, args.samplesize_accuracy, args.samplesize_attack, args.validation_run, args.dataset_root, args.model, 
        args.model_norm, args.hyperparameter1, args.hyperparameter2, args.hyperparameter_range1, args.hyperparameter_range2, 
        args.attack_type, args.epsilon_l0_feature, args.epsilon_l0_pixel, args.epsilon_l1, args.epsilon_l2, args.eps_iter, 
        args.attack_norm, args.max_iterations, args.max_batchsize, args.save_images, **kwargs
    )