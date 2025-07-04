import argparse
import utils
import adversarial_attack.attack_utils as attack_utils
import json
import torch
import os

def main(dataset, samplesize_accuracy, samplesize_attack, validation_run, dataset_root, model, model_norm, attack_types, 
         epsilon_l0_feature, epsilon_l0_pixel, epsilon_l1, epsilon_l2, 
         eps_iter, norm, max_iterations, max_batchsize, save_images, track_distance, **kwargs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset
    xtest, ytest = utils.load_dataset(dataset=dataset, dataset_split=samplesize_accuracy, root=dataset_root)

    # Load model
    net, art_net, fb_net, alias = utils.get_model(dataset=dataset, modelname=model, norm=model_norm)

    # calculate accuracy, select a subset from the correctly classified images
    correct_map = utils.test_accuracy(net, xtest, ytest, batch_size=100)
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

    # Attack comparison
    results_dict_attack_comparison = Experiment.attack_comparison(attack_types, track_distance=track_distance, **kwargs)

    json_file_path = f'./results/attack_comparison_{alias}_{samplesize_attack}samples_l1-epsilon-{epsilon_l1}_{max_iterations}_iters.json'
    with open(json_file_path, 'w') as f:
        json.dump(results_dict_attack_comparison, f, indent=4)
    print(f'Evaluation results are saved under "{json_file_path}".')

if __name__ == "__main__":
    #os.environ["CUDA_LAUNCH_BLOCKING"] = "1" #prevents "CUDA error: unspecified launch failure" and is recommended for some illegal memory access errors #increases train time by ~15%

    parser = argparse.ArgumentParser(description="Hyperparameter Sweep Script")
    parser.add_argument('--dataset', type=str, default='imagenet', choices=['cifar10', 'imagenet'],
                        help="Dataset to use")
    parser.add_argument('--samplesize_accuracy', type=int, default=10000, help="Split size for test accuracy evaluation")
    parser.add_argument('--samplesize_attack', type=int, default=1000, help="Split size for attack evaluation")
    parser.add_argument('--validation_run', type=utils.str2bool, nargs='?', const=False, default=False, 
                        help="True for validation/tuning, False for testing. Selects attackset from the front or the back")
    parser.add_argument('--dataset_root', type=str, default='../data', help="data folder relative root")
    parser.add_argument('--model', type=str, default='vgg19',
                        help="Model name (e.g., standard,ViT_revisiting,ConvNext_iso_CvSt_revisiting,Salman2020Do_R50,corruption_robust,MainiAVG...)")
    parser.add_argument('--model_norm', type=str, default='Linf',
                        help="Attack Norm the selected model was trained with. Only necessary if you load robustbench models")
    parser.add_argument('--attack_types', type=str, nargs='+',
                        default=['pgd',
                                'SLIDE',
                                'custom_apgd',
                                'ead_fb',
                                'ead_fb_L1_rule_higher_beta',
                                #'exp_attack_l1',
                                #'exp_attack',
                                 ], 
                        choices=['fast_gradient_method',
                                'pgd',
                                'pgd_early_stopping',
                                'pgd_l0',
                                'pgd_l0_linf',
                                'GSE_attack',
                                'sigma_zero',
                                'apgd_art',
                                'AutoAttack',
                                'custom_apgd',
                                'custom_apgdg',
                                'deep_fool',
                                'brendel_bethge',
                                'pgd_blackbox',
                                'pointwise_blackbox',
                                'boundary_blackbox',
                                'hopskipjump_blackbox',
                                'sparse_rs_blackbox',
                                'sparse_rs_custom_L1_blackbox',
                                'square_l1_blackbox',
                                'geoda_blackbox',
                                'carlini_wagner_l2',
                                'carlini_wagner_l0',
                                'ead',
                                'ead_L1_rule_higher_beta',
                                'exp_attack', 
                                'exp_attack_L1_rule_higher_beta',
                                'exp_attack_blackbox', 
                                'exp_attack_blackbox_L1_rule_higher_beta',
                                'exp_attack_l1_blackbox',
                                'exp_attack_l1',
                                'exp_attack_l1_fb',
                                'exp_attack_l1_linf',
                                'exp_attack_l1_ada',
                                'exp_attack_l0',
                                'L1pgd_fb',
                                'SLIDE',
                                'ead_fb',
                                'ead_fb_L1_rule_higher_beta'], 
                        help="List of attack types for comparison (space-separated). ")
    parser.add_argument('--epsilon_l0_feature', type=int, default=25, help="L0 epsilon, translates to overall number of input features altered")
    parser.add_argument('--epsilon_l0_pixel', type=int, default=10, help="L0 epsilon, translates to overall number of pixels (grouped features along channels) altered")
    parser.add_argument('--epsilon_l1', type=float, default=12, help="L1 norm epsilon (default: 12 for CIFAR10)")
    parser.add_argument('--epsilon_l2', type=float, default=0.5, help="L2 norm epsilon")
    parser.add_argument('--eps_iter', type=float, default=0.1, help="Step size for manual iterative attacks")
    parser.add_argument('--learning_rate', type=float, help="Learning rate for exp attacks")
    parser.add_argument('--beta', type=float, help="Beta for exp_attack_l1, exp_attack, EAD")
    parser.add_argument('--attack_norm', type=int, default=1, choices=[1, 2, float('inf')],
                        help="Attack norm type (1, 2, float('inf'))")
    parser.add_argument('--max_iterations', type=int, default=100, help="Maximum iterations for attacks")
    parser.add_argument('--max_batchsize', type=int, default=100, help="Maximum Batchsize to run every adversarial attack on." \
                        "If attack is not optimized or not working with batches, will be returned by attacks.AdversarialAttacks class.")
    parser.add_argument('--save_images', type=int, default=1, help="Integer > 0: number of saved images per attack, 0: do not save)")
    parser.add_argument('--verbose', type=utils.str2bool, nargs='?', const=False, default=False, help="Verbose output")
    parser.add_argument('--track_distance', type=utils.str2bool, nargs='?', const=False, default=False, help="Whether to track all images L1-distance")
    parser.add_argument('--track_c', type=utils.str2bool, nargs='?', const=False, default=False, help="Whether to track all images L1-distance")

    args = parser.parse_args()

    # Convert Namespace to dictionary and filter some arguments to kwargs
    filtered_kwargs = {"learning_rate", "beta", "verbose"}
    kwargs = {k: v for k, v in vars(args).items() if k in filtered_kwargs and v is not None}

    if args.track_c:
        dir = f"c_values_{args.model}_{args.dataset}_{args.max_iterations}_iters"
    else:
        dir = None
    def add_argument_to_kwargs(kwargs, key, value):
        kwargs[key] = value
        return kwargs

    kwargs = add_argument_to_kwargs(kwargs, "track_c", dir)

    main(
        args.dataset, args.samplesize_accuracy, args.samplesize_attack, args.validation_run, args.dataset_root, 
        args.model, args.model_norm, args.attack_types, args.epsilon_l0_feature, args.epsilon_l0_pixel, args.epsilon_l1, 
        args.epsilon_l2, args.eps_iter, args.attack_norm, args.max_iterations, args.max_batchsize, args.save_images, 
        args.track_distance, **kwargs
    )