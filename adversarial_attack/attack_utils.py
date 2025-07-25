import validation.validate_image
from adversarial_attack.attacks import AdversarialAttacks
import torch
import time
import json
import art.config
import numpy
import os
from PIL import Image
art.config.ART_NUMPY_DTYPE=numpy.float64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_cpu = torch.device('cpu')

def extract_2nd_and_3rd(kwargs: dict) -> str:
    """
    From kwargs dict return a string of
      key2-value2-key3-value3
    if the 2nd and 3rd items exist.  
    If only a 2nd item exists, returns "key2-value2".  
    Otherwise returns empty string.
    """
    items = list(kwargs.items())
    out_parts = []
    # if there is a 3nd item
    if len(items) > 2:
        k2, v2 = items[2]
        out_parts.extend([str(k2), str(v2)])
    # if there is a 3rd item
    if len(items) > 3:
        k3, v3 = items[3]
        out_parts.extend([str(k3), str(v3)])
    return "-".join(out_parts)

class Experiment_class():
    def __init__(self, art_net, fb_net, net, xtest, ytest, alias, epsilon_l0_pixel, epsilon_l0_feature, epsilon_l1, epsilon_l2, eps_iter, norm, max_iterations, max_batchsize, save_images):
        self.art_net = art_net
        self.fb_net=fb_net
        self.net = net
        self.xtest=xtest
        self.ytest=ytest
        self.alias = alias
        self.epsilon_l0_pixel = epsilon_l0_pixel
        self.epsilon_l0_feature = epsilon_l0_feature
        self.epsilon_l1=epsilon_l1
        self.epsilon_l2=epsilon_l2
        self.eps_iter=eps_iter
        self.norm=norm
        self.max_iterations=max_iterations
        self.max_batchsize=max_batchsize
        self.save_images = save_images

    def hyperparameter_sweep(self, attack_type, **kwargs):
        
        results_dict = {}

        _, _, results_dict["mean_runtime_per_image"], results_dict["attack_success_rate"], results_dict["attack_success_rate_in_epsilon_l0_pixels"], results_dict["attack_success_rate_in_epsilon_l0_features"], results_dict["attack_success_rate_in_epsilon_l1"], results_dict["attack_success_rate_in_epsilon_l1_linf"], results_dict["attack_success_rate_in_epsilon_l2"], results_dict["mean_adv_distance_l1"], results_dict["mean_adv_distance_l2"], adv_images, results_dict["average_sparsity_feature_ratio"], results_dict["average_sparsity_pixels"] = calculation(
                                                            art_net=self.art_net,
                                                            fb_net=self.fb_net,
                                                            net = self.net,
                                                            xtest=self.xtest,
                                                            ytest=self.ytest,
                                                            epsilon_l1=self.epsilon_l1,
                                                            epsilon_l2=self.epsilon_l2,
                                                            epsilon_l0_pixel = self.epsilon_l0_pixel,
                                                            epsilon_l0_feature = self.epsilon_l0_feature,
                                                            eps_iter=self.eps_iter,
                                                            norm=self.norm,
                                                            max_iterations=self.max_iterations,
                                                            attack_type=attack_type,
                                                            max_batchsize=self.max_batchsize,
                                                            save_images=self.save_images,
                                                            **kwargs)
        
        print(f'\nTotal runtime: {len(self.ytest) * results_dict["mean_runtime_per_image"]: .4f} seconds\n')
        print(f'\nAverage sparsity (feature-ratio / pixels): {results_dict["average_sparsity_feature_ratio"]: .5f} / {results_dict["average_sparsity_pixels"]: .3f}\n')
        print('attack success rate in epsilon (Overall / L0 Features / L0 Pixels / L1 / Pixel L1 / L2): ',
                round(results_dict[attack_type]["attack_success_rate"], 4), 
                  ' / ',
                round(results_dict[attack_type]["attack_success_rate_in_epsilon_l0_features"], 4),
                  ' / ',
                round(results_dict[attack_type]["attack_success_rate_in_epsilon_l0_pixels"], 4),
                  ' / ',
                round(results_dict["attack_success_rate_in_epsilon_l1"], 4),
                ' / ',
                round(results_dict["attack_success_rate_in_epsilon_l1_linf"], 4),
                ' / ',
                round(results_dict["attack_success_rate_in_epsilon_l2"], 4))           
        print('mean adv. distance (L1 / L2): ', 
                round(results_dict["mean_adv_distance_l1"], 5), 
                ' / ', 
                round(results_dict["mean_adv_distance_l2"], 5))
    
        if adv_images:
            image_dir = f'./results/images/hyperparameter_sweep_{attack_type}_{self.alias}_eps{self.epsilon_l1}_images'
            os.makedirs(image_dir, exist_ok=True)
            for i, img in enumerate(adv_images):
                if img.dim() == 3:  
                    img = img.permute(1, 2, 0)

                #validation.validate_image.validate_tensor(img)

                img = (img * 255).clamp(0, 255).byte().numpy()
                img = Image.fromarray(img)

                if i % 3 == 0:
                    img.save(os.path.join(image_dir, f'{extract_2nd_and_3rd(kwargs)}_{i}_original.png'))
                if i % 3 == 1:
                    img.save(os.path.join(image_dir, f'{extract_2nd_and_3rd(kwargs)}_{i}_adversarial.png'))                    
                if i % 3 == 2:
                    img.save(os.path.join(image_dir, f'{extract_2nd_and_3rd(kwargs)}_{i}_delta.png'))

        return results_dict

    def attack_comparison(self, attack_types, track_distance, **kwargs):
        results_dict = {}

        for attack_type in attack_types:
            results_dict[attack_type] = {}
            print(f'\t\t-------------------------- Processing Attack: {attack_type} --------------------------\n')
            distance_list_l1,_, results_dict[attack_type]["mean_runtime_per_image"], results_dict[attack_type]["attack_success_rate"], results_dict[attack_type]["attack_success_rate_in_epsilon_l0_pixels"], results_dict[attack_type]["attack_success_rate_in_epsilon_l0_features"], results_dict[attack_type]["attack_success_rate_in_epsilon_l1"], results_dict[attack_type]["attack_success_rate_in_epsilon_l1_linf"], results_dict[attack_type]["attack_success_rate_in_epsilon_l2"], results_dict[attack_type]["mean_adv_distance_l1"], results_dict[attack_type]["mean_adv_distance_l2"], adv_images, results_dict[attack_type]["average_sparsity_feature_ratio"], results_dict[attack_type]["average_sparsity_pixels"] = calculation(
                                                                art_net=self.art_net,
                                                                fb_net=self.fb_net,
                                                                net = self.net,
                                                                xtest=self.xtest,
                                                                ytest=self.ytest,
                                                                epsilon_l0_pixel = self.epsilon_l0_pixel,
                                                                epsilon_l0_feature = self.epsilon_l0_feature,
                                                                epsilon_l1=self.epsilon_l1,
                                                                epsilon_l2=self.epsilon_l2,
                                                                eps_iter=self.eps_iter,
                                                                norm=self.norm,
                                                                max_iterations=self.max_iterations,
                                                                attack_type=attack_type,
                                                                max_batchsize=self.max_batchsize,
                                                                save_images=self.save_images,
                                                                **kwargs)
            if track_distance:
                results_dict[attack_type]["distance_list_l1"] = distance_list_l1
                for eps in [2, 4, 12, 25, 50, 75, 255]:
                    string = f'ASR_in_L1={eps}'
                    results_dict[attack_type][string] = sum(1 for v in distance_list_l1 if v < eps) / len(self.xtest)

            print(f'\nTotal runtime: {len(self.ytest) * results_dict[attack_type]["mean_runtime_per_image"]: .4f} seconds\n')
            print(f'\nAverage sparsity (feature-ratio / pixels): {results_dict[attack_type]["average_sparsity_feature_ratio"]: .5f} / {results_dict[attack_type]["average_sparsity_pixels"]: .3f}\n')
            print('attack success rate in epsilon (Overall / L0 Features / L0 Pixels / L1 / Pixel L1 / L2): ',
                  round(results_dict[attack_type]["attack_success_rate"], 4), 
                  ' / ',
                  round(results_dict[attack_type]["attack_success_rate_in_epsilon_l0_features"], 4),
                  ' / ',
                  round(results_dict[attack_type]["attack_success_rate_in_epsilon_l0_pixels"], 4),
                  ' / ',
                round(results_dict[attack_type]["attack_success_rate_in_epsilon_l1"], 4),
                ' / ',
                round(results_dict[attack_type]["attack_success_rate_in_epsilon_l1_linf"], 4),
                ' / ',
                round(results_dict[attack_type]["attack_success_rate_in_epsilon_l2"], 4))           
            print('mean adv. distance (L1 / L2): ', 
                   round(results_dict[attack_type]["mean_adv_distance_l1"], 5), 
                   ' / ', 
                   round(results_dict[attack_type]["mean_adv_distance_l2"], 5))
        
            if adv_images:
                image_dir = f'./results/images/attack_comparison_{self.alias}_eps{self.epsilon_l1}_{self.max_iterations}_iters_images'
                os.makedirs(image_dir, exist_ok=True)
                for i, img in enumerate(adv_images):
                    if img.dim() == 3:  
                        img = img.permute(1, 2, 0)
                    img = (img * 255).clamp(0, 255).byte().numpy()
                    img = Image.fromarray(img)

                    if i % 3 == 0:
                        img.save(os.path.join(image_dir, f'{attack_type}_{i}_original.png'))
                    if i % 3 == 1:
                        img.save(os.path.join(image_dir, f'{attack_type}_{i}_adversarial.png'))
                    if i % 3 == 2:
                        img.save(os.path.join(image_dir, f'{attack_type}_{i}_delta.png'))
        
        return results_dict


def attack_with_early_stopping(art_net, x, y, PGD_iterations, attacker, verbose=False):
    label_flipped = False

    for j in range(PGD_iterations):
        adv_inputs = attacker.generate(x, y, verbose=False)

        outputs = art_net.predict(adv_inputs)
        _, predicted = torch.max(torch.tensor(outputs).data, 1)
        label_flipped = bool(predicted.item() != int(y.item()))

        if label_flipped:
            if verbose:
                print(f'\tIterations for successful iterative attack: {j+1}')
            break
        
        x = adv_inputs.copy()
            
    return adv_inputs

def calculation(art_net, fb_net, net, xtest, ytest, epsilon_l0_pixel, epsilon_l0_feature, epsilon_l1, epsilon_l2, 
                eps_iter, norm, max_iterations, attack_type, max_batchsize = 1, learning_rate = None, beta = None, 
                quantile = None, max_iterations_sweep = None, save_images: int = 0, **kwargs):

    sparsity_list_feature_ratio, sparsity_list_pixel_count, distance_list_l1, distance_list_l1_linf, distance_list_l2, runtime_list = [], [], [], [], [], []
    assert save_images <= len(xtest), "Number of images to be saved is larger than the number processed"
    saved_images = []

    xtest = xtest.to(device)
    ytest = ytest.to(device)
    verbose = kwargs.get("verbose", False)
    
    attacks = AdversarialAttacks(art_net=art_net,
                                 net = net,
                          epsilon_l1=epsilon_l1,
                          epsilon_l0_pixel=epsilon_l0_pixel, 
                          epsilon_l0_feature=epsilon_l0_feature, 
                          eps_iter=eps_iter,
                          norm=norm,
                          max_iterations=max_iterations)
    attacker, batchsize = attacks.init_attacker(attack_type,
                                     max_batchsize=max_batchsize,
                          lr=learning_rate,
                          beta=beta,
                          quantile=quantile,
                          max_iterations_sweep=max_iterations_sweep,
                          **kwargs)
    attack_successes_in_epsilon_l0_pixel = 0
    attack_successes_in_epsilon_l0_feature = 0
    attack_successes_in_epsilon_l1_linf = 0
    attack_successes_in_epsilon_l1 = 0
    attack_successes_in_epsilon_l1_linf = 0
    attack_successes_in_epsilon_l2 = 0
    attack_successes_in_en = 0
    attack_successes = 0
    counter = 0
    
    for i in range(0, len(xtest), batchsize):
        x, y = xtest[i:min(i+batchsize, len(xtest))].clamp(0, 1), ytest[i:min(i+batchsize, len(xtest))]

        start_time = time.time()
        if attack_type == 'pgd_early_stopping':
            assert x.shape[0] == 1
            x_adversarial = attack_with_early_stopping(art_net=art_net,
                                                                x=x.cpu().numpy(),
                                                                y=y.cpu().numpy(),
                                                                PGD_iterations=max_iterations,
                                                                attacker=attacker,
                                                                verbose = verbose)
            x_adversarial = torch.from_numpy(x_adversarial)
        elif attack_type in ['exp_attack_sparse', 'L1pgd_fb', 'SLIDE']:
            _, x_adversarial, _ = attacker(fb_net, x, criterion=y, epsilons=[epsilon_l1])
            x_adversarial = x_adversarial[0].cpu()    
        elif attack_type in ['exp_attack_l1_fb']:
            _, x_adversarial, _ = attacker(fb_net, x, criterion=y, epsilons=[epsilon_l0_pixel])
            x_adversarial = x_adversarial[0].cpu()    
        elif attack_type in ['brendel_bethge_l0', 'brendel_bethge_l1', 'pointwise_blackbox', 'boundary_blackbox', 'ead_fb', 'ead_fb_L1_rule_higher_beta']:
            _, x_adversarial, _ = attacker(fb_net, x, criterion=y, epsilons=None)
            x_adversarial = x_adversarial.cpu()
        elif attack_type in ['sparse_rs_blackbox', 'sparse_rs_custom_L1_blackbox']:
            _, x_adversarial = attacker.perturb(x, y)
            x_adversarial = x_adversarial.cpu()    
        elif attack_type in ['custom_apgd', 'custom_apgdg','AutoAttack', 'square_l1_blackbox', 'FAB']:
            x_adversarial = attacker.run_standard_evaluation(x, y)
            x_adversarial = x_adversarial.cpu()
        elif attack_type in ['pgd_l0', 'pgd_l0_linf']:
            x = x.permute(0,2,3,1)
            x_adversarial, adv_acc = attacker.perturb(x.cpu().numpy(), y.cpu().numpy())
            x_adversarial = torch.from_numpy(x_adversarial).float()
            x_adversarial = x_adversarial.permute(0,3,1,2)
            x = x.permute(0,3,1,2)
        elif attack_type in ['sigma_zero', 'GSE_attack']:
            x_adversarial = attacker(x, y)
            x_adversarial = x_adversarial.detach().cpu()
        else:             
            x_adversarial = attacker.generate(x.cpu().numpy(), y.cpu().numpy())
            x_adversarial = torch.from_numpy(x_adversarial).float()
        
        end_time = time.time()
        runtime = end_time - start_time
        runtime_list.append(runtime)

        # Adversarial accuracy calculation
        output_adversarial = art_net.predict(x_adversarial)
        _, predicted_adversarial = torch.max(torch.tensor(output_adversarial).data, 1)
        # Adversarial distance calculation: if no AE found, save 0.0 as distance
        delta = x.cpu() - x_adversarial.cpu()
        distance_l1_linf = torch.sum(torch.max(torch.abs(delta),dim=1).values, dim=(1,2)) # sum of max pixelwise distance (L1 in pixelspace)
        distance_l1 = torch.norm(delta.view(delta.size(0), -1), p=1, dim=1)  # Batch-wise L1 distance
        distance_l2 = torch.norm(delta.view(delta.size(0), -1), p=2, dim=1)  # Batch-wise L2 distance

        # Iterate over the batch
        for j in range(x.size(0)):
            if int(predicted_adversarial[j].item()) == int(y[j].item()):
                if verbose:
                    print(f'Image {i + j}: No adversarial example found.')
            else:
                if 3 * save_images > len(saved_images):  # Save only successful adversarial examples
                    saved_images.append(x.cpu()[j])
                    saved_images.append(x_adversarial.cpu()[j])
                    inverted_delta = (delta[j] + 0.5).clamp(0, 1) #perturbations are magnified 10x and applied on 0.5 (greyscale) so that positive and negative deltas are shown
                    saved_images.append(inverted_delta)

                distance_list_l1.append(distance_l1[j].item())
                distance_list_l1_linf.append(distance_l1_linf[j].item())
                distance_list_l2.append(distance_l2[j].item())

                attack_successes_in_epsilon_l1 += (round(distance_l1[j].item(), 1) <= epsilon_l1)
                attack_successes_in_epsilon_l1_linf += (round(distance_l1_linf[j].item(), 1) <= epsilon_l1)
                attack_successes_in_epsilon_l2 += (round(distance_l2[j].item(), 1) <= epsilon_l2)
                attack_successes_in_en += ((round(distance_l2[j].item(), 1) <= epsilon_l2) or (round(distance_l1[j].item(), 1) <= epsilon_l1))
                attack_successes += 1

                dim = torch.numel(delta[j])
                changed_features = torch.count_nonzero(delta[j]).item()
                changed_pixels = torch.count_nonzero(torch.max(torch.abs(delta[j]),dim=0).values).item()

                sparsity_feature_ratio = (dim - changed_features) / dim

                sparsity_list_feature_ratio.append(sparsity_feature_ratio)
                sparsity_list_pixel_count.append(changed_pixels)

                attack_successes_in_epsilon_l0_feature += (changed_features <= epsilon_l0_feature)
                attack_successes_in_epsilon_l0_pixel += (changed_pixels <= epsilon_l0_pixel)

                if verbose:
                    print(f'Image {i + j}\t\tSuccesful attack with (L1-distance / Feature sparsity): {distance_l1[j]:.4f} / {sparsity_feature_ratio:.5f}')

        # Print progress summary after every some images
        if (i + x.size(0) - counter) >= max(50, batchsize):
            counter = i + x.size(0)
            print(
                f'{i+x.size(0)} images done. Current Attack Success Rate: Overall - {attack_successes * 100 / (i+x.size(0)):.2f}% / '
                f'L0 Features - {attack_successes_in_epsilon_l0_feature * 100 / (i+x.size(0)):.2f}% / '
                f'L0 Pixels - {attack_successes_in_epsilon_l0_pixel * 100 / (i+x.size(0)):.2f}% / '
                f'L1 - {attack_successes_in_epsilon_l1 * 100 / (i+x.size(0)):.2f}% / '
                f'Pixel L1 - {attack_successes_in_epsilon_l1_linf * 100 / (i+x.size(0)):.2f}% / '
                f'L2 - {attack_successes_in_epsilon_l2 * 100 / (i+x.size(0)):.2f}% / '
                f'EN - {attack_successes_in_en * 100 / (i+x.size(0)):.2f}% / ')

    attack_success_rate = (attack_successes / len(xtest)) * 100
    attack_success_rate_in_epsilon_l0_feature = (attack_successes_in_epsilon_l0_feature / len(xtest)) * 100
    attack_success_rate_in_epsilon_l0_pixel = (attack_successes_in_epsilon_l0_pixel / len(xtest)) * 100
    attack_success_rate_in_epsilon_l1 = (attack_successes_in_epsilon_l1 / len(xtest)) * 100
    attack_success_rate_in_epsilon_l1_linf = (attack_successes_in_epsilon_l1_linf / len(xtest)) * 100
    attack_success_rate_in_epsilon_l2 = (attack_successes_in_epsilon_l2 / len(xtest)) * 100
    mean_adv_distance_l1 = (sum(distance_list_l1) / attack_successes) if attack_successes!=0 else 0.0
    mean_adv_distance_l2 = (sum(distance_list_l2) / attack_successes) if attack_successes!=0 else 0.0
    mean_sparsity_feature_ratio=sum(sparsity_list_feature_ratio)/attack_successes if attack_successes else 0.0
    mean_sparsity_pixels=sum(sparsity_list_pixel_count)/attack_successes if attack_successes else 0.0
    mean_runtime=sum(runtime_list) / len(xtest)

    return distance_list_l1, distance_list_l2, mean_runtime, attack_success_rate, attack_success_rate_in_epsilon_l0_pixel, attack_success_rate_in_epsilon_l0_feature,  attack_success_rate_in_epsilon_l1, attack_success_rate_in_epsilon_l1_linf, attack_success_rate_in_epsilon_l2, mean_adv_distance_l1, mean_adv_distance_l2, saved_images, mean_sparsity_feature_ratio, mean_sparsity_pixels