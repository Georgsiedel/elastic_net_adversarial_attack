
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

class Experiment_class():
    def __init__(self, art_net, fb_net, net, xtest, ytest, alias, epsilon_l1, epsilon_l2, eps_iter, norm, max_iterations, batchsize, save_images, verbose):
        self.art_net = art_net
        self.fb_net=fb_net
        self.net = net
        self.xtest=xtest
        self.ytest=ytest
        self.alias = alias
        self.epsilon_l1=epsilon_l1
        self.epsilon_l2=epsilon_l2
        self.eps_iter=eps_iter
        self.norm=norm
        self.max_iterations=max_iterations
        self.batchsize=batchsize
        self.save_images = save_images
        self.verbose=verbose

    def hyperparameter_sweep(self, hyperparameter, range, attack_type):
        
        '''
        hyperparameter sweep. Pick only one model.
        hyperparameter = 'learning_rate', 'beta' , 'quantile'
        hyperparameter_range: iterable
        '''
            
        results_dict = {}
        for value in range:

            kwargs = {hyperparameter: value}

            results_dict[hyperparameter+str(value)] = {}
            print(f'\t\t-------------- Hyperparameter Sweep for Attack: {attack_type}: {hyperparameter} = {value} ----------------\n')
            _, _, _, _, results_dict[hyperparameter+str(value)]["attack_success_rate_in_epsilon_l1"], results_dict[hyperparameter+str(value)]["attack_success_rate_in_epsilon_l2"], results_dict[hyperparameter+str(value)]["mean_adv_distance_l1"], results_dict[hyperparameter+str(value)]["mean_adv_distance_l2"], adv_images, results_dict[hyperparameter+str(value)]["average_sparsity"] = calculation(
                                                                art_net=self.art_net,
                                                                fb_net=self.fb_net,
                                                                net = self.net,
                                                                xtest=self.xtest,
                                                                ytest=self.ytest,
                                                                epsilon_l1=self.epsilon_l1,
                                                                epsilon_l2=self.epsilon_l2,
                                                                eps_iter=self.eps_iter,
                                                                norm=self.norm,
                                                                max_iterations=self.max_iterations,
                                                                attack_type=attack_type,
                                                                batchsize=self.batchsize,
                                                                save_images=self.save_images,
                                                                verbose=self.verbose,
                                                                **kwargs)
            
            print(hyperparameter+str(value), 'attack success rate in epsilon (L1 / L2): ',
                round(results_dict[hyperparameter+str(value)]["attack_success_rate_in_epsilon_l1"], 4),
                ' / ',
                round(results_dict[hyperparameter+str(value)]["attack_success_rate_in_epsilon_l2"], 4))           
            print('mean adv. distance (L1 / L2): ', 
                   round(results_dict[hyperparameter+str(value)]["mean_adv_distance_l1"], 4), 
                   ' / ', 
                   round(results_dict[hyperparameter+str(value)]["mean_adv_distance_l2"], 4))
        
            if adv_images:
                image_dir = f'./data/hyperparameter_sweep_{attack_type}_{self.alias}_images'
                os.makedirs(image_dir, exist_ok=True)
                for i, img in enumerate(adv_images):
                    if img.dim() == 3:  
                        img = img.permute(1, 2, 0)
                    img = (img * 255).clamp(0, 255).byte().numpy()
                    img = Image.fromarray(img)

                    if i % 3 == 0:
                        img.save(os.path.join(image_dir, f'{hyperparameter}={value}_{i}_original.png'))
                    if i % 3 == 1:
                        img.save(os.path.join(image_dir, f'{hyperparameter}={value}_{i}_adversarial.png'))                    
                    if i % 3 == 2:
                        img.save(os.path.join(image_dir, f'{hyperparameter}={value}_{i}_delta.png'))

        return results_dict

    def attack_comparison(self, attack_types):
        results_dict = {}

        for attack_type in attack_types:
            results_dict[attack_type] = {}
            print(f'\t\t-------------------------- Processing Attack: {attack_type} --------------------------\n')
            results_dict[attack_type]["adversarial_distance_l1"], results_dict[attack_type]["adversarial_distance_l2"], results_dict[attack_type]["runtime"], results_dict[attack_type]["attack_success_rate"], results_dict[attack_type]["attack_success_rate_in_epsilon_l1"], results_dict[attack_type]["attack_success_rate_in_epsilon_l2"], results_dict[attack_type]["mean_adv_distance_l1"], results_dict[attack_type]["mean_adv_distance_l2"], adv_images, results_dict[attack_type]["average_sparsity"] = calculation(
                                                                art_net=self.art_net,
                                                                fb_net=self.fb_net,
                                                                net = self.net,
                                                                xtest=self.xtest,
                                                                ytest=self.ytest,
                                                                epsilon_l1=self.epsilon_l1,
                                                                epsilon_l2=self.epsilon_l2,
                                                                eps_iter=self.eps_iter,
                                                                norm=self.norm,
                                                                max_iterations=self.max_iterations,
                                                                attack_type=attack_type,
                                                                batchsize=self.batchsize,
                                                                save_images=self.save_images,
                                                                verbose=self.verbose)
            
            print(f'\nTotal runtime: {sum(results_dict[attack_type]["runtime"]): .4f} seconds\n')
            print('attack success rate in epsilon (L1 / L2): ',
                round(results_dict[attack_type]["attack_success_rate_in_epsilon_l1"], 4),
                ' / ',
                round(results_dict[attack_type]["attack_success_rate_in_epsilon_l2"], 4))           
            print('mean adv. distance (L1 / L2): ', 
                   round(results_dict[attack_type]["mean_adv_distance_l1"], 4), 
                   ' / ', 
                   round(results_dict[attack_type]["mean_adv_distance_l2"], 4))
        
            if adv_images:
                image_dir = f'./data/attack_comparison_{self.alias}_images'
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

def calculation(art_net, fb_net, net, xtest, ytest, epsilon_l1, epsilon_l2, eps_iter, norm, max_iterations, attack_type, batchsize = 1, learning_rate = None, beta = None, quantile = None, save_images: int = 0, verbose: bool = False):

    sparsity_list,distance_list_l1, distance_list_l2, runtime_list = [], [], [], []
    assert save_images <= len(xtest), "Number of images to be saved is larger than the number processed"
    saved_images = []

    xtest = xtest.to(device)
    ytest = ytest.to(device)
    
    attacks = AdversarialAttacks(art_net=art_net,
                                 net = net,
                          epsilon=epsilon_l1,
                          eps_iter=eps_iter,
                          norm=norm,
                          max_iterations=max_iterations)
    attacker = attacks.init_attacker(attack_type,
                          lr=learning_rate,
                          beta=beta,
                          quantile=quantile,
                          verbose=verbose)
    #robust_predictions_l1 = 0
    #robust_predictions_l2 = 0
    #robust_predictions_en = 0
    attack_successes_in_epsilon_l1 = 0
    attack_successes_in_epsilon_l2 = 0
    attack_successes_in_en = 0
    attack_successes = 0
    #clean_correct = 0
    counter = 0
    
    for i in range(0, len(xtest), batchsize):
        x, y = xtest[i:min(i+batchsize, len(xtest))].clamp(0, 1), ytest[i:min(i+batchsize, len(xtest))]

#    for i, x in enumerate(xtest):

#        x = x.unsqueeze(0).clamp(0, 1)
#        y = ytest[i].unsqueeze(0)
        
        #outputs = art_net.predict(x.cpu())
        #_, clean_predicted = torch.max(torch.tensor(outputs).data, 1)
        #    
        #if int(clean_predicted.item()) != int(y.item()):
        #    if verbose:
        #        print('Misclassified input. Not attacking.')
        #    continue        

        #clean_correct += 1
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
        elif attack_type == 'brendel_bethge':
            _, x_adversarial, _ = attacker(fb_net, x, criterion=y, epsilons=[epsilon_l1])
            x_adversarial = x_adversarial[0].cpu()
        elif attack_type == 'pointwise_blackbox' or attack_type == 'pointwise_blackbox+boundary' or attack_type == 'pointwise_blackbox+hopskipjump':
            _, x_adversarial, _ = attacker(fb_net, x, criterion=y, epsilons=[epsilon_l1])
            x_adversarial = x_adversarial[0].cpu()    
        elif attack_type == 'sparse_rs_blackbox':
            _, x_adversarial = attacker.perturb(x, y)
            x_adversarial = x_adversarial.cpu()    
        elif attack_type == 'original_AutoAttack':
            x_adversarial = attacker.run_standard_evaluation(x, y)
            x_adversarial = x_adversarial.cpu()
        elif attack_type == 'original_AutoAttack_apgd_only':
            x_adversarial = attacker.run_standard_evaluation(x, y,bs=1)
            x_adversarial = x_adversarial.cpu()
        elif attack_type == 'custom_apgd':
            x_adversarial = attacker.run_standard_evaluation(x, y,bs=1)
            x_adversarial = x_adversarial.cpu()
        else:             
            x_adversarial = attacker.generate(x.cpu().numpy(), y.cpu().numpy())
            x_adversarial = torch.from_numpy(x_adversarial)
        
        end_time = time.time()
        runtime = end_time - start_time
        runtime_list.append(runtime)

        # Adversarial accuracy calculation
        output_adversarial = art_net.predict(x_adversarial)
        _, predicted_adversarial = torch.max(torch.tensor(output_adversarial).data, 1)

                # Adversarial distance calculation: if no AE found, save 0.0 as distance
        delta = x.cpu() - x_adversarial.cpu()
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
                    inverted_delta = 1.0 - delta[j]
                    saved_images.append(inverted_delta)

                distance_list_l1.append(distance_l1[j].item())
                distance_list_l2.append(distance_l2[j].item())

                attack_successes_in_epsilon_l1 += (round(distance_l1[j].item(), 3) <= epsilon_l1)
                attack_successes_in_epsilon_l2 += (round(distance_l2[j].item(), 3) <= epsilon_l2)
                attack_successes_in_en += ((round(distance_l2[j].item(), 3) <= epsilon_l2) or (round(distance_l1[j].item(), 3) <= epsilon_l1))
                attack_successes += 1

                dim = torch.numel(delta[j])
                sparsity = (dim - torch.count_nonzero(delta[j]).item()) / dim
                sparsity_list.append(sparsity)

                if verbose:
                    print(f'Image {i + j}\t\tSuccesful attack with adversarial_distance (L1 / L2): {distance_l1[j]:.4f} / {distance_l2[j]:.5f}')

        # Print progress summary after every 50 images
        if (i + x.size(0) - counter) >= 20:
            counter = i + x.size(0)
            print(
                f'{i+x.size(0)} images done. Current Attack Success Rate (Overall / L1 / L2 / EN): '
                f'{attack_successes * 100 / (i+x.size(0)):.2f}% / {attack_successes_in_epsilon_l1 * 100 / (i+x.size(0)):.2f}% / '
                f'{attack_successes_in_epsilon_l2 * 100 / (i+x.size(0)):.2f}% / {attack_successes_in_en * 100 / (i+x.size(0)):.2f}%'
            )

    #adversarial_accuracy_l1 = (robust_predictions_l1 / len(xtest)) * 100
    #adversarial_accuracy_l2 = (robust_predictions_l2 / len(xtest)) * 100
    #adversarial_accuracy_en = (robust_predictions_en / len(xtest)) * 100
    attack_success_rate = (attack_successes / len(xtest)) * 100
    attack_success_rate_in_epsilon_l1 = (attack_successes_in_epsilon_l1 / len(xtest)) * 100
    attack_success_rate_in_epsilon_l2 = (attack_successes_in_epsilon_l2 / len(xtest)) * 100
    attack_success_rate_in_epsilon_en = (attack_successes_in_en / len(xtest)) * 100
    mean_adv_distance_l1 = (sum(distance_list_l1) / attack_successes) if attack_successes else 12
    mean_adv_distance_l2 = (sum(distance_list_l2) / attack_successes) if attack_successes else 5
    mean_sparsity=sum(sparsity_list)/attack_successes if attack_successes else 1.0

    print(f'\naverage sparsity: {mean_sparsity*100:.2f}%\n')

    return distance_list_l1, distance_list_l2, runtime_list, attack_success_rate, attack_success_rate_in_epsilon_l1, attack_success_rate_in_epsilon_l2, mean_adv_distance_l1, mean_adv_distance_l2, saved_images, mean_sparsity