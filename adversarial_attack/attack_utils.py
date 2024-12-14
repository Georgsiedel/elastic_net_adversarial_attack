
from adversarial_attack.attacks import AdversarialAttacks
import torch
import time
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Experiment_class():
    def __init__(self, art_net, fb_net, net, xtest, ytest, alias, epsilon_l1, epsilon_l2, eps_iter, norm, max_iterations, verbose):
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
        self.verbose=verbose

    def hyperparameter_sweep(self, hyperparameter, range, attack_type):
        
        '''
        hyperparameter sweep. Pick only one model.
        hyperparameter = 'learning_rate', 'beta' 
        attack_type= 
            #'fast_gradient_method',
            #'projected_gradient_descent', 
            #'pgd_early_stopping', #not-bounded
            #'auto_projected_gradient_descent', #bounded-full
            #'deep_fool', #not-bounded
            #'brendel_bethge', #bounded-min
            #'carlini_wagner_l2', #not-bounded
            #'elastic_net', #not-bounded
            #'exp_attack',
            #'exp_attack_smooth',
            #'elastic_net_L1_rule', #not-bounded
            #'elastic_net_L1_rule_higher_beta', #not-bounded
            #'ART_AutoAttack', #bounded-full
            #'original_AutoAttack', #bounded-full
        hyperparameter_range: iterable

        '''
            
        results_dict = {}
        for value in range:

            kwargs = {hyperparameter: value}

            results_dict[hyperparameter+str(value)] = {}
            print(f'\t\t-------------- Hyperparameter Sweep for Attack: {attack_type}: {hyperparameter} = {value} ----------------\n')
            _, _, _, results_dict[hyperparameter+str(value)]["adversarial_accuracy_l1"], results_dict[hyperparameter+str(value)]["adversarial_accuracy_l2"], _, results_dict[hyperparameter+str(value)]["attack_success_rate_in_epsilon_l1"], results_dict[hyperparameter+str(value)]["attack_success_rate_in_epsilon_l2"], results_dict[hyperparameter+str(value)]["mean_adv_distance_l1"], results_dict[hyperparameter+str(value)]["mean_adv_distance_l2"] = calculation(
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
            
        json_file_path = f'./data/hyperparameter_sweep_{attack_type}_{self.alias}.json'
        with open(json_file_path, 'w') as f:
            json.dump(results_dict, f, indent=4)
        print(f'Evaluation results are saved under "{json_file_path}".')

        return results_dict

    def attack_comparison(self, attack_types):
        results_dict = {}

        for attack_type in attack_types:
            results_dict[attack_type] = {}
            print(f'\t\t-------------------------- Processing Attack: {attack_type} --------------------------\n')
            results_dict[attack_type]["adversarial_distance_l1"], results_dict[attack_type]["adversarial_distance_l2"], results_dict[attack_type]["runtime"], results_dict[attack_type]["adversarial_accuracy_l1"], results_dict[attack_type]["adversarial_accuracy_l2"], results_dict[attack_type]["attack_success_rate"], results_dict[attack_type]["attack_success_rate_in_epsilon_l1"], results_dict[attack_type]["attack_success_rate_in_epsilon_l2"], results_dict[attack_type]["mean_adv_distance_l1"], results_dict[attack_type]["mean_adv_distance_l2"] = calculation(
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
        
        json_file_path = f'./data/attack_comparison_{self.alias}.json'
        with open(json_file_path, 'w') as f:
            json.dump(results_dict, f, indent=4)
        print(f'Evaluation results are saved under "{json_file_path}".')
        
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

def calculation(art_net, fb_net, net, xtest, ytest, epsilon_l1, epsilon_l2, eps_iter, norm, max_iterations, attack_type, learning_rate = None, beta = None, verbose: bool = False):

    sparsity_list,distance_list_l1, distance_list_l2, runtime_list = [], [], [], []
    
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
                          verbose=verbose)

    robust_predictions_l1 = 0
    attack_successes_in_epsilon_l1 = 0
    robust_predictions_l2 = 0
    robust_predictions_en = 0
    attack_successes_in_epsilon_l2 = 0
    attack_successes_in_en = 0
    attack_successes = 0
    clean_correct = 0

    for i, x in enumerate(xtest):

        x = x.unsqueeze(0)
        y = ytest[i].unsqueeze(0)
        outputs = art_net.predict(x.cpu())
        
        _, clean_predicted = torch.max(torch.tensor(outputs).data, 1)
            
        if int(clean_predicted.item()) != int(y.item()):
            if verbose:
                print('Misclassified input. Not attacking.')
            if (i + 1) % 20 == 0:
                print(f'{i+1} images done. Current Adversarial Accuracy (L1 / L2): {robust_predictions_l1*100/(i+1)} / {robust_predictions_l2*100/(i+1)}%')

            distance_list_l1.append(False)
            distance_list_l2.append(False)
            runtime_list.append(False)
            continue        

        clean_correct += 1

        start_time = time.time()

        if attack_type == 'pgd_early_stopping':
            x_adversarial = attack_with_early_stopping(art_net=art_net,
                                                                x=x.cpu().numpy(),
                                                                y=y.cpu().numpy(),
                                                                PGD_iterations=max_iterations,
                                                                attacker=attacker,
                                                                verbose = verbose)
            x_adversarial = torch.from_numpy(x_adversarial)
        elif attack_type == 'brendel_bethge':
            _, x_adversarial, _ = attacker(fb_net, x, y, epsilons=[epsilon_l1])
            x_adversarial = x_adversarial[0].cpu()
        elif attack_type == 'original_AutoAttack':
            x_adversarial = attacker.run_standard_evaluation(x, y)
            x_adversarial = x_adversarial.cpu()
        elif attack_type == 'original_AutoAttack_apgd-only':
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
        delta=x.cpu() - x_adversarial
        distance_l1 = torch.norm(delta, p=float(1))
        distance_l2 = torch.norm(delta, p=float(2))
        distance_list_l1.append(distance_l1.item())
        distance_list_l2.append(distance_l2.item())

        if int(predicted_adversarial.item()) == int(y.item()):
            robust_predictions_l1 += 1
            robust_predictions_l2 += 1
            robust_predictions_en+=1
            if verbose:
                print(f'Image {i}: No adversarial example found.')
        else:
            robust_predictions_l1 += (round(distance_l1.item(), 2) > epsilon_l1) 
            robust_predictions_l2 += (round(distance_l2.item(), 2) > epsilon_l2) 
            robust_predictions_en += (round(distance_l2.item(), 2) > epsilon_l2) or  (round(distance_l1.item(), 2) > epsilon_l1) 
            attack_successes_in_epsilon_l1 += (round(distance_l1.item(), 3) <= epsilon_l1)
            attack_successes_in_epsilon_l2 += (round(distance_l2.item(), 3) <= epsilon_l2) 
            attack_successes_in_en += ((round(distance_l2.item(), 3) <= epsilon_l2) and (round(distance_l1.item(), 3) <= epsilon_l1)) 
            attack_successes += 1
            dim=torch.numel(delta)
            sparsity = (dim-torch.count_nonzero(delta).item())/dim
            sparsity_list.append(sparsity)

        if verbose:
            print(f'Image {i}\t\tAdversarial_distance (L1 / L2): {distance_l1:.4f} / {distance_l2:.5f}\t\tRuntime: {runtime:5f} seconds')
        if (i + 1) % 20 == 0:
            print(f'{i+1} images done. Current Adversarial Accuracy (L1 / L2/ EN): {robust_predictions_l1*100/(i+1)} / {robust_predictions_l2*100/(i+1)}/{robust_predictions_en*100/(i+1)}%')

    adversarial_accuracy_l1 = (robust_predictions_l1 / len(xtest)) * 100
    adversarial_accuracy_l2 = (robust_predictions_l2 / len(xtest)) * 100
    adversarial_accuracy_en = (robust_predictions_en / len(xtest)) * 100
    attack_success_rate = (attack_successes / clean_correct) * 100
    attack_success_rate_in_epsilon_l1 = (attack_successes_in_epsilon_l1 / clean_correct) * 100
    attack_success_rate_in_epsilon_l2 = (attack_successes_in_epsilon_l2 / clean_correct) * 100
    attack_success_rate_in_epsilon_en = (attack_successes_in_en / clean_correct) * 100
    mean_adv_distance_l1 = (sum(distance_list_l1) / clean_correct)
    mean_adv_distance_l2 = (sum(distance_list_l2) / clean_correct)
    mean_sparsity=sum(sparsity_list)/attack_successes if attack_successes else 1.0

    print(f'\nAdversarial accuracy (L1 / L2/ EN): {adversarial_accuracy_l1:.4f} / {adversarial_accuracy_l2:.4f}/ {adversarial_accuracy_en:.4f}%\n')
    print(f'\naverage sparsity: {mean_sparsity:.4f}%\n')

    return distance_list_l1, distance_list_l2, runtime_list, adversarial_accuracy_l1, adversarial_accuracy_l2, attack_success_rate, attack_success_rate_in_epsilon_l1, attack_success_rate_in_epsilon_l2, mean_adv_distance_l1, mean_adv_distance_l2