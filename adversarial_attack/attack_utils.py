
from adversarial_attack.attacks import AdversarialAttacks
import torch
import time
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Experiment_class():
    def __init__(self, art_net, fb_net, net, xtest, ytest, alias, epsilon, eps_iter, norm, max_iterations_fast_attacks, max_iterations_slow_attacks, verbose):
        self.art_net = art_net
        self.fb_net=fb_net
        self.net = net
        self.xtest=xtest
        self.ytest=ytest
        self.alias = alias
        self.epsilon=epsilon
        self.eps_iter=eps_iter
        self.norm=norm
        self.max_iterations_fast_attacks=max_iterations_fast_attacks
        self.max_iterations_slow_attacks=max_iterations_slow_attacks
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
        print(range)
        for value in range:

            kwargs = {hyperparameter: value}

            results_dict[hyperparameter+str(value)] = {}
            print(f'\t\t-------------------------- Processing Attack: {attack_type} --------------------------\n')
            _, _, results_dict[hyperparameter+str(value)]["adversarial_accuracy"], _, results_dict[hyperparameter+str(value)]["attack_success_rate_in_epsilon"], results_dict[hyperparameter+str(value)]["mean_adv_distance"] = calculation(
                                                                art_net=self.art_net,
                                                                fb_net=self.fb_net,
                                                                net = self.net,
                                                                xtest=self.xtest,
                                                                ytest=self.ytest,
                                                                epsilon=self.epsilon,
                                                                eps_iter=self.eps_iter,
                                                                norm=self.norm,
                                                                max_iterations_fast_attacks=self.max_iterations_fast_attacks,
                                                                max_iterations_slow_attacks=self.max_iterations_slow_attacks,
                                                                attack_type=attack_type,
                                                                verbose=self.verbose,
                                                                **kwargs)
            
            print(hyperparameter+str(value), ' attack success rate in epsilon: ', results_dict[hyperparameter+str(value)]["attack_success_rate_in_epsilon"])
            print(hyperparameter+str(value), ' mean adv. distance: ', results_dict[hyperparameter+str(value)]["mean_adv_distance"])
        
        json_file_path = f'./data/hyperparameter_sweep_{attack_type}_{self.alias}.json'
        with open(json_file_path, 'w') as f:
            json.dump(results_dict, f, indent=4)
        print(f'Evaluation results are saved under "{json_file_path}".')

        return results_dict

    def attack_comparison(self, attack_types):
        results_dict = {}
        print(attack_types)

        for attack_type in attack_types:
            results_dict[attack_type] = {}
            print(f'\t\t-------------------------- Processing Attack: {attack_type} --------------------------\n')
            results_dict[attack_type]["adversarial_distance"], results_dict[attack_type]["runtime"], results_dict[attack_type]["adversarial_accuracy"], results_dict[attack_type]["attack_success_rate"], results_dict[attack_type]["attack_success_rate_in_epsilon"], results_dict[attack_type]["mean_adv_distance"] = calculation(
                                                                art_net=self.art_net,
                                                                fb_net=self.fb_net,
                                                                net = self.net,
                                                                xtest=self.xtest,
                                                                ytest=self.ytest,
                                                                epsilon=self.epsilon,
                                                                eps_iter=self.eps_iter,
                                                                norm=self.norm,
                                                                max_iterations_fast_attacks=self.max_iterations_fast_attacks,
                                                                max_iterations_slow_attacks=self.max_iterations_slow_attacks,
                                                                attack_type=attack_type,
                                                                verbose=self.verbose)
            
            print(f'\nTotal runtime: {sum(results_dict[attack_type]["runtime"]): .5f} seconds\n')
            print('attack success rate in epsilon: ', results_dict[attack_type]["attack_success_rate_in_epsilon"])
            print('mean adv. distance: ', results_dict[attack_type]["mean_adv_distance"])
        
        json_file_path = f'./data/attack_comparison_{self.alias}.json'
        with open(json_file_path, 'w') as f:
            json.dump(results_dict, f, indent=4)
        print(f'Evaluation results are saved under "{json_file_path}".')
        
        return results_dict


def attack_with_early_stopping(art_net, x, y, PGD_iterations, attacker):
    label_flipped = False

    for j in range(PGD_iterations):
        adv_inputs = attacker.generate(x, y.numpy(), verbose=False)

        outputs = art_net.predict(adv_inputs)
        _, predicted = torch.max(torch.tensor(outputs).data, 1)
        label_flipped = bool(predicted.item() != int(y.item()))

        if label_flipped:
            print(f'\tIterations for successful iterative attack: {j+1}')
            break
        
        x = adv_inputs.copy()
            
    return adv_inputs

def calculation(art_net, fb_net, net, xtest, ytest, epsilon, eps_iter, norm, max_iterations_slow_attacks, max_iterations_fast_attacks, attack_type, learning_rate = None, beta = None, verbose: bool = False):

    distance_list, runtime_list = [], []
    
    xtest = xtest.to(device)
    ytest = ytest.to(device)
    
    attacks = AdversarialAttacks(art_net=art_net,
                                 net = net,
                          epsilon=epsilon,
                          eps_iter=eps_iter,
                          norm=norm,
                          max_iterations_fast_attacks=max_iterations_fast_attacks,
                          max_iterations_slow_attacks=max_iterations_slow_attacks)
    attacker = attacks.init_attacker(attack_type,
                          lr=learning_rate,
                          beta=beta,
                          verbose=verbose)

    robust_predictions = 0
    attack_successes_in_epsilon = 0
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
            distance_list.append(False)
            runtime_list.append(False)
            continue        

        clean_correct += 1

        start_time = time.time()

        if attack_type == 'pgd_early_stopping':
            x_adversarial = attack_with_early_stopping(art_net=art_net,
                                                                x=x.numpy(),
                                                                y=y,
                                                                PGD_iterations=max_iterations_fast_attacks,
                                                                attacker=attacker)
            x_adversarial = torch.from_numpy(x_adversarial)
        elif attack_type == 'brendel_bethge':
            _, x_adversarial, _ = attacker(fb_net, x, y, epsilons=[epsilon])
            x_adversarial = x_adversarial[0]
        elif attack_type == 'original_AutoAttack':
            x_adversarial = attacker.run_standard_evaluation(x, y)
            x_adversarial = x_adversarial
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
        if int(predicted_adversarial.item()) == int(y.item()):
            robust_predictions += 1
            distance = 0.0
            distance_list.append(distance)
            if verbose:
                print(f'Image {i}: No adversarial example found.')
        else:
            distance = torch.norm((x.cpu() - x_adversarial), p=float(norm))
            robust_predictions += (round(distance.item(), 2) > epsilon) 
            attack_successes_in_epsilon += (round(distance.item(), 2) <= epsilon) 
            attack_successes += 1
            distance_list.append(distance.item())

        if verbose:
            print(f'Image {i}\t\tAdversarial_distance: {distance:.5f}\t\tRuntime: {runtime:5f} seconds')
        if (i + 1) % 20 == 0:
            print(f'{i+1} images done. Current Adversarial Accuracy: {robust_predictions*100/(i+1)}%')

    adversarial_accuracy = (robust_predictions / len(xtest)) * 100
    attack_success_rate = (attack_successes / clean_correct) * 100
    attack_success_rate_in_epsilon = (attack_successes_in_epsilon / clean_correct) * 100
    mean_adv_distance = (sum(distance_list) / clean_correct)

    print(f'\nAdversarial accuracy: {adversarial_accuracy}%\n')

    return distance_list, runtime_list, adversarial_accuracy, attack_success_rate, attack_success_rate_in_epsilon, mean_adv_distance