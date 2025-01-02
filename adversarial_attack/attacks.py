import torch
import foolbox as fb
from art.attacks.evasion import (FastGradientMethod,
                                 ProjectedGradientDescentNumpy,
                                 AutoProjectedGradientDescent,
                                 AutoAttack,
                                 CarliniL2Method,
                                 DeepFool,
                                 ElasticNet)
from adversarial_attack.exp_attack import ExpAttack
from adversarial_attack.exp_attack_l1 import ExpAttackL1
#from adversarial_attack.acc_exp_attack import AccExpAttack
from autoattack import AutoAttack as original_AutoAttack
from adversarial_attack.auto_attack.autoattack_custom import AutoAttack_Custom
from adversarial_attack.exp_attack_l1_ada import ExpAttackL1Ada
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AdversarialAttacks:
  def __init__(self, art_net, net, epsilon, eps_iter, norm, max_iterations):
    self.art_net = art_net
    self.epsilon = epsilon
    self.eps_iter = eps_iter
    self.norm = norm
    self.max_iterations = max_iterations
    self.net = net
  def init_attacker(self, attack_type, lr=None, beta=None, verbose=False):

    kwargs = {'verbose': verbose}
    if lr is not None:
        kwargs['learning_rate'] = lr
    if beta is not None:
        kwargs['beta'] = beta

    if attack_type=='fast_gradient_method':
        return FastGradientMethod(self.art_net,
                                eps=self.epsilon,
                                eps_step=self.epsilon,
                                norm=self.norm)
    elif attack_type=='projected_gradient_descent':
        return ProjectedGradientDescentNumpy(self.art_net,
                                             eps=self.epsilon,
                                             eps_step=self.eps_iter,
                                             max_iter=self.max_iterations,
                                             norm=self.norm,
                                             **kwargs)
    elif attack_type=='pgd_early_stopping':
        return ProjectedGradientDescentNumpy(self.art_net,
                                             eps=self.epsilon,
                                             eps_step=self.eps_iter,
                                             max_iter=1,
                                             norm=self.norm,
                                             **kwargs)
    elif attack_type=='ART_AutoAttack':
        return AutoAttack(estimator=self.art_net,
                        eps=self.epsilon,
                        eps_step=self.eps_iter,
                        norm=self.norm)
    elif attack_type=='original_AutoAttack':
        return original_AutoAttack(self.net, 
                                   norm='L1', 
                                   eps=self.epsilon,
                                   device=device,
                                   version='standard',
                                   **kwargs)
    elif attack_type=='original_AutoAttack_apgd-only':
        attack= original_AutoAttack(self.net, 
                                   norm='L1', 
                                   eps=self.epsilon,
                                   device=device,
                                   version='custom',
                                   attacks_to_run=['apgd-ce'],
                                   **kwargs)
        attack.apgd.n_restarts=1
        attack.apgd.n_iter=self.max_iterations
        attack.apgd.verbose=False
        attack.apgd.use_largereps=False
        return attack
    elif attack_type=='auto_projected_gradient_descent':
        return AutoProjectedGradientDescent(estimator=self.art_net,
                                          eps=self.epsilon,
                                          eps_step=self.eps_iter,
                                          norm=self.norm,
                                          max_iter=self.max_iterations,
                                          **kwargs)
    elif attack_type=='brendel_bethge':
        return fb.attacks.L1BrendelBethgeAttack(steps=self.max_iterations)
    elif attack_type=='carlini_wagner_l2':
        return CarliniL2Method(self.art_net,
                               max_iter=self.max_iterations,
                               **kwargs)
    elif attack_type=='deep_fool':
        return DeepFool(self.art_net,
                      max_iter=self.max_iterations,
                      epsilon=self.eps_iter,
                      **kwargs)
    elif attack_type=='elastic_net':
        return ElasticNet(self.art_net,
                      max_iter=self.max_iterations,
                      **kwargs)
    elif attack_type=='elastic_net_L1_rule':
        return ElasticNet(self.art_net,
                      max_iter=self.max_iterations,
                      decision_rule='L1',
                      **kwargs)
    elif attack_type=='elastic_net_L1_rule_higher_beta':
        return ElasticNet(self.art_net,
                      max_iter=self.max_iterations,
                      decision_rule='L1',
                      beta=0.01,
                      **kwargs)
    elif attack_type=='exp_attack':
        return ExpAttack(self.art_net,
                      max_iter=self.max_iterations,
                      **kwargs)
    elif attack_type=='exp_attack_l1':
        return ExpAttackL1(self.art_net,
                      max_iter=self.max_iterations,
                      epsilon=self.epsilon,
                      **kwargs)
    elif attack_type=='exp_attack_l1_ada':
        return ExpAttackL1Ada(self.art_net,
                        max_iter=self.max_iterations,
                        epsilon=12,
                        **kwargs)
    elif attack_type=='custom_apgd':
        attack= AutoAttack_Custom(self.net, 
                                   norm='L1', 
                                   eps=self.epsilon,
                                   device=device,
                                   version='custom',
                                   attacks_to_run=['apgd-ce'],
                                   **kwargs)
        attack.apgd.n_restarts=1
        attack.apgd.n_iter=self.max_iterations
        attack.apgd.verbose=False
        attack.apgd.use_largereps=False
        attack.apgd.eot_iter=1
        return attack
    else:
        raise ValueError(f'Attack type "{attack_type}" not supported!')
