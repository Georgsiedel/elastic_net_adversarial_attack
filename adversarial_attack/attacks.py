import torch
import foolbox as fb
from art.attacks.evasion import (FastGradientMethod,
                                 ProjectedGradientDescentNumpy,
                                 CarliniL2Method,
                                 DeepFool,
                                 ElasticNet,
                                 HopSkipJump)
from adversarial_attack.geometric_decision_based_attack import GeoDA
from adversarial_attack.rs_attacks import RSAttack
from adversarial_attack.exp_attack import ExpAttack
from adversarial_attack.exp_attack_l1 import ExpAttackL1
#from adversarial_attack.acc_exp_attack import AccExpAttack
from autoattack import AutoAttack
#from adversarial_attack.auto_attack.autoattack_custom import AutoAttack_Custom
from adversarial_attack.exp_attack_l1_ada import ExpAttackL1Ada
from adversarial_attack.exp_attack_l0 import ExpAttackL0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AdversarialAttacks:
  def __init__(self, art_net, net, epsilon, eps_iter, norm, max_iterations):
    self.art_net = art_net
    self.epsilon = epsilon
    self.eps_iter = eps_iter
    self.norm = norm
    self.max_iterations = max_iterations
    self.net = net
  def init_attacker(self, attack_type, max_batchsize=1, lr=None, beta=None, quantile=None, max_iterations_sweep=None, **kwargs):

    #kwargs = {'verbose': verbose}
    if lr is not None:
        kwargs['learning_rate'] = lr
    if beta is not None:
        kwargs['beta'] = beta
    if quantile is not None:
        kwargs['quantile'] = quantile
    if max_iterations_sweep is not None:
        max_iterations_sweep = int(max_iterations_sweep)
        if attack_type == 'sparse_rs_blackbox':
            kwargs['n_queries'] = max_iterations_sweep
        else:
            self.max_iterations = max_iterations_sweep

    if attack_type=='fast_gradient_method':
        return FastGradientMethod(self.art_net,
                                eps=self.epsilon,
                                eps_step=self.epsilon,
                                norm=self.norm
                                ), max_batchsize
    elif attack_type=='projected_gradient_descent':
        relevant_kwargs = {k: v for k, v in kwargs.items() if k in ["verbose"]}
        return ProjectedGradientDescentNumpy(self.art_net,
                                             eps=self.epsilon,
                                             eps_step=self.eps_iter,
                                             max_iter=self.max_iterations,
                                             norm=self.norm,
                                             **relevant_kwargs
                                             ), max_batchsize
    elif attack_type=='pgd_early_stopping':
        return ProjectedGradientDescentNumpy(self.art_net,
                                             eps=self.epsilon,
                                             eps_step=self.eps_iter,
                                             max_iter=1,
                                             norm=self.norm,
                                             **kwargs
                                             ), 1
    elif attack_type=='AutoAttack':
        relevant_kwargs = {k: v for k, v in kwargs.items() if k in ["verbose"]}
        return AutoAttack(self.net, 
                                   norm='L1', 
                                   eps=self.epsilon,
                                   device=device,
                                   version='standard',
                                   **relevant_kwargs
                                   ), max_batchsize

    elif attack_type=='custom_apgd':
        relevant_kwargs = {k: v for k, v in kwargs.items() if k in ["verbose"]}
        attack= AutoAttack(self.net, 
                                   norm='L1', 
                                   eps=self.epsilon,
                                   device=device,
                                   version='custom',
                                   attacks_to_run=['apgd-ce'],
                                   **relevant_kwargs)
        attack.apgd.n_restarts=1
        attack.apgd.n_iter=self.max_iterations
        attack.apgd.verbose=False
        attack.apgd.use_largereps=False
        attack.apgd.eot_iter=1
        attack.apgd.use_rs = False

        return attack, max_batchsize
    
    elif attack_type=='deep_fool':
        return DeepFool(self.art_net,
                      max_iter=self.max_iterations,
                      epsilon=self.eps_iter,
                      **kwargs
                      ), max_batchsize
    elif attack_type=='brendel_bethge':
        att = fb.attacks.L1BrendelBethgeAttack(steps=self.max_iterations, 
                                               init_attack=fb.attacks.SaltAndPepperNoiseAttack(steps=5000, 
                                                                                               across_channels=False))
        return att, max_batchsize
    elif attack_type=='pointwise_blackbox':
        #https://openreview.net/pdf?id=S1EHOsC9tX
        att = fb.attacks.pointwise.PointwiseAttack(init_attack=fb.attacks.SaltAndPepperNoiseAttack(steps=5000, across_channels=False))
        att._distance = fb.distances.l1
        return att, max_batchsize
    elif attack_type=='boundary_blackbox':
        att = fb.attacks.boundary_attack.BoundaryAttack(steps=25000)
        att._distance = fb.distances.l1
        return att, max_batchsize
    elif attack_type=='hopskipjump_blackbox':
        return HopSkipJump(self.art_net,
                           max_iter=64, 
                           **kwargs
                           ), max_batchsize
    elif attack_type=='sparse_rs_custom_L1_blackbox':
        #https://ojs.aaai.org/index.php/AAAI/article/view/20595/20354
        assert self.norm == 1, "only norm=1 translates correctly into sparse_rs attack budget"
        return RSAttack(predict=self.net,
                        norm='L0+L1', #'L0+L1' to reject L0 perturbations that are larger than the L1 epsilon. Combine with sensible eps parameter below, otherwise you will reject everything
                        eps=int(self.epsilon/3*2), # approximating the L0 epsilon that corresponds to the L1 epsilon on 3 channels
                        eps_L1=self.epsilon,
                        device=device,
                        **kwargs
                        ), max_batchsize
    elif attack_type=='sparse_rs_blackbox':
        #https://ojs.aaai.org/index.php/AAAI/article/view/20595/20354
        assert self.norm == 1, "only norm=1 translates correctly into sparse_rs attack budget"
        return RSAttack(predict=self.net,
                        norm='L0', #'L0+L1' to reject L0 perturbations that are larger than the L1 epsilon. Combine with sensible eps parameter below, otherwise you will reject everything
                        eps=int(self.epsilon/3*2), # approximating the L0 epsilon that corresponds to the L1 epsilon on 3 channels
                        device=device,
                        **kwargs
                        ), max_batchsize
    elif attack_type=='geoda_blackbox':
        #this is the ART implementation, but without a deprecated np function
        #https://openaccess.thecvf.com/content_CVPR_2020/papers/Rahmati_GeoDA_A_Geometric_Framework_for_Black-Box_Adversarial_Attacks_CVPR_2020_paper.pdf
        return GeoDA(self.art_net,
                     max_iter=10000,
                        **kwargs
                        ), max_batchsize
    elif attack_type=='carlini_wagner_l2':
        return CarliniL2Method(self.art_net,
                               max_iter=self.max_iterations,
                               **kwargs
                               ), 1
    elif attack_type=='elastic_net':
        relevant_kwargs = {k: v for k, v in kwargs.items() if k in ["verbose", "learning_rate", "beta"]}
        return ElasticNet(self.art_net,
                      max_iter=self.max_iterations,
                      **relevant_kwargs
                      ), 1
    elif attack_type=='elastic_net_L1_rule_higher_beta':
        relevant_kwargs = {k: v for k, v in kwargs.items() if k in ["verbose", "learning_rate", "beta"]}
        return ElasticNet(self.art_net,
                      max_iter=self.max_iterations,
                      decision_rule='L1',
                      beta=0.01,
                      **relevant_kwargs
                      ), 1
    elif attack_type=='exp_attack':
        relevant_kwargs = {k: v for k, v in kwargs.items() if k in ["verbose", "learning_rate", "beta"]}
        return ExpAttack(self.art_net,
                      max_iter=self.max_iterations,
                      **relevant_kwargs
                      ), 1
    elif attack_type=='exp_attack_blackbox':
        return ExpAttack(self.art_net,
                      max_iter=self.max_iterations,
                      perturbation_blackbox=0.001,
                      samples_blackbox=100,
                      final_quantile=0.0,
                      **kwargs
                      ), 1
    elif attack_type=='exp_attack_blackbox_L1_rule_higher_beta':
        return ExpAttack(self.art_net,
                      max_iter=self.max_iterations,
                      decision_rule='L1',
                      beta=0.01,
                      perturbation_blackbox=0.001,
                      samples_blackbox=100,
                      **kwargs
                      ), 1
    elif attack_type=='exp_attack_l1':
        relevant_kwargs = {k: v for k, v in kwargs.items() if k in ["verbose", "learning_rate", "beta"]}
        return ExpAttackL1(self.art_net,
                      max_iter=self.max_iterations,
                      epsilon=self.epsilon,
                      **relevant_kwargs
                      ), 1
    elif attack_type=='exp_attack_l1_blackbox':
        return ExpAttackL1(self.art_net,
                      max_iter=self.max_iterations,
                      epsilon=self.epsilon,
                      perturbation_blackbox=0.001,
                      samples_blackbox=100,
                      quantile=0.0,
                      **kwargs
                      ), 1
    elif attack_type=='exp_attack_l1_ada_bb':
        return ExpAttackL1Ada(self.art_net,
                        max_iter=self.max_iterations,
                        epsilon=self.epsilon,
                        perturbation_blackbox=0.001,
                        samples_blackbox=100,
                        quantile=0.0,
                        **kwargs
                        ), 1
    elif attack_type=='exp_attack_l0':
        return ExpAttackL0(self.art_net,
                        max_iter=self.max_iterations,
                        epsilon=self.epsilon,
                        perturbation_blackbox=0.0,
                        samples_blackbox=100,
                        **kwargs
                        ), 1
    elif attack_type=='exp_attack_l0_bb':
        return ExpAttackL0(self.art_net,
                        max_iter=self.max_iterations,
                        epsilon=self.epsilon,
                        perturbation_blackbox=0.001,
                        samples_blackbox=100,
                        **kwargs
                        ), 1
    else:
        raise ValueError(f'Attack type "{attack_type}" not supported!')
