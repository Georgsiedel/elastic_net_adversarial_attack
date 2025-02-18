import torch
import foolbox as fb
from art.attacks.evasion import (FastGradientMethod,
                                 ProjectedGradientDescentNumpy,
                                 AutoProjectedGradientDescent,
                                 AutoAttack,
                                 CarliniL2Method,
                                 DeepFool,
                                 ElasticNet)
from adversarial_attack.geometric_decision_based_attack import GeoDA
from adversarial_attack.rs_attacks import RSAttack
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
  def init_attacker(self, attack_type, lr=None, beta=None, quantile=None, max_iterations_sweep=None, verbose=False):

    kwargs = {'verbose': verbose}
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
                                norm=self.norm,
                                **kwargs)
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
                        norm=self.norm,
                        **kwargs)
    elif attack_type=='original_AutoAttack':
        return original_AutoAttack(self.net, 
                                   norm='L1', 
                                   eps=self.epsilon,
                                   device=device,
                                   version='standard',
                                   **kwargs)
    elif attack_type=='original_AutoAttack_apgd_only':
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
    elif attack_type=='pointwise_blackbox':
        #https://openreview.net/pdf?id=S1EHOsC9tX
        att = fb.attacks.pointwise.PointwiseAttack(init_attack=fb.attacks.SaltAndPepperNoiseAttack(steps=5000, across_channels=False))
        att._distance = fb.distances.l1
        return att
    elif attack_type=='pointwise_blackbox+boundary':
        #https://openreview.net/pdf?id=S1EHOsC9tX
        att = fb.attacks.pointwise.PointwiseAttack(init_attack=fb.attacks.boundary_attack.BoundaryAttack())
        att._distance = fb.distances.l1
        return att
    elif attack_type=='pointwise_blackbox+hopskipjump':
        #https://openreview.net/pdf?id=S1EHOsC9tX
        att = fb.attacks.pointwise.PointwiseAttack(init_attack=fb.attacks.hop_skip_jump.HopSkipJumpAttack())
        att._distance = fb.distances.l1
        return att
    elif attack_type=='sparse_rs_blackbox':
        #https://ojs.aaai.org/index.php/AAAI/article/view/20595/20354
        assert self.norm == 1, "only norm=1 translates correctly into sparse_rs attack budget"
        return RSAttack(predict=self.net,
                        norm='L0+L1', #'L0+L1' to reject L0 perturbations that are larger than the L1 epsilon. Combine with sensible eps parameter below, otherwise you will reject everything
                        eps=int(self.epsilon/3*2), # approximating the L0 epsilon that corresponds to the L1 epsilon on 3 channels
                        eps_L1=self.epsilon,
                        device=device,
                        **kwargs
                        )
    elif attack_type=='brendel_bethge':
        return fb.attacks.L1BrendelBethgeAttack(steps=self.max_iterations)
    elif attack_type=='geoda_blackbox':
        #this is the ART implementation, but without a deprecated np function
        #https://openaccess.thecvf.com/content_CVPR_2020/papers/Rahmati_GeoDA_A_Geometric_Framework_for_Black-Box_Adversarial_Attacks_CVPR_2020_paper.pdf
        fb.attacks.sparse_l1_descent_attack.SparseL1DescentAttack
        return GeoDA(self.art_net,
                        batch_size=1,
                        norm=self.norm,
                        max_iter=4000,
                        lambda_param=0.6,
                        **kwargs)
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
    elif attack_type=='exp_attack_l1_blackbox':
        return ExpAttackL1(self.art_net,
                      max_iter=self.max_iterations,
                      epsilon=self.epsilon,
                      quantile=0.0,
                      perturbation_blackbox=0.001,
                      samples_blackbox=100,
                      **kwargs)
    elif attack_type=='exp_attack_l1_ada':
        return ExpAttackL1Ada(self.art_net,
                        max_iter=self.max_iterations,
                        epsilon=self.epsilon,
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
