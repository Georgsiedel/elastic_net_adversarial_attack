from autoattack import AutoAttack
from .apgd_custom import APGDAttackCustom
class AutoAttack_Custom(AutoAttack):
    def __init__(self, model, norm='Linf', eps=.3, seed=None, verbose=True,
                 attacks_to_run=[], version='standard', is_tf_model=False,
                 device='cuda', log_path=None):
        super().__init__( model, norm, eps, seed, verbose, attacks_to_run, version, is_tf_model, device, log_path)
        self.apgd = APGDAttackCustom(self.model,n_restarts=1, n_iter=100, verbose=False,
                eps=self.epsilon, norm=self.norm, eot_iter=1, rho=.75, seed=self.seed,
                device=self.device, logger=self.logger)

