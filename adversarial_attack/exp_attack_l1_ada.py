
from __future__ import absolute_import, division, print_function, unicode_literals, annotations

import numpy as np

import logging
from typing import TYPE_CHECKING
from cmath import inf
from scipy.special import lambertw
import numpy as np
from tqdm.auto import trange

from art.estimators.estimator import BaseEstimator, LossGradientsMixin
from art.estimators.classification.classifier import ClassifierMixin
from art.estimators.classification import  PyTorchClassifier


from art.config import ART_NUMPY_DTYPE
from art.attacks.attack import EvasionAttack
from art.estimators.estimator import BaseEstimator
from art.utils import (
    compute_success,
    get_labels_np_array,
    check_and_transform_label_format,
    is_probability
)

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_LOSS_GRADIENTS_TYPE

logger = logging.getLogger(__name__)


class ExpAttackL1Ada(EvasionAttack):
    attack_params = EvasionAttack.attack_params + [
        "targeted",
        "learning_rate",
        "max_iter",
        "beta",
        "epsilon",
        "decision_rule",
        "quantile",
        "verbose"
    ]
    _estimator_requirements = (BaseEstimator, LossGradientsMixin, ClassifierMixin)
    _predefined_losses = [None, "cross_entropy", "difference_logits_ratio"]
        
    def __init__(
        self,
        estimator: "CLASSIFIER_LOSS_GRADIENTS_TYPE",
        targeted: bool = False,
        learning_rate: float =1.0,
        max_iter: int = 100,
        beta: float =2,
        verbose: bool = True,
        epsilon:float=12,
        quantile:float=0.0,
        loss_type= "cross_entropy"
    ) -> None:
        """
        Create an ElasticNet attack instance.

        :param classifier: A trained classifier.
        :param confidence: Confidence of adversarial examples: a higher value produces examples that are farther
               away, from the original input, but classified with higher confidence as the target class.
        :param targeted: Should the attack target one specific class.
        :param learning_rate: The initial learning rate for the attack algorithm. Smaller values produce better
               results but are slower to converge.
        :param binary_search_steps: Number of times to adjust constant with binary search (positive value).
        :param max_iter: The maximum number of iterations.
        :param beta: Hyperparameter trading off L2 minimization for L1 minimization.
        :param initial_const: The initial trade-off constant `c` to use to tune the relative importance of distance
               and confidence. If `binary_search_steps` is large, the initial constant is not important, as discussed in
               Carlini and Wagner (2016).
        :param batch_size: Internal size of batches on which adversarial samples are generated.
        :param decision_rule: Decision rule. 'EN' means Elastic Net rule, 'L1' means L1 rule, 'L2' means L2 rule.
        :param verbose: Show progress bars.
        """

        import torch

        if loss_type == "cross_entropy":
            if is_probability(
                estimator.predict(x=np.ones(shape=(1, *estimator.input_shape), dtype=np.float32))
            ):
                raise ValueError(  # pragma: no cover
                    "The provided estimator seems to predict probabilities. If loss_type='cross_entropy' "
                    "the estimator has to to predict logits."
                )

            # modification for image-wise stepsize update
            class CrossEntropyLossTorch(torch.nn.modules.loss._Loss):
                """Class defining cross entropy loss with reduction options."""

                def __init__(self, reduction="mean"):
                    super().__init__()
                    self.ce_loss = torch.nn.CrossEntropyLoss(reduction="none")
                    self.reduction = reduction

                def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
                    if self.reduction == "mean":
                        return self.ce_loss(y_pred, y_true).mean()
                    if self.reduction == "sum":
                        return self.ce_loss(y_pred, y_true).sum()
                    if self.reduction == "none":
                        return self.ce_loss(y_pred, y_true)
                    raise NotImplementedError()

                def forward(
                    self, input: torch.Tensor, target: torch.Tensor  # pylint: disable=redefined-builtin
                ) -> torch.Tensor:
                    """
                    Forward method.

                    :param input: Predicted labels of shape (nb_samples, nb_classes).
                    :param target: Target labels of shape (nb_samples, nb_classes).
                    :return: Difference Logits Ratio Loss.
                    """
                    return self.__call__(y_pred=input, y_true=target)

            _loss_object_pt: torch.nn.modules.loss._Loss = CrossEntropyLossTorch(reduction="none")



        elif loss_type == "difference_logits_ratio":
            if is_probability(
                estimator.predict(x=np.ones(shape=(1, *estimator.input_shape), dtype=ART_NUMPY_DTYPE))
            ):
                raise ValueError(  # pragma: no cover
                    "The provided estimator seems to predict probabilities. "
                    "If loss_type='difference_logits_ratio' the estimator has to to predict logits."
                )

            class DifferenceLogitsRatioPyTorch(torch.nn.modules.loss._Loss):
                """
                Callable class for Difference Logits Ratio loss in PyTorch.
                """

                def __init__(self):
                    super().__init__()
                    self.reduction = "mean"

                def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
                    if isinstance(y_true, np.ndarray):
                        y_true = torch.from_numpy(y_true)
                    if isinstance(y_pred, np.ndarray):
                        y_pred = torch.from_numpy(y_pred)

                    y_true = y_true.float()

                    i_y_true = torch.argmax(y_true, dim=1)
                    i_y_pred_arg = torch.argsort(y_pred, dim=1)
                    i_z_i_list = []

                    for i in range(y_true.shape[0]):
                        if i_y_pred_arg[i, -1] != i_y_true[i]:
                            i_z_i_list.append(i_y_pred_arg[i, -1])
                        else:
                            i_z_i_list.append(i_y_pred_arg[i, -2])

                    i_z_i = torch.stack(i_z_i_list)

                    z_1 = y_pred[:, i_y_pred_arg[:, -1]]
                    z_3 = y_pred[:, i_y_pred_arg[:, -3]]
                    z_i = y_pred[:, i_z_i]
                    z_y = y_pred[:, i_y_true]

                    z_1 = torch.diagonal(z_1)
                    z_3 = torch.diagonal(z_3)
                    z_i = torch.diagonal(z_i)
                    z_y = torch.diagonal(z_y)

                    # modification for image-wise stepsize update
                    dlr = (-(z_y - z_i) / (z_1 - z_3)).float()
                    if self.reduction == "mean":
                        return dlr.mean()
                    if self.reduction == "sum":
                        return dlr.sum()
                    if self.reduction == "none":
                        return dlr
                    raise NotImplementedError()

                def forward(
                    self, input: torch.Tensor, target: torch.Tensor  # pylint: disable=redefined-builtin
                ) -> torch.Tensor:
                    """
                    Forward method.

                    :param input: Predicted labels of shape (nb_samples, nb_classes).
                    :param target: Target labels of shape (nb_samples, nb_classes).
                    :return: Difference Logits Ratio Loss.
                    """
                    return self.__call__(y_true=target, y_pred=input)

            _loss_object_pt = DifferenceLogitsRatioPyTorch()


        else:
            raise NotImplementedError()

        estimator = PyTorchClassifier(
            model=estimator.model,
            loss=_loss_object_pt,
            input_shape=estimator.input_shape,
            nb_classes=estimator.nb_classes,
            optimizer=None,
            channels_first=estimator.channels_first,
            clip_values=estimator.clip_values,
            preprocessing_defences=estimator.preprocessing_defences,
            postprocessing_defences=estimator.postprocessing_defences,
            preprocessing=estimator.preprocessing,
            device_type=str(estimator._device),
        )
        super().__init__(estimator=estimator)
        self._targeted = targeted
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.beta = beta
        self.verbose = verbose
        self.eta=0.0
        self.epsilon=epsilon
        self._check_params()
        self.loss_type=loss_type
        self.quantile=quantile

    def generate(self, x: np.ndarray, y: np.ndarray | None = None, **kwargs) -> np.ndarray:
        """
        Generate adversarial samples and return them in an array.

        :param x: An array with the original inputs to be attacked.
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
                  (nb_samples,). If `self.targeted` is true, then `y` represents the target labels. Otherwise, the
                  targets are the original class labels.
        :return: An array holding the adversarial examples.
        """
        if y is not None:
            y = check_and_transform_label_format(y, nb_classes=self.estimator.nb_classes)
        x_batch = x.astype(ART_NUMPY_DTYPE)

        # Assert that, if attack is targeted, y is provided:
        if self.targeted and y is None:  # pragma: no cover
            raise ValueError("Target labels `y` need to be provided for a targeted attack.")

        # No labels provided, use model prediction as correct class
        if y is None:
            y = get_labels_np_array(self.estimator.predict(x, batch_size=self.batch_size))

        if self.estimator.nb_classes == 2 and y.shape[1] == 1:  # pragma: no cover
            raise ValueError(
                "This attack has not yet been tested for binary classification with a single output classifier."
            )

        # Compute adversarial examples with implicit batching
        for batch_id in trange( x_batch.shape[0], desc="ExpAttackL1Ada", disable=not self.verbose):
            x_i = x_batch[batch_id]
            y_i= y[batch_id]
            x_adv,success = self._generate_bss(x_i,y_i)
        # Apply clip
        
        if self.estimator.clip_values is not None:
            x_adv = np.clip(x_adv, self.estimator.clip_values[0], self.estimator.clip_values[1])

        return x_adv


    
    def _generate_bss(self, x_batch: np.ndarray, y_batch: np.ndarray) -> tuple:
        """
        Generate adversarial examples for a batch of inputs with a specific batch of constants.

        :param x_batch: A batch of original examples.
        :param y_batch: A batch of targets (0-1 hot).
        :param c_batch: A batch of constants.
        :return: A tuple of the best elastic distances, best labels, best attacks
        """

        def compare(o_1, o_2):
            if self.targeted:
                return o_1 == o_2
            return o_1 != o_2

        # Initialize the best distortions and best changed labels and best attacks
        best_dist = np.inf 
        best_attack = x_batch.copy()
        x_0=x_batch.copy()
        upper=1.0-x_0
        lower=0.0-x_0
        delta=np.zeros(x_0.shape)
        x_adv=x_0+delta
        success=[]

        for i_iter in range(self.max_iter):

            grad = -self.estimator.loss_gradient(x_adv.astype(ART_NUMPY_DTYPE), y_batch) * (1 - 2 * int(self.targeted))
            
          
            grad_val=np.abs(grad)
            topk_val=np.quantile(grad_val,self.quantile)
            grad[grad_val<topk_val]=0.0             
            
            delta = self._md(grad,delta, lower,upper)
            

            logger.debug("Iteration step %i out of %i", i_iter, self.max_iter)
            x_adv=x_0+delta
            logits, l1dist = self._loss(x=x_batch, x_adv=x_adv.astype(ART_NUMPY_DTYPE))
            if l1dist < best_dist and compare(logits, np.argmax(y_batch)):
                best_dist = logits
                best_attack = x_adv
            success.append(compare(logits, np.argmax(y_batch[j])))
        return best_attack, np.array(success)

    def _md(self,g,x,lower,upper):
        beta=self.beta
        dual_x=(np.log(np.abs(x) / beta + 1.0)) * np.sign(x)
        self.eta+=(np.max(np.abs(g))**2)
        eta_t=np.sqrt(self.eta)/self.learning_rate
        descent=g/eta_t
        z=dual_x -descent
        z_sgn=np.sign(z)
        z_val=np.abs(z)
        v=self._project(z_sgn,z_val,beta,self.epsilon,lower,upper)
        return v

    def _reg(self,x,beta):
        return np.sum(np.log(np.abs(x)/beta+1.0)*(np.abs(x)+beta)-abs(x))
    def _reg_prim(self,x,beta):
        return np.log(np.abs(x)/beta+1.0)*np.sign(x)



    def _project(self, y_sgn,y_val, beta, D,l,u):
        log_beta=np.log(beta)
        y_val_max= np.max(y_val)
        dim=y_val.size

        # inside desicion set
        if np.log(np.sum(np.exp(y_val+log_beta-y_val_max)))+y_val_max<=np.log(D+dim*beta):
            phi=y_sgn*(np.exp(y_val+log_beta)-beta)
            if np.all(phi>=l) and np.all(phi<=u):
                return phi
        
        # otherwise it has to be mapped to l1 sphere
        c=np.where(y_sgn<=0,np.abs(l),u)
        log_c_beta=np.log(c+beta)
        lam_l=-y_val
        lam_u=np.minimum(0,-y_val-log_beta+log_c_beta)
    
        lam=np.stack((lam_l,lam_u))
        lam=lam.reshape(-1)
        sort_idx=np.argsort(lam)
        idx_l=0
        idx_u=sort_idx.size-1
        while idx_u-idx_l>1:
            idx=(idx_u+idx_l)//2
            normaliser=lam[sort_idx[idx]]
            phi=np.exp(np.maximum(np.minimum(y_val+log_beta+normaliser,log_c_beta),log_beta))-beta
            radius=np.sum(phi)
            if radius>D:
                idx_u=idx
            elif radius<D:
                idx_l=idx
            else:
                idx_u=idx
                idx_l=idx
        if lam[sort_idx[idx_l]]<0:
            phi[lam_l>=lam[sort_idx[idx_u]]]=0
            phi[lam_u<=lam[sort_idx[idx_l]]]=c[lam_u<=lam[sort_idx[idx_l]]]
            active_index=np.logical_and(lam_l<lam[sort_idx[idx_u]] ,lam_u>lam[sort_idx[idx_l]])
            num_active=np.count_nonzero(active_index)
            y_bound=np.sum(c[lam_u<=lam[sort_idx[idx_l]]])
            if num_active!=0:
            #nummerical instability 
                #if(y_bound>D):
                    #print(f"norm of active coordinatte {y_bound} larger than radius")
                    #assert(y_bound<=D)
                y_max_active=np.max(y_val[active_index])
                normaliser=np.log(np.sum(np.exp(y_val[active_index]-y_max_active+log_beta)))-np.log(D-y_bound+num_active*beta)+y_max_active
                phi[active_index]=np.exp(y_val[active_index]+np.log(beta)-normaliser)-beta
        #radius=np.sum(phi)
        #if np.abs(radius)>self.epsilon+1.0: 
        #print(f"radius {np.sum(np.abs(phi))}")
        phi=phi*y_sgn
        return phi
    


    def _loss(self, x: np.ndarray, x_adv: np.ndarray) -> tuple:
        """
        Compute the loss function values.

        :param x: An array with the original input.
        :param x_adv: An array with the adversarial input.
        :return: A tuple of shape `(np.ndarray, float, float, float)` holding the current predictions, l1 distance,
                 l2 distance and elastic net loss.
        """
        l1dist = np.sum(np.abs(x - x_adv))
        predictions = self.estimator.predict(np.array(x_adv, dtype=ART_NUMPY_DTYPE))
        return np.argmax(predictions), l1dist
