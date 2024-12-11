
from __future__ import absolute_import, division, print_function, unicode_literals, annotations

import numpy as np
from art.attacks.evasion import ElasticNet

import logging
from typing import TYPE_CHECKING
from cmath import inf
from scipy.special import lambertw
import numpy as np
import six
from tqdm.auto import trange

from art.config import ART_NUMPY_DTYPE
from art.attacks.attack import EvasionAttack
from art.estimators.estimator import BaseEstimator
from art.estimators.classification.classifier import ClassGradientsMixin
from art.utils import (
    compute_success,
    get_labels_np_array,
    check_and_transform_label_format,
)

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_CLASS_LOSS_GRADIENTS_TYPE

logger = logging.getLogger(__name__)


class ExpAttackL1(ElasticNet):
    attack_params = EvasionAttack.attack_params + [
        "confidence",
        "targeted",
        "learning_rate",
        "max_iter",
        "beta",
        "binary_search_steps",
        "initial_const",
        "batch_size",
        "decision_rule",
        "verbose",
    ]

    _estimator_requirements = (BaseEstimator, ClassGradientsMixin)
        
    def __init__(
        self,
        classifier: "CLASSIFIER_CLASS_LOSS_GRADIENTS_TYPE",
        confidence: float = 0.0,
        targeted: bool = False,
        learning_rate: float =1.0,
        max_iter: int = 100,
        beta: float =12,
        batch_size: int = 1,
        verbose: bool = True,
        smooth:float=False
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
        EvasionAttack.__init__(self,estimator=classifier)
        self.confidence = confidence
        self._targeted = targeted
        self.learning_rate = learning_rate
        self.binary_search_steps = 1
        self.max_iter = max_iter
        self.beta = beta
        self.initial_const = 1.0
        self.batch_size = batch_size
        self.decision_rule = 'L1'
        self.verbose = verbose
        self.eta=0.0
        self.smooth=smooth
        self._check_params()

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
        x_adv = x.astype(ART_NUMPY_DTYPE)

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
        nb_batches = int(np.ceil(x_adv.shape[0] / float(self.batch_size)))
        for batch_id in trange(nb_batches, desc="ExpAttackL1", disable=not self.verbose):
            batch_index_1, batch_index_2 = batch_id * self.batch_size, (batch_id + 1) * self.batch_size
            x_batch = x_adv[batch_index_1:batch_index_2]
            y_batch = y[batch_index_1:batch_index_2]
            x_adv[batch_index_1:batch_index_2] = self._generate_batch(x_batch, y_batch)

        # Apply clip
        if self.estimator.clip_values is not None:
            x_adv = np.clip(x_adv, self.estimator.clip_values[0], self.estimator.clip_values[1])

        # Compute success rate of the EAD attack
        logger.info(
            "Success rate of EAD attack: %.2f%%",
            100 * compute_success(self.estimator, x, y, x_adv, self.targeted, batch_size=self.batch_size),
        )

        return x_adv


    def _generate_batch(self, x_batch: np.ndarray, y_batch: np.ndarray) -> np.ndarray:
        """
        Run the attack on a batch of images and labels.

        :param x_batch: A batch of original examples.
        :param y_batch: A batch of targets (0-1 hot).
        :return: A batch of adversarial examples.
        """


        # Initialize binary search:
        c_current = self.initial_const * np.ones(x_batch.shape[0])
        c_lower_bound = np.zeros(x_batch.shape[0])
        c_upper_bound = 10e10 * np.ones(x_batch.shape[0])

        # Initialize the best distortions and best attacks globally
        o_best_dist = np.inf * np.ones(x_batch.shape[0])
        o_best_attack = x_batch.copy()



        # Run with 1 specific binary search step
        best_dist, best_label, best_attack = self._generate_bss(x_batch, y_batch, c_current)

        # Update best results so far
        o_best_attack[best_dist < o_best_dist] = best_attack[best_dist < o_best_dist]
        o_best_dist[best_dist < o_best_dist] = best_dist[best_dist < o_best_dist]

        return o_best_attack
    
    def _generate_bss(self, x_batch: np.ndarray, y_batch: np.ndarray, c_batch: np.ndarray) -> tuple:
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
        best_dist = np.inf * np.ones(x_batch.shape[0])
        best_label = [-np.inf] * x_batch.shape[0]
        best_attack = x_batch.copy()
        # Implement the algorithm 1 in the EAD paper
        delta= np.zeros(x_batch.shape)
        x_0=x_batch.copy()
        upper=1.0-x_0
        lower=0.0-x_0
        x_adv=x_0+delta
        self.eta=0.0
        for i_iter in range(self.max_iter):
            logger.debug("Iteration step %i out of %i", i_iter, self.max_iter)
            rnd=np.random.normal(size=x_0.shape)
            rnd1=np.random.normal()
            rnd=rnd/((np.linalg.norm(rnd)**2+rnd1**2)**0.5)
            # updating rule
            grad = self._gradient_of_loss(target=y_batch, x=x_batch, x_adv=x_adv.astype(np.float32)+(self.smooth*rnd).astype(np.float32), c_weight=c_batch)
            delta = self._md(grad,delta,lower,upper)
            
            x_adv=x_0+delta
            # Adjust the best result
            (logits, l1dist, l2dist, endist) = self._loss(x=x_batch, x_adv=x_adv.astype(np.float32))

            if self.decision_rule == "EN":
                zip_set = zip(endist, logits)
            elif self.decision_rule == "L1":
                zip_set = zip(l1dist, logits)
            elif self.decision_rule == "L2":
                zip_set = zip(l2dist, logits)
            else:  # pragma: no cover
                raise ValueError("The decision rule only supports `EN`, `L1`, `L2`.")

            for j, (distance, label) in enumerate(zip_set):
                if distance < best_dist[j] and compare(label, np.argmax(y_batch[j])):
                    best_dist[j] = distance
                    best_attack[j] = x_adv[j]
                    best_label[j] = label

        return best_dist, best_label, best_attack

    def _md(self,g,x,lower,upper):
        beta = 1.0 / g.size
        init=False
        if self.eta==0.0:
            init=True
            self.eta+=(np.linalg.norm((g).flatten(), ord=inf)**2)
        eta_t=np.sqrt(self.eta)/self.learning_rate
        z=(np.log(np.abs(x) / beta + 1.0)) * np.sign(x) - g/eta_t
        y_sgn=np.sign(z)
        y_val=beta*np.exp(np.abs(z))-beta
        v=self._project(y_sgn,y_val,beta,self.beta,lower,upper)
        if not init:
            D=np.maximum(np.linalg.norm(x.flatten(), ord=1),np.linalg.norm((v).flatten(), ord=1))
            self.eta+=(eta_t/(D+1)*np.linalg.norm((x-v).flatten(), ord=1))**2
            eta_t_1=np.sqrt(self.eta)/self.learning_rate
            v=(1.0-eta_t/eta_t_1)*x+eta_t/eta_t_1*v
        return v
    
    def _project(self, y_sgn,y_val, beta, D,l,u):
       
        if np.sum(y_val)<=D:
            return np.clip(y_sgn*y_val, l, u)
        c=np.where(y_sgn<=0,np.abs(l),u)
        normaliser=0.5
        phi=np.maximum(np.minimum(normaliser*(y_val+beta)-beta,c),0.0)
        radius=np.sum(phi)
        radius_l=0.0
        radius_u=1.0    
        min_y= np.min(y_val[np.nonzero(y_val)])
        while np.abs(radius_u-radius_l)>min_y:
            if radius>D:
                radius_u=normaliser
            
            else:
                radius_l=normaliser
            normaliser= 0.5*(radius_l+radius_u) 
            phi=np.maximum(np.minimum(normaliser*(y_val+beta)-beta,c),0.0)
            radius=np.sum(phi)
        y_bound=np.sum(phi[phi==c])
        active_index=np.logical_and(phi>0.0, phi<c)
        num_active=np.count_nonzero(active_index)
        normaliser=(D-np.sum(y_bound)+beta*num_active)/(np.sum(y_val[active_index]+beta))
        return np.maximum(np.minimum(normaliser*(y_val+beta)-beta,c),0.0)*y_sgn

    def _gradient_of_loss(
        self,
        target: np.ndarray,
        x: np.ndarray,
        x_adv: np.ndarray,
        c_weight: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the gradient of the loss function.

        :param target: An array with the target class (one-hot encoded).
        :param x: An array with the original input.
        :param x_adv: An array with the adversarial input.
        :param c_weight: Weight of the loss term aiming for classification as target.
        :return: An array with the gradient of the loss function.
        """
        # Compute the current predictions
        predictions = self.estimator.predict(np.array(x_adv, dtype=ART_NUMPY_DTYPE), batch_size=self.batch_size)

        if self.targeted:
            i_sub = np.argmax(target, axis=1)
            i_add = np.argmax(
                predictions * (1 - target) + (np.min(predictions, axis=1) - 1)[:, np.newaxis] * target,
                axis=1,
            )
        else:
            i_add = np.argmax(target, axis=1)
            i_sub = np.argmax(
                predictions * (1 - target) + (np.min(predictions, axis=1) - 1)[:, np.newaxis] * target,
                axis=1,
            )
        cost=i_add-i_sub
        
        loss_gradient = self.estimator.class_gradient(x_adv, label=i_add)
        loss_gradient -= self.estimator.class_gradient(x_adv, label=i_sub)
        loss_gradient = loss_gradient.reshape(x.shape)

        c_mult = c_weight
        for _ in range(len(x.shape) - 1):
            c_mult = c_mult[:, np.newaxis]


        loss_gradient *= c_mult
        if self.smooth:
        #loss_gradient += 2 * (x_adv - x)
            loss_gradient=loss_gradient-np.exp(-cost)/(1.0+np.exp(-cost))*loss_gradient
        # Set gradients where loss is constant to zero
        else:
            cond = (
                predictions[np.arange(x.shape[0]), i_add] - predictions[np.arange(x.shape[0]), i_sub] + self.confidence
            ) < 0
            loss_gradient[cond] = 0.0

        return loss_gradient
    
    def _loss(self, x: np.ndarray, x_adv: np.ndarray) -> tuple:
        """
        Compute the loss function values.

        :param x: An array with the original input.
        :param x_adv: An array with the adversarial input.
        :return: A tuple of shape `(np.ndarray, float, float, float)` holding the current predictions, l1 distance,
                 l2 distance and elastic net loss.
        """
        l1dist = np.sum(np.abs(x - x_adv).reshape(x.shape[0], -1), axis=1)
        l2dist = np.sum(np.square(x - x_adv).reshape(x.shape[0], -1), axis=1)
        endist =  l1dist + l2dist
        predictions = self.estimator.predict(np.array(x_adv, dtype=ART_NUMPY_DTYPE), batch_size=self.batch_size)

        return np.argmax(predictions, axis=1), l1dist, l2dist, endist
