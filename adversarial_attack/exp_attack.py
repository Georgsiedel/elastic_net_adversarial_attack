
from __future__ import absolute_import, division, print_function, unicode_literals, annotations

import numpy as np
from art.attacks.evasion import ElasticNet

import logging
from typing import TYPE_CHECKING, Optional
from cmath import inf
from scipy.special import lambertw
import numpy as np
import six
from tqdm.auto import trange
import json
import os

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


class ExpAttack(ElasticNet):
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
        "perturbation_blackbox",
        "samples_blackbox",
        "max_batchsize_blackbox"
    ]

    _estimator_requirements = (BaseEstimator, ClassGradientsMixin)
        
    def __init__(
        self,
        classifier: "CLASSIFIER_CLASS_LOSS_GRADIENTS_TYPE",
        confidence: float = 0.0,
        targeted: bool = False,
        learning_rate: float = 1.0,
        binary_search_steps: int = 9,
        max_iter: int = 100,
        l1: float = 0.001,
        beta:float=1.0,
        initial_const: float = 1e-3,
        batch_size: int = 250,
        decision_rule: str = "EN",
        verbose: bool = True,
        perturbation_blackbox:float=0.0,
        samples_blackbox:int=100,
        max_batchsize_blackbox:int=100,
        track_c: Optional[str] = None,
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
        self.binary_search_steps = binary_search_steps
        self.max_iter = max_iter
        self.l1 = l1
        self.initial_const = initial_const
        self.batch_size = batch_size
        self.decision_rule = decision_rule
        self.verbose = verbose
        self.eta=0.0
        self.beta=beta
        self.perturbation_blackbox = abs(perturbation_blackbox)
        self.samples_blackbox = samples_blackbox
        self.max_batchsize_blackbox = max_batchsize_blackbox
        self.track_c=track_c
        if self.track_c:
            self.final_c_list = self.load_c_list(track_c, decision_rule)
        else:
            self.final_c_list = None
        self._check_params()

    def load_c_list(self, track_c, decision_rule):
        # Construct the file path
        json_file_path = f'./results/c_values_ead/{track_c}_{decision_rule}.json'
        
        # Check if the file exists
        if not os.path.exists(json_file_path):
            raise FileNotFoundError(f"File not found: {json_file_path}")
        
        # Load the JSON data
        with open(json_file_path, 'r') as json_file:
            c_dict = json.load(json_file)
        
        # Extract and return the list
        return c_dict.get("final_c", None)

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
        for batch_id in trange(nb_batches, desc="Exp Gradient for Elastic Net", disable=not self.verbose):
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

        if self.final_c_list is not None:
            
            # Reduce bss to 1
            self.binary_search_steps = 1
            batchsize = x_batch.shape[0]
            # Ensure there are enough constants for the batch
            if len(self.final_c_list) < batchsize:
                raise ValueError("Not enough constants in final_c_list for the batch size.")
            # Extract the first numbers for the current batch
            c_current = np.array(self.final_c_list[:batchsize])

            # Remove these numbers from the list
            self.final_c_list = self.final_c_list[batchsize:]

        # Initialize the best distortions and best attacks globally
        o_best_dist = np.inf * np.ones(x_batch.shape[0])
        o_best_attack = x_batch.copy()

        # Start with a binary search
        for bss in range(self.binary_search_steps):
            logger.debug(
                "Binary search step %i out of %i (c_mean==%f)",
                bss,
                self.binary_search_steps,
                np.mean(c_current),
            )

            # Run with 1 specific binary search step
            best_dist, best_label, best_attack = self._generate_bss(x_batch, y_batch, c_current)
            #print(f"current {c_current} with best dist {best_dist}")
            # Update best results so far
            o_best_attack[best_dist < o_best_dist] = best_attack[best_dist < o_best_dist]
            o_best_dist[best_dist < o_best_dist] = best_dist[best_dist < o_best_dist]

            # Adjust the constant as needed
            c_current, c_lower_bound, c_upper_bound = super()._update_const(
                y_batch, best_label, c_current, c_lower_bound, c_upper_bound
            )

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

        self.eta=np.zeros(shape=(x_0.shape[0],1,1,1))
        for i_iter in range(self.max_iter):
            logger.debug("Iteration step %i out of %i", i_iter, self.max_iter)
            # updating rule
            grad = self._gradient_of_loss(target=y_batch, x=x_batch, x_adv=x_adv.astype(ART_NUMPY_DTYPE), c_weight=c_batch)             
            delta = self._md(grad,delta, lower,upper)
            x_adv=x_0+delta
            # Adjust the best result
            (logits, l1dist, l2dist, endist) = self._loss(x=x_batch, x_adv=x_adv.astype(ART_NUMPY_DTYPE))

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
        beta = self.beta
        dual_x=(np.log(np.abs(x) / beta + 1.0)) * np.sign(x)
        self.eta+=np.max(g**2,axis=(1,2,3))[:, np.newaxis, np.newaxis, np.newaxis]
        eta_t=np.sqrt(self.eta)/self.learning_rate
        dual_x=(np.log(np.abs(x) / beta + 1.0)) * np.sign(x)
        z=dual_x- g/eta_t
        v_sgn = np.sign(z)
        a = beta
        b = 2.0/eta_t
        c = np.minimum(self.l1/eta_t- np.abs(z),0.0)
        abc=-c+np.log(a*b)+a*b
        v_val = np.where(abc>=15.0,np.log(abc)-np.log(np.log(abc))+np.log(np.log(abc))/np.log(abc), lambertw( np.exp(abc), k=0).real)/b-a
        v = v_sgn * v_val
        v = np.clip(v, lower, upper)

        #divergence=self._bd(v,x,beta)[:, np.newaxis, np.newaxis, np.newaxis]
        #dist=(eta_t**2)*divergence
        #self.eta+=dist 
        #eta_t_1=(np.maximum(np.sqrt(self.eta),self.tol)/self.learning_rate)
        #gamma=eta_t/eta_t_1
        #v=(1.0-gamma)*x+gamma*v 
        return v
    

    def _bd(self,x: np.ndarray,y:  np.ndarray,beta :float) -> np.ndarray:
        return self._reg(x,beta)-self._reg(y,beta)-np.sum(self._reg_prim(y,beta)*(x-y),axis=(1,2,3))
    
    def _reg(self,x: np.ndarray,beta :float)-> np.ndarray:
        return np.sum(np.log(np.abs(x)/beta+1.0)*(np.abs(x)+beta)-abs(x),axis=(1,2,3))
    
    def _reg_prim(self,x: np.ndarray, beta :float)-> np.ndarray:
        return np.log(np.abs(x)/beta+1.0)*np.sign(x)
    
    
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
        
        if self.perturbation_blackbox > 0:
            loss_gradient = self._estimate_class_gradients_blackbox(x_adv, i_add, i_sub, predictions)
        else:
            loss_gradient = self.estimator.class_gradient(x_adv, label=i_add)
            loss_gradient -= self.estimator.class_gradient(x_adv, label=i_sub)
        loss_gradient = loss_gradient.reshape(x.shape)

        c_mult = c_weight
        for _ in range(len(x.shape) - 1):
            c_mult = c_mult[:, np.newaxis]


        loss_gradient *= c_mult
        cond = (
            predictions[np.arange(x.shape[0]), i_add] - predictions[np.arange(x.shape[0]), i_sub] + self.confidence
        ) < 0
        loss_gradient[cond] = 0.0

        return loss_gradient
    
    def _estimate_class_gradients_blackbox(self, x_adv, i_add, i_sub, predictions):
        """
        Efficient batched gradient estimation using black-box sampling.
        """
        # Initialize the gradient estimate
        gradient_estimate_add = np.zeros_like(x_adv)
        gradient_estimate_sub = np.zeros_like(x_adv)

        # Extend x_adv to a batch with size equal to self.samples_blackbox
        x_adv_extended = np.repeat(x_adv, self.samples_blackbox, axis=0)
        
        # Generate Rademacher samples (±1) for the extended batch
        vi = np.random.choice([-1, 1], size=x_adv_extended.shape).astype(np.float32)

        # Compute perturbed inputs for all samples in the extended batch
        rand_perturbed_inputs = np.clip(x_adv_extended + self.perturbation_blackbox * vi, 0.0, 1.0)

        # Split into batches of size self.max_batchsize_blackbox
        num_batches = int(np.ceil(self.samples_blackbox / self.max_batchsize_blackbox))
        
        for batch_idx in range(num_batches):
            # Determine the range of indices for this batch
            start_idx = batch_idx * self.max_batchsize_blackbox
            end_idx = min((batch_idx + 1) * self.max_batchsize_blackbox, self.samples_blackbox)
            batch_size = end_idx - start_idx
            
            # Slice the current batch
            batch_perturbed_inputs = rand_perturbed_inputs[start_idx:end_idx]
            batch_vi = vi[start_idx:end_idx]

            # Compute classes for the perturbed inputs and original inputs
            pred_perturbed = self.estimator.predict(np.array(batch_perturbed_inputs, dtype=ART_NUMPY_DTYPE))
            pred_add_perturbed = pred_perturbed[:, i_add]
            pred_sub_perturbed = pred_perturbed[:, i_sub]
            pred_add_original = predictions[:, i_add]
            pred_sub_original = predictions[:, i_sub]

            # Compute delta loss with broadcasting
            delta_add = (pred_add_perturbed - pred_add_original) / self.perturbation_blackbox
            delta_sub = (pred_sub_perturbed - pred_sub_original) / self.perturbation_blackbox

            # Accumulate the gradient estimate
            gradient_estimate_add += np.sum(batch_vi * delta_add.reshape(batch_size)[:, None, None, None], axis=0)
            gradient_estimate_sub += np.sum(batch_vi * delta_sub.reshape(batch_size)[:, None, None, None], axis=0)

        # Average the accumulated gradient estimate over the total number of samples
        gradient_estimate_add /= self.samples_blackbox
        gradient_estimate_sub /= self.samples_blackbox

        return gradient_estimate_add - gradient_estimate_sub


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
        endist = self.l1 * l1dist + l2dist
        predictions = self.estimator.predict(np.array(x_adv, dtype=ART_NUMPY_DTYPE), batch_size=self.batch_size)

        return np.argmax(predictions, axis=1), l1dist, l2dist, endist
    
    def _sparsify_attack(self, o_best_attack, x_batch, quantile):
        # Compute the absolute differences
        differences = np.abs(o_best_attack - x_batch)
        
        # Initialize a mask of zeros with the same shape as differences
        mask = np.zeros_like(differences, dtype=bool)
        
        # Iterate through each image in the batch
        for i in range(differences.shape[0]):
            # Get the quantile threshold for the current image
            threshold = np.quantile(differences[i], quantile)
            
            # Create a mask for values greater than or equal to the threshold
            mask[i] = differences[i] >= threshold
        
        # Apply the mask to keep only the highest percentage modifications in o_best_attack
        o_best_attack_sparsified = np.where(mask, o_best_attack, x_batch)
        
        return o_best_attack_sparsified
