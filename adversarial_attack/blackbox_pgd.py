# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2020
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
This module implements the Projected Gradient Descent attack `ProjectedGradientDescent` as an iterative method in which,
after each iteration, the perturbation is projected on a lp-ball of specified radius (in addition to clipping the
values of the adversarial sample so that it lies in the permitted data range). This is the attack proposed by Madry et
al. for adversarial training.

| Paper link: https://arxiv.org/abs/1706.06083
"""
from __future__ import absolute_import, division, print_function, unicode_literals, annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import truncnorm
from tqdm.auto import trange

from art.attacks.evasion.fast_gradient import FastGradientMethod
from art.config import ART_NUMPY_DTYPE
from art.estimators.classification.classifier import ClassifierMixin
from art.estimators.estimator import BaseEstimator, LossGradientsMixin
from art.utils import compute_success, get_labels_np_array, check_and_transform_label_format, compute_success_array
from art.summary_writer import SummaryWriter

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_LOSS_GRADIENTS_TYPE, OBJECT_DETECTOR_TYPE

logger = logging.getLogger(__name__)


class ProjectedGradientDescentCommon(FastGradientMethod):
    """
    Common class for different variations of implementation of the Projected Gradient Descent attack. The attack is an
    iterative method in which, after each iteration, the perturbation is projected on a lp-ball of specified radius (in
    addition to clipping the values of the adversarial sample so that it lies in the permitted data range). This is the
    attack proposed by Madry et al. for adversarial training.

    | Paper link: https://arxiv.org/abs/1706.06083
    """

    attack_params = FastGradientMethod.attack_params + ["decay", "max_iter", "random_eps", "verbose"]
    _estimator_requirements = (BaseEstimator, LossGradientsMixin)

    def __init__(
        self,
        estimator: "CLASSIFIER_LOSS_GRADIENTS_TYPE" | "OBJECT_DETECTOR_TYPE",
        norm: int | float | str = np.inf,
        eps: int | float | np.ndarray = 0.3,
        eps_step: int | float | np.ndarray = 0.1,
        decay: float | None = None,
        max_iter: int = 100,
        targeted: bool = False,
        num_random_init: int = 0,
        batch_size: int = 32,
        random_eps: bool = False,
        summary_writer: str | bool | SummaryWriter = False,
        verbose: bool = True,
    ) -> None:
        """
        Create a :class:`.ProjectedGradientDescentCommon` instance.

        :param estimator: A trained classifier.
        :param norm: The norm of the adversarial perturbation, supporting  "inf", `np.inf` or a real `p >= 1`.
                     Currently, when `p` is not infinity, the projection step only rescales the noise, which may be
                     suboptimal for `p != 2`.
        :param eps: Maximum perturbation that the attacker can introduce.
        :param eps_step: Attack step size (input variation) at each iteration.
        :param random_eps: When True, epsilon is drawn randomly from truncated normal distribution. The literature
            suggests this for FGSM based training to generalize across different epsilons. eps_step is
            modified to preserve the ratio of eps / eps_step. The effectiveness of this method with PGD
            is untested (https://arxiv.org/pdf/1611.01236.pdf).
        :param decay: Decay factor for accumulating the velocity vector when using momentum.
        :param max_iter: The maximum number of iterations.
        :param targeted: Indicates whether the attack is targeted (True) or untargeted (False).
        :param num_random_init: Number of random initialisations within the epsilon ball. For num_random_init=0
            starting at the original input.
        :param batch_size: Size of the batch on which adversarial samples are generated.
        :param summary_writer: Activate summary writer for TensorBoard.
                               Default is `False` and deactivated summary writer.
                               If `True` save runs/CURRENT_DATETIME_HOSTNAME in current directory.
                               If of type `str` save in path.
                               If of type `SummaryWriter` apply provided custom summary writer.
                               Use hierarchical folder structure to compare between runs easily. e.g. pass in
                               ‘runs/exp1’, ‘runs/exp2’, etc. for each new experiment to compare across them.
        :param verbose: Show progress bars.
        """
        super().__init__(
            estimator=estimator,  # type: ignore
            norm=norm,
            eps=eps,
            eps_step=eps_step,
            targeted=targeted,
            num_random_init=num_random_init,
            batch_size=batch_size,
            minimal=False,
            summary_writer=summary_writer,
        )
        self.decay = decay
        self.max_iter = max_iter
        self.random_eps = random_eps
        self.verbose = verbose
        ProjectedGradientDescentCommon._check_params(self)

        lower: int | float | np.ndarray
        upper: int | float | np.ndarray
        var_mu: int | float | np.ndarray
        sigma: int | float | np.ndarray

        if self.random_eps:
            if isinstance(eps, (int, float)):
                lower, upper = 0, eps
                var_mu, sigma = 0, (eps / 2)
            else:
                lower, upper = np.zeros_like(eps), eps
                var_mu, sigma = np.zeros_like(eps), (eps / 2)

            self.norm_dist = truncnorm((lower - var_mu) / sigma, (upper - var_mu) / sigma, loc=var_mu, scale=sigma)

    def _random_eps(self):
        """
        Check whether random eps is enabled, then scale eps and eps_step appropriately.
        """
        if self.random_eps:
            ratio = self.eps_step / self.eps

            if isinstance(self.eps, (int, float)):
                self.eps = np.round(self.norm_dist.rvs(1)[0], 10)
            else:
                self.eps = np.round(self.norm_dist.rvs(size=self.eps.shape), 10)

            self.eps_step = ratio * self.eps

    def _set_targets(self, x: np.ndarray, y: np.ndarray | None, classifier_mixin: bool = True) -> np.ndarray:
        """
        Check and set up targets.

        :param x: An array with the original inputs.
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices of shape
                  (nb_samples,). Only provide this parameter if you'd like to use true labels when crafting adversarial
                  samples. Otherwise, model predictions are used as labels to avoid the "label leaking" effect
                  (explained in this paper: https://arxiv.org/abs/1611.01236). Default is `None`.
        :param classifier_mixin: Whether the estimator is of type `ClassifierMixin`.
        :return: The targets.
        """
        if classifier_mixin:
            if y is not None:
                y = check_and_transform_label_format(y, nb_classes=self.estimator.nb_classes)

        if y is None:
            # Throw error if attack is targeted, but no targets are provided
            if self.targeted:  # pragma: no cover
                raise ValueError("Target labels `y` need to be provided for a targeted attack.")

            # Use model predictions as correct outputs
            if classifier_mixin:
                targets = get_labels_np_array(self.estimator.predict(x, batch_size=self.batch_size))
            else:
                targets = self.estimator.predict(x, batch_size=self.batch_size)

        else:
            targets = y

        return targets

    def _check_params(self) -> None:  # pragma: no cover

        norm: float = np.inf if self.norm == "inf" else float(self.norm)
        if norm < 1:
            raise ValueError('Norm order must be either "inf", `np.inf` or a real `p >= 1`.')

        if not (
            isinstance(self.eps, (int, float))
            and isinstance(self.eps_step, (int, float))
            or isinstance(self.eps, np.ndarray)
            and isinstance(self.eps_step, np.ndarray)
        ):
            raise TypeError(
                "The perturbation size `eps` and the perturbation step-size `eps_step` must have the same type of `int`"
                ", `float`, or `np.ndarray`."
            )

        if isinstance(self.eps, (int, float)):
            if self.eps < 0:
                raise ValueError("The perturbation size `eps` has to be non-negative.")
        else:
            if (self.eps < 0).any():
                raise ValueError("The perturbation size `eps` has to be non-negative.")

        if isinstance(self.eps_step, (int, float)):
            if self.eps_step <= 0:
                raise ValueError("The perturbation step-size `eps_step` has to be positive.")
        else:
            if (self.eps_step <= 0).any():
                raise ValueError("The perturbation step-size `eps_step` has to be positive.")

        if isinstance(self.eps, np.ndarray) and isinstance(self.eps_step, np.ndarray):
            if self.eps.shape != self.eps_step.shape:
                raise ValueError(
                    "The perturbation size `eps` and the perturbation step-size `eps_step` must have the same shape."
                )

        if not isinstance(self.targeted, bool):
            raise ValueError("The flag `targeted` has to be of type bool.")

        if not isinstance(self.num_random_init, int):
            raise TypeError("The number of random initialisations has to be of type integer.")

        if self.num_random_init < 0:
            raise ValueError("The number of random initialisations `random_init` has to be greater than or equal to 0.")

        if self.batch_size <= 0:
            raise ValueError("The batch size `batch_size` has to be positive.")

        if self.max_iter < 0:
            raise ValueError("The number of iterations `max_iter` has to be a non-negative integer.")

        if self.decay is not None and self.decay < 0.0:
            raise ValueError("The decay factor `decay` has to be a nonnegative float.")

        if not isinstance(self.verbose, bool):
            raise ValueError("The verbose has to be a Boolean.")


class BlackBoxProjectedGradientDescentNumpy(ProjectedGradientDescentCommon):
    """
    The Projected Gradient Descent attack is an iterative method in which, after each iteration, the perturbation is
    projected on a lp-ball of specified radius (in addition to clipping the values of the adversarial sample so that it
    lies in the permitted data range). This is the attack proposed by Madry et al. for adversarial training.

    | Paper link: https://arxiv.org/abs/1706.06083
    """

    def __init__(
        self,
        estimator: "CLASSIFIER_LOSS_GRADIENTS_TYPE" | "OBJECT_DETECTOR_TYPE",
        norm: int | float | str = np.inf,
        eps: int | float | np.ndarray = 0.3,
        eps_step: int | float | np.ndarray = 0.1,
        decay: float | None = None,
        max_iter: int = 100,
        targeted: bool = False,
        num_random_init: int = 0,
        batch_size: int = 32,
        random_eps: bool = False,
        summary_writer: str | bool | SummaryWriter = False,
        verbose: bool = True,
        estimator_blackbox:str='gaussian_nes',
        perturbation_blackbox:float=0.001,
        samples_blackbox:int=50,
        max_batchsize_blackbox:int=100
    ) -> None:
        """
        Create a :class:`.ProjectedGradientDescentNumpy` instance.

        :param estimator: A trained estimator.
        :param norm: The norm of the adversarial perturbation, supporting  "inf", `np.inf` or a real `p >= 1`.
                     Currently, when `p` is not infinity, the projection step only rescales the noise, which may be
                     suboptimal for `p != 2`.
        :param eps: Maximum perturbation that the attacker can introduce.
        :param eps_step: Attack step size (input variation) at each iteration.
        :param random_eps: When True, epsilon is drawn randomly from truncated normal distribution. The literature
                           suggests this for FGSM based training to generalize across different epsilons. eps_step
                           is modified to preserve the ratio of eps / eps_step. The effectiveness of this method with
                           PGD is untested (https://arxiv.org/pdf/1611.01236.pdf).
        :param max_iter: The maximum number of iterations.
        :param targeted: Indicates whether the attack is targeted (True) or untargeted (False)
        :param num_random_init: Number of random initialisations within the epsilon ball. For num_random_init=0 starting
                                at the original input.
        :param batch_size: Size of the batch on which adversarial samples are generated.
        :param summary_writer: Activate summary writer for TensorBoard.
                               Default is `False` and deactivated summary writer.
                               If `True` save runs/CURRENT_DATETIME_HOSTNAME in current directory.
                               If of type `str` save in path.
                               If of type `SummaryWriter` apply provided custom summary writer.
                               Use hierarchical folder structure to compare between runs easily. e.g. pass in
                               ‘runs/exp1’, ‘runs/exp2’, etc. for each new experiment to compare across them.
        :param verbose: Show progress bars.
        """
        if summary_writer and num_random_init > 1:
            raise ValueError("TensorBoard is not yet supported for more than 1 random restart (num_random_init>1).")

        super().__init__(
            estimator=estimator,
            norm=norm,
            eps=eps,
            eps_step=eps_step,
            decay=decay,
            max_iter=max_iter,
            targeted=targeted,
            num_random_init=num_random_init,
            batch_size=batch_size,
            random_eps=random_eps,
            summary_writer=summary_writer,
            verbose=verbose,
        )

        self._project = True
        self.estimator_blackbox = estimator_blackbox
        self.perturbation_blackbox = perturbation_blackbox
        self.samples_blackbox = samples_blackbox
        self.max_batchsize_blackbox = max_batchsize_blackbox
    
    def _compute_perturbation(
        self,
        x: np.ndarray,
        y: np.ndarray,
        mask: np.ndarray | None,
        decay: float | None = None,
        momentum: np.ndarray | None = None,
    ) -> np.ndarray:
        # Get gradient wrt loss; invert it if attack is targeted
        #grad = self.estimator.loss_gradient(x, y) * (1 - 2 * int(self.targeted))

        # Use black-box gradient estimation with gaussian antithetic samples, no clipping (see https://arxiv.org/pdf/1804.08598)
        grad = self._estimate_gradient_blackbox(x, y, estimator='gaussian_nes') * (1 - 2 * int(self.targeted))

        # Write summary
        if self.summary_writer is not None:  # pragma: no cover
            self.summary_writer.update(
                batch_id=self._batch_id,
                global_step=self._i_max_iter,
                grad=grad,
                patch=None,
                estimator=self.estimator,
                x=x,
                y=y,
                targeted=self.targeted,
            )

        # Check for NaN before normalisation and replace with 0
        if grad.dtype != object and np.isnan(grad).any():  # pragma: no cover
            logger.warning("Elements of the loss gradient are NaN and have been replaced with 0.0.")
            grad = np.where(np.isnan(grad), 0.0, grad)
        else:
            for i, _ in enumerate(grad):
                grad_i_array = grad[i].astype(np.float32)
                if np.isnan(grad_i_array).any():
                    grad[i] = np.where(np.isnan(grad_i_array), 0.0, grad_i_array).astype(object)

        # Apply mask
        if mask is not None:
            grad = np.where(mask == 0.0, 0.0, grad)

        # Apply norm bound
        def _apply_norm(norm, grad, object_type=False):
            """Returns an x maximizing <grad, x> subject to ||x||_norm<=1."""
            if (grad.dtype != object and np.isinf(grad).any()) or np.isnan(  # pragma: no cover
                grad.astype(np.float32)
            ).any():
                logger.info("The loss gradient array contains at least one positive or negative infinity.")

            grad_2d = grad.reshape(1 if object_type else len(grad), -1)
            if norm in [np.inf, "inf"]:
                grad_2d = np.ones_like(grad_2d)
            elif norm == 1:
                i_max = np.argmax(np.abs(grad_2d), axis=1)
                grad_2d = np.zeros_like(grad_2d)
                grad_2d[range(len(grad_2d)), i_max] = 1
            elif norm > 1:
                conjugate = norm / (norm - 1)
                q_norm = np.linalg.norm(grad_2d, ord=conjugate, axis=1, keepdims=True)
                grad_2d = (np.abs(grad_2d) / np.where(q_norm, q_norm, np.inf)) ** (conjugate - 1)
            grad = grad_2d.reshape(grad.shape) * np.sign(grad)
            return grad

        # Compute gradient momentum
        if decay is not None and momentum is not None:
            if x.dtype == object:
                raise NotImplementedError("Momentum Iterative Method not yet implemented for object type input.")
            # Update momentum in-place (important).
            # The L1 normalization for accumulation is an arbitrary choice of the paper.
            grad_2d = grad.reshape(len(grad), -1)
            norm1 = np.linalg.norm(grad_2d, ord=1, axis=1, keepdims=True)
            normalized_grad = (grad_2d / np.where(norm1, norm1, np.inf)).reshape(grad.shape)
            momentum *= decay
            momentum += normalized_grad
            # Use the momentum to compute the perturbation, instead of the gradient
            grad = momentum

        if x.dtype == object:
            for i_sample in range(x.shape[0]):
                grad[i_sample] = _apply_norm(self.norm, grad[i_sample], object_type=True)
                assert x[i_sample].shape == grad[i_sample].shape
        else:
            grad = _apply_norm(self.norm, grad)

        assert x.shape == grad.shape

        return grad
    
    def generate(self, x: np.ndarray, y: np.ndarray | None = None, **kwargs) -> np.ndarray:
        """
        Generate adversarial samples and return them in an array.

        :param x: An array with the original inputs.
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices of shape
                  (nb_samples,). Only provide this parameter if you'd like to use true labels when crafting adversarial
                  samples. Otherwise, model predictions are used as labels to avoid the "label leaking" effect
                  (explained in this paper: https://arxiv.org/abs/1611.01236). Default is `None`.

        :param mask: An array with a mask broadcastable to input `x` defining where to apply adversarial perturbations.
                     Shape needs to be broadcastable to the shape of x and can also be of the same shape as `x`. Any
                     features for which the mask is zero will not be adversarially perturbed.
        :type mask: `np.ndarray`
        :return: An array holding the adversarial examples.
        """
        mask = self._get_mask(x, **kwargs)

        # Ensure eps is broadcastable
        self._check_compatibility_input_and_eps(x=x)

        # Check whether random eps is enabled
        self._random_eps()

        if isinstance(self.estimator, ClassifierMixin):
            # Set up targets
            targets = self._set_targets(x, y)

            # Start to compute adversarial examples
            adv_x = x.astype(ART_NUMPY_DTYPE)

            for batch_id in range(int(np.ceil(x.shape[0] / float(self.batch_size)))):

                self._batch_id = batch_id

                for rand_init_num in trange(
                    max(1, self.num_random_init), desc="PGD - Random Initializations", disable=not self.verbose
                ):
                    batch_index_1, batch_index_2 = batch_id * self.batch_size, (batch_id + 1) * self.batch_size
                    batch_index_2 = min(batch_index_2, x.shape[0])
                    batch = x[batch_index_1:batch_index_2]
                    batch_labels = targets[batch_index_1:batch_index_2]
                    mask_batch = mask

                    if mask is not None:
                        if len(mask.shape) == len(x.shape):
                            mask_batch = mask[batch_index_1:batch_index_2]

                    momentum = np.zeros(batch.shape)

                    for i_max_iter in trange(
                        self.max_iter, desc="PGD - Iterations", leave=False, disable=not self.verbose
                    ):
                        self._i_max_iter = i_max_iter

                        batch = self._compute(
                            batch,
                            x[batch_index_1:batch_index_2],
                            batch_labels,
                            mask_batch,
                            self.eps,
                            self.eps_step,
                            self._project,
                            self.num_random_init > 0 and i_max_iter == 0,
                            self._batch_id,
                            decay=self.decay,
                            momentum=momentum,
                        )

                    if rand_init_num == 0:
                        # initial (and possibly only) random restart: we only have this set of
                        # adversarial examples for now
                        adv_x[batch_index_1:batch_index_2] = np.copy(batch)
                    else:
                        # replace adversarial examples if they are successful
                        attack_success = compute_success_array(
                            self.estimator,  # type: ignore
                            x[batch_index_1:batch_index_2],
                            targets[batch_index_1:batch_index_2],
                            batch,
                            self.targeted,
                            batch_size=self.batch_size,
                        )
                        adv_x[batch_index_1:batch_index_2][attack_success] = batch[attack_success]

            logger.info(
                "Success rate of attack: %.2f%%",
                100
                * compute_success(
                    self.estimator,  # type: ignore
                    x,
                    targets,
                    adv_x,
                    self.targeted,
                    batch_size=self.batch_size,  # type: ignore
                ),
            )
        else:
            if self.num_random_init > 0:  # pragma: no cover
                raise ValueError("Random initialisation is only supported for classification.")

            # Set up targets
            targets = self._set_targets(x, y, classifier_mixin=False)

            # Start to compute adversarial examples
            if x.dtype == object:
                adv_x = x.copy()
            else:
                adv_x = x.astype(ART_NUMPY_DTYPE)

            momentum = np.zeros(adv_x.shape)

            for i_max_iter in trange(self.max_iter, desc="PGD - Iterations", disable=not self.verbose):
                self._i_max_iter = i_max_iter

                adv_x = self._compute(
                    adv_x,
                    x,
                    targets,
                    mask,
                    self.eps,
                    self.eps_step,
                    self._project,
                    self.num_random_init > 0 and i_max_iter == 0,
                    decay=self.decay,
                    momentum=momentum,
                )

        if self.summary_writer is not None:
            self.summary_writer.reset()

        return adv_x
    
    def _estimate_gradient_blackbox(self, x_adv, y_batch, estimator='gaussian_nes'):
            """
            Efficient batched gradient estimation using black-box sampling.
            """
            # Initialize the gradient estimate
            gradient_estimate = np.zeros_like(x_adv)

            # Determine the effective number of perturbation samples
            if estimator == 'gaussian_nes':
                half_samples = self.samples_blackbox // 2  # only half noise vectors
                # Extend x_adv to a batch with size equal to half_samples
                x_adv_extended = np.repeat(x_adv, half_samples, axis=0)
            else:
                x_adv_extended = np.repeat(x_adv, self.samples_blackbox, axis=0)

            try:
                if estimator == 'rademacher':
                    # Generate Rademacher samples (±1) for the extended batch
                    vi = np.random.choice([-1, 1], size=x_adv_extended.shape).astype(np.float32)
                elif estimator == 'uniform':
                    vi = np.random.uniform(size=x_adv_extended.shape).astype(np.float32)
                elif estimator == 'gaussian_nes':
                    vi = np.random.normal(size=x_adv_extended.shape).astype(np.float32)
                elif estimator == 'l1':
                    exp_samples = np.random.exponential(scale=1.0, size=x_adv_extended.shape).astype(np.float32)
                    norm = np.sum(exp_samples, axis=1, keepdims=True)
                    signs = np.random.choice([-1, 1], size=x_adv_extended.shape).astype(np.float32)
                    vi = signs * exp_samples / norm
                else:
                    raise ValueError("Invalid estimator_blackbox. Valid options are: 'rademacher', 'uniform', 'gaussian_nes', 'l1'.")
            except ValueError as e:
                print(e)

            if estimator == 'gaussian_nes': #antithetic samples, no clipping (see https://arxiv.org/pdf/1804.08598)
                rand_perturbed_inputs = x_adv_extended + self.perturbation_blackbox * vi
                anti_rand_perturbed_inputs = x_adv_extended - self.perturbation_blackbox * vi
                total_samples = half_samples
            else:
                # Compute perturbed inputs for all samples in the extended batch, clip inputs and perturbations
                rand_perturbed_inputs = np.clip(x_adv_extended + self.perturbation_blackbox * vi, 0.0, 1.0)
                vi = (rand_perturbed_inputs - x_adv_extended) / self.perturbation_blackbox
                total_samples = self.samples_blackbox

            # Split into batches of size self.max_batchsize_blackbox
            num_batches = int(np.ceil(total_samples / self.max_batchsize_blackbox))
            
            for batch_idx in range(num_batches):
                # Determine the range of indices for this batch
                start_idx = batch_idx * self.max_batchsize_blackbox
                end_idx = min((batch_idx + 1) * self.max_batchsize_blackbox, total_samples)
                batch_size = end_idx - start_idx
                
                # Slice the current batch
                batch_perturbed_inputs = rand_perturbed_inputs[start_idx:end_idx]
                batch_vi = vi[start_idx:end_idx]
                repeated_y = np.repeat(y_batch, batch_size, axis=0)

                # Compute losses
                loss_perturbed = self.estimator.compute_loss(batch_perturbed_inputs, repeated_y)
        
                if estimator == 'gaussian_nes':
                    #estimation between noise and reversed noise (antithetic samples https://arxiv.org/pdf/1804.08598)
                    anti_batch_perturbed_inputs = anti_rand_perturbed_inputs[start_idx:end_idx]
                    second_loss = self.estimator.compute_loss(anti_batch_perturbed_inputs, repeated_y)
                else:
                    #estimation between noise and original point
                    second_loss = self.estimator.compute_loss(x_adv, y_batch)
                
                # Compute delta loss and accumulate the gradient estimate with broadcasting
                delta_loss = (loss_perturbed - second_loss) / self.perturbation_blackbox
                gradient_estimate += np.sum(batch_vi * np.atleast_1d(delta_loss)[:, None, None, None], axis=0)

            # Average the accumulated gradient estimate over the total number of samples
            gradient_estimate /= self.samples_blackbox

            return gradient_estimate
