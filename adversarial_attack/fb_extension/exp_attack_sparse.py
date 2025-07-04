from typing import Union, Any, Optional, Callable, Tuple
from abc import ABC, abstractmethod
import eagerpy as ep
import math
import torch as pt
from foolbox.devutils import flatten
from foolbox.devutils import atleast_kd

from foolbox.types import Bounds

from foolbox.models.base import Model

from foolbox.criteria import Misclassification, TargetedMisclassification
from foolbox.distances import l1

from foolbox.attacks.base import FixedEpsilonAttack
from foolbox.attacks.base import T
from foolbox.attacks.base import get_criterion
from foolbox.attacks.base import raise_if_kwargs
from foolbox.attacks.base import verify_input_bounds
from foolbox.attacks.gradient_descent_base import GDOptimizer
from foolbox.attacks.gradient_descent_base import Optimizer
from foolbox.attacks.gradient_descent_base import normalize_lp_norms

def _best_other_classes(logits: ep.Tensor, exclude: ep.Tensor) -> ep.Tensor:
    other_logits = logits - ep.onehot_like(logits, exclude, value=ep.inf)
    return other_logits.argmax(axis=-1)
        #@abstractmethod
        #def get_random_start(self, x0: ep.Tensor, epsilon: float) -> ep.Tensor:
        #    ...

class SparseExpGradient(FixedEpsilonAttack, ABC):
    distance = l1
    def __init__(
        self,
        *,
        learning_rate: float = 1.0,
        steps: int,
        beta:float,
    ):
        self.learning_rate = learning_rate
        self.steps = steps
        self.beta=beta
    


    def get_loss_fn(
        self, model: Model, labels: ep.Tensor
    ) -> Callable[[ep.Tensor], ep.Tensor]:
        # can be overridden by users
        def loss_fn(inputs: ep.Tensor) -> ep.Tensor:
            logits = model(inputs)
            return ep.crossentropy(logits, labels).sum()

        return loss_fn
    
    def get_loss_fn_vec(
        self, model: Model, labels: ep.Tensor
    ) -> Callable[[ep.Tensor], ep.Tensor]:
        # can be overridden by users
        def loss_fn(inputs: ep.Tensor) -> ep.Tensor:
            logits = model(inputs)
            return ep.crossentropy(logits, labels)
        return loss_fn
    
    def get_optimizer(self, x: ep.Tensor, stepsize: float) -> Optimizer:
        # can be overridden by users
        return GDOptimizer(x, stepsize)

    def value_and_grad(
        # can be overridden by users
        self,
        loss_fn: Callable[[ep.Tensor], ep.Tensor],
        x: ep.Tensor,
    ) -> Tuple[ep.Tensor, ep.Tensor]:
        return ep.value_and_grad(loss_fn, x)

    def run(
        self,
        model: Model,
        inputs: T,
        criterion: Union[Misclassification, TargetedMisclassification, T],
        *,
        epsilon: float,
        **kwargs: Any,
    ) -> T:
        raise_if_kwargs(kwargs)
        x0, restore_type = ep.astensor_(inputs)
        criterion_ = get_criterion(criterion)
        del inputs, criterion, kwargs

        verify_input_bounds(x0, model)

        # perform a gradient ascent (targeted attack) or descent (untargeted attack)
        
        if isinstance(criterion_, Misclassification):
            gradient_step_sign = 1.0
            classes = criterion_.labels
        elif hasattr(criterion_, "target_classes"):
            gradient_step_sign = -1.0
            classes = criterion_.target_classes
        else:
            raise ValueError("unsupported criterion")

        loss_fn = self.get_loss_fn(model, classes)
        loss_fn_vec=self.get_loss_fn_vec(model,classes)

        

        #def is_adversarial(perturbed: ep.Tensor, logits: ep.Tensor) -> ep.Tensor:
        #    if change_classes_logits != 0:
        #        logits += ep.onehot_like(logits, classes, value=change_classes_logits)
        #    return criterion_(perturbed, logits)



        optimizer = self.get_optimizer(x0, self.learning_rate)

        #if self.random_start:
        #    x = self.get_random_start(x0, epsilon)
        #    x = ep.clip(x, *model.bounds)
        #else:
            #x = x0
        delta=ep.zeros(t=x0.T,shape=x0.shape)
        upper=1.0-x0
        lower=0.0-x0
        x=x0+delta
        x_best=x0+delta
        loss_best=loss_fn_vec(x_best)
        loss_best=loss_best[:, ep.newaxis, ep.newaxis, ep.newaxis]
        
        c=ep.where(delta>0,upper,lower)
        c=ep.where(delta==0,1.0,c)

        eta=ep.zeros(t=x0.T,shape=x0.shape[0])[:, ep.newaxis, ep.newaxis, ep.newaxis]
        for _iter in range(self.steps):
            #_, _, gradient = loss_aux_and_grad(x)
            _, gradient = self.value_and_grad(loss_fn, x)
            descent_max= ep.max(ep.abs(gradient),axis=[1,2,3])[:, ep.newaxis, ep.newaxis, ep.newaxis]
            #descent=descent/descent_max
            eta+= descent_max**2
            descent= -gradient_step_sign*gradient/ep.sqrt(eta)
            #x = x + gradient_step_sign * optimizer(gradients)
            delta=self.mirror_descent(descent,delta,lower,upper,self.beta,epsilon)
            
            delta_raw=delta.raw
            delta_raw_flat=delta_raw.view([delta.shape[0],-1])
            topk_vals, topk_idx = pt.topk(pt.abs(delta_raw_flat),int(epsilon), dim=1)
            flat_masked = pt.zeros_like(delta_raw_flat)
            flat_masked.scatter_(dim=1, index=topk_idx, value=1)
            deLta_masked = ep.astensor(flat_masked.view_as(delta_raw))
            

            #print(topk_val)
            #topk_val=-ep.sort(-ep.abs(delta).reshape([delta.shape[0],-1]),axis=1)[:,int(epsilon)-1][:, ep.newaxis, ep.newaxis, ep.newaxis]
            
            delta_topk=delta*deLta_masked
            #print(ep.sum(delta_topk>0,axis=[1,2,3]))
            attack=ep.where(delta_topk>0,upper,lower)
            attack=ep.where(delta_topk==0,0.0,attack)
            x=x0+attack
            
            loss_val=loss_fn_vec(x)
            loss_val=loss_val[:, ep.newaxis, ep.newaxis, ep.newaxis]
            
            x_best= ep.where(loss_val>loss_best,x,x_best)
            loss_best= ep.where(loss_val>loss_best,loss_val,loss_best)
            print(ep.sum(loss_best))

        return restore_type(x_best)
    
    def _get_descent(self,g:ep.tensor,x:ep.tensor, l:ep.tensor,u:ep.tensor,eps)-> ep.tensor:
        c=ep.where(g<0.0,u-x,l-x)
        prod=(c*g).reshape([g.shape[0],-1])
        sorted_idx=ep.argsort(prod,axis=1)
        reverse_idx=ep.argsort(sorted_idx,axis=1)
        sorted_c=ep.take_along_axis(ep.abs(c-x).reshape([g.shape[0],-1]),sorted_idx,axis=1)
        sorted_g=ep.take_along_axis(g.reshape([g.shape[0],-1]),sorted_idx,axis=1)
        sum_c=ep.cumsum(sorted_c,axis=1)
        sorted_g=ep.where(sum_c>eps,0.0,sorted_c)
        reverse_g=ep.take_along_axis(sorted_g,reverse_idx,axis=1).reshape(g.shape)
        return reverse_g


    def mirror_descent(self,descent: ep.Tensor,x: ep.Tensor,lower: ep.Tensor,upper: ep.Tensor,beta:float,epsilon:float)-> ep.Tensor:
        beta=epsilon/x.shape[1]/x.shape[2]/x.shape[3]
        dual_x=(ep.log(ep.abs(x) / beta + 1.0)) * ep.sign(x)
        z=dual_x -descent
        z_sgn=ep.sign(z)
        z_val=ep.abs(z)
        v =self.project(z_sgn,z_val,beta,epsilon,lower,upper) 
        #v = ep.stack([self._project(z_sgn[d],z_val[d],beta,epsilon,lower[d],upper[d]) for d in range(dual_x.shape[0])], axis=0)
        return v
    
    def _project(self, y_sgn: ep.tensor,dual_y_val: ep.tensor, beta:float, D:float,l: ep.tensor,u: ep.tensor)-> ep.tensor:
                #upper bound optimal value

        c=ep.where(y_sgn<=0,ep.abs(l),u)
        dual_c=ep.log(c/beta+1.0)
        #y lies outside hyper cube
        phi_0=beta*ep.exp(ep.maximum(ep.minimum(dual_y_val,dual_c),0.0))-beta
        if ep.sum(phi_0)<=D:
            return phi_0*y_sgn
        z=ep.sort(ep.stack((dual_y_val,dual_y_val-dual_c)).reshape(-1))
        z=z[z>=0]
        idx_l=0
        idx_u= math.prod(z.shape)-1
        while idx_u-idx_l>1:
            idx=(idx_u+idx_l)//2
            lam=z[idx]
            phi=ep.sum(beta*ep.exp(ep.maximum(ep.minimum(dual_y_val-lam,dual_c),0.0))-beta)
            if phi>D:
                idx_l=idx
            elif phi<D:
                idx_u=idx
            else:
                idx_u=idx
                idx_l=idx-1
        lam_lower=z[idx_u]
        lam_upper=z[idx_l]
        phi_upper=ep.sum(beta*ep.exp(ep.maximum(ep.minimum(dual_y_val-lam_upper,dual_c),0.0))-beta)
        if phi_upper==D:
            v=beta*ep.exp(ep.maximum(ep.minimum(dual_y_val-lam_upper,dual_c),0.0))-beta
        else:
            lam=(lam_lower+lam_upper)/2.0
            idx_clip=dual_y_val-lam>=dual_c
            idx_active=ep.logical_and((dual_y_val-lam)<dual_c , (dual_y_val-lam)>0)
            v=ep.where(idx_clip,c,0.0)
            num_active=ep.sum(idx_active)
            if num_active!=0:
                sum_active=D-ep.sum(c[idx_clip])
                max_dual_y=ep.max((dual_y_val)[idx_active])
                normaliser=(sum_active+beta*num_active)/ep.sum(beta*ep.exp(dual_y_val[idx_active]-max_dual_y))
                mask = ep.zeros_like(v).bool()
                mask = mask.index_update(idx_active, True)
                new_values = beta * ep.exp(dual_y_val - max_dual_y) * normaliser - beta
                v = ep.where(idx_active, new_values, v)
                #v[idx_active]=beta*ep.exp(dual_y_val[idx_active]-max_dual_y)*normaliser-beta
        return v*y_sgn

    def project(self, y_sgn: ep.tensor,dual_y_val: ep.tensor, beta:float, D:float,l: ep.tensor,u: ep.tensor)-> ep.tensor:
                #upper bound optimal value
        orig_shape=y_sgn.shape
        B=orig_shape[0]
        #N=orig_shape[1:]
        sample_dims=tuple(range(1, dual_y_val.ndim))

        c=ep.where(y_sgn<=0,ep.abs(l),u)
        dual_c=ep.log(c/beta+1.0)
        #y lies outside hyper cube
        
        phi_0=beta*ep.exp(ep.maximum(ep.minimum(dual_y_val,dual_c),0.0))-beta
        mask_lam_null=ep.sum(phi_0,axis=sample_dims,keepdims=True)<=D
        v=ep.zeros(dual_y_val,dual_y_val.shape)

        lam=ep.zeros(dual_y_val,[B,1])

        z = ep.sort(ep.concatenate([dual_y_val.reshape([B,-1]),lam, (dual_y_val - dual_c).reshape([B,-1])], axis=1), axis=1)
        idx_l = ep.argmax( (z >= 0)+0, axis=1)#[(...,) + (None,) * (mask_search.ndim - 1)]
        idx_u = ep.where(mask_lam_null[(...,) + (0,) * (mask_lam_null.ndim - 1)],idx_l,z.shape[1]-1)
        # maximal number of iterations of binary search
        max_iter=math.ceil(math.log2(z.shape[1]))
        for _ in range(max_iter):
            idx=(idx_u+idx_l)//2
            lam=z[range(z.shape[0]),idx][(...,) + (None,) * (dual_y_val.ndim - 1)]
            phi=ep.sum(beta*ep.exp(ep.maximum(ep.minimum(dual_y_val-lam,dual_c),0.0))-beta,axis=tuple(range(1, dual_y_val.ndim)))
            idx_l=ep.where(phi>D,idx,idx_l)
            idx_u=ep.where(phi<D,idx,idx_u)
            idx_u=ep.where(phi==D,idx,idx_u)
            idx_l=ep.where(phi==D,idx-1,idx_l)

        lam_lower=z[range(z.shape[0]),idx_u][(...,) + (None,) * (dual_y_val.ndim - 1)]
        lam_upper=z[range(z.shape[0]),idx_l][(...,) + (None,) * (dual_y_val.ndim - 1)]


        phi_upper=ep.sum(beta*ep.exp(ep.maximum(ep.minimum(dual_y_val-lam_upper,dual_c),0.0))-beta,keepdims=True,axis=tuple(range(1, dual_y_val.ndim)))
        
        #phi_lower=ep.sum(beta*ep.exp(ep.maximum(ep.minimum(dual_y_val-lam_lower,dual_c),0.0))-beta,keepdims=True,axis=tuple(range(1, dual_y_val.ndim)))
        #print(ep.sum(phi_upper,axis=sample_dims))
        #print(ep.sum(phi_lower,axis=sample_dims))
        mask_set_upper=(ep.logical_and(phi_upper==D,ep.logical_not(mask_lam_null)))
    
        lam=(lam_lower+lam_upper)/2.0
        #phi_lower=ep.sum(beta*ep.exp(ep.maximum(ep.minimum(dual_y_val-lam,dual_c),0.0))-beta,keepdims=True,axis=tuple(range(1, dual_y_val.ndim)))
        #print(ep.sum(phi_lower,axis=sample_dims))
        idx_clip=dual_y_val-lam>=dual_c
        idx_active=ep.logical_and(ep.logical_not(idx_clip), (dual_y_val-lam)>0)
        
        #print(ep.sum(idx_active,axis=sample_dims,keepdims=True))
        v=ep.where(idx_clip,c,0.0)
        #print(ep.sum(v,axis=sample_dims,keepdims=True))
        
        num_active=ep.sum(idx_active,axis=sample_dims,keepdims=True)

        #if num_active!=0:
        sum_active=D-ep.sum(v,axis=sample_dims,keepdims=True)
         
        max_dual_y=ep.max(ep.where(idx_active,dual_y_val,0.0),keepdims=True,axis=sample_dims)
        normaliser=(sum_active+beta*num_active)/ep.sum(beta*ep.exp(dual_y_val-max_dual_y)*idx_active,keepdims=True,axis=sample_dims)
        #normaliser=ep.log(ep.sum(ep.where(idx_active,beta*ep.exp(dual_y_val),0.0),keepdims=True,axis=sample_dims)/(sum_active+beta*num_active))
        
        #print(ep.logical_and(ep.exp(-normaliser- max_dual_y)>=lam_lower,ep.exp(-normaliser- max_dual_y)<=lam_upper))
        new_values = beta * ep.exp(dual_y_val - max_dual_y) * normaliser - beta
        
        v = ep.where(idx_active, new_values, v)
        v = ep.where(mask_set_upper,beta*ep.exp(ep.maximum(ep.minimum(dual_y_val-lam_upper,dual_c),0.0))-beta,v)
        v=ep.where(mask_lam_null,phi_0,v)
        
        return v*y_sgn
