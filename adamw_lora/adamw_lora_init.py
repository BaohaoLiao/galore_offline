# copy dependencies from transformers/optimization.py
import math
import warnings
from typing import Callable, Iterable, Tuple

import torch
from torch import nn
from torch.optim import Optimizer

from transformers.utils.versions import require_version


class AdamW(Optimizer):
    """
    Implements Adam algorithm with weight decay fix as introduced in [Decoupled Weight Decay
    Regularization](https://arxiv.org/abs/1711.05101).

    Parameters:
        params (`Iterable[nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (`float`, *optional*, defaults to 0.001):
            The learning rate to use.
        betas (`Tuple[float,float]`, *optional*, defaults to `(0.9, 0.999)`):
            Adam's betas parameters (b1, b2).
        eps (`float`, *optional*, defaults to 1e-06):
            Adam's epsilon for numerical stability.
        weight_decay (`float`, *optional*, defaults to 0.0):
            Decoupled weight decay to apply.
        correct_bias (`bool`, *optional*, defaults to `True`):
            Whether or not to correct bias in Adam (for instance, in Bert TF repository they use `False`).
        no_deprecation_warning (`bool`, *optional*, defaults to `False`):
            A flag used to disable the deprecation warning (set to `True` to disable the warning).
    """

    def __init__(
        self,
        named_params: Iterable[Tuple[str, nn.Parameter]],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
        no_deprecation_warning: bool = False,
        lora_init_gap: int = 200,
    ):
        if not no_deprecation_warning:
            warnings.warn(
                "This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch"
                " implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this"
                " warning",
                FutureWarning,
            )
        require_version("torch>=1.5.0")  # add_ with alpha
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay, "correct_bias": correct_bias}

        params = []
        for param_name, param in named_params:
            if not param.requires_grad:
                continue
            state = {}
            state["name"] = param_name
            state["params"] = param
            """
            if "norm" in param_name or "ln_f" in param_name:
                state["weight_decay"] = 0.0
            else:
                state["weight_decay"] = weight_decay
            """
            params.append(state)
        super().__init__(params, defaults)
        self.lora_init_gap = lora_init_gap
        self.global_step = 0

    @torch.no_grad()
    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        # init LoRA
        rank = 128
        if self.global_step == 0:
            print("Reinitialize A and B")
            lora_ABs = {}
            for group in self.param_groups:
                name = group["name"]
                if "lora_" in name:
                    assert len(group["params"]) == 1
                    lora_ABs[name] = group["params"][0]

            for group in self.param_groups:
                name = group["name"]
                if "base_layer" not in name:
                    continue

                assert len(group["params"]) == 1
                lora_A_name = ".".join(name.split(".")[:-2]) + ".lora_A.default.weight"
                lora_B_name = ".".join(name.split(".")[:-2]) + ".lora_B.default.weight"

                p = group["params"][0]
                grad = p.grad
                U, _, V = torch.svd_lowrank(grad.float(), q=4 * rank, niter=4)
                V = V.T
                B = U[:, rank:2*rank]
                A = V[:rank, :]

                m, n = grad.shape
                gamma = 16
                B = (B * m**0.25 / gamma**0.5).to(torch.bfloat16)
                A = (A * m**0.25 / gamma**0.5).to(torch.bfloat16)

                lora_ABs[lora_A_name].data = A
                lora_ABs[lora_B_name].data = B

                p.data = p.data - B @ A
            
            """
            elif self.global_step % self.lora_init_gap == self.lora_init_gap - 1:
                print("Merge A and B to W")
                lora_ABs = {}
                for group in self.param_groups:
                    name = group["name"]
                    if "lora_" in name:
                        assert len(group["params"]) == 1
                        lora_ABs[name] = group["params"][0]

                for group in self.param_groups:
                    name = group["name"]
                    if "base_layer" not in name:
                        continue

                    assert len(group["params"]) == 1
                    lora_A_name = ".".join(name.split(".")[:-2]) + ".lora_A.default.weight"
                    lora_B_name = ".".join(name.split(".")[:-2]) + ".lora_B.default.weight"

                    p = group["params"][0]
                    p.data = p.data + lora_ABs[lora_B_name].data @ lora_ABs[lora_A_name].data
                    
                    lora_ABs[lora_A_name].data = torch.zeros_like(lora_ABs[lora_A_name].data)
                    lora_ABs[lora_B_name].data = torch.zeros_like(lora_ABs[lora_B_name].data)

                for group in self.param_groups:
                    if "lora_B" in group["name"]:
                        assert group["params"][0].sum() == 0

                    if "lora_" in group["name"]:
                        for p in group["params"]:   
                            state = self.state[p]
                            state["exp_avg"] = 0.5 * state["exp_avg"]
                            state["exp_avg_sq"] = 0.5 * state["exp_avg_sq"]
            """               
        else:
            lora_ABs_norm_grad = {}
            lora_ABs_data = {}
            for group in self.param_groups:
                name = group["name"]
                if "lora_" not in name:
                    continue

                for p in group["params"]:
                    if p.grad is None:
                        continue
                    grad = p.grad
                    if grad.is_sparse:
                        raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                    state = self.state[p]

                    # State initialization
                    if "step" not in state:
                        state["step"] = 0

                    if "exp_avg" not in state:
                        # Exponential moving average of gradient values
                        state["exp_avg"] = torch.zeros_like(p)
                        # Exponential moving average of squared gradient values
                        state["exp_avg_sq"] = torch.zeros_like(p)

                    exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                    beta1, beta2 = group["betas"]

                    state["step"] += 1

                    # Decay the first and second moment running average coefficient
                    # In-place operations to update the averages at the same time
                    exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                    denom = exp_avg_sq.sqrt().add_(group["eps"])

                    step_size = group["lr"]
                    if group["correct_bias"]:  # No bias correction for Bert
                        bias_correction1 = 1.0 - beta1 ** state["step"]
                        bias_correction2 = 1.0 - beta2 ** state["step"]
                        step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                    norm_grad = exp_avg / denom
                    lora_ABs_norm_grad[name] = norm_grad
                    assert len(group["params"]) == 1
                    lora_ABs_data[name] = p

            """
            lora_ABs_exp_avg = {}
            lora_ABs_exp_avg_sq = {}
            lora_ABs_data = {}
            for group in self.param_groups:
                name = group["name"]
                if "lora_" not in name:
                    continue

                for p in group["params"]:
                    if p.grad is None:
                        continue
                    grad = p.grad
                    if grad.is_sparse:
                        raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                    state = self.state[p]

                    # State initialization
                    if "step" not in state:
                        state["step"] = 0

                    if "exp_avg" not in state:
                        # Exponential moving average of gradient values
                        state["exp_avg"] = torch.zeros_like(p)
                        # Exponential moving average of squared gradient values
                        state["exp_avg_sq"] = torch.zeros_like(p)

                    exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                    beta1, beta2 = group["betas"]

                    import copy
                    lora_ABs_exp_avg[name] = copy.deepcopy(exp_avg)
                    lora_ABs_exp_avg_sq[name] = copy.deepcopy(exp_avg_sq)

                    state["step"] += 1

                    # Decay the first and second moment running average coefficient
                    # In-place operations to update the averages at the same time
                    exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                    #denom = exp_avg_sq.sqrt().add_(group["eps"])

                    step_size = group["lr"]
                    if group["correct_bias"]:  # No bias correction for Bert
                        bias_correction1 = 1.0 - beta1 ** state["step"]
                        bias_correction2 = 1.0 - beta2 ** state["step"]
                        step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                    lora_ABs_data[name] = p
            """

            for group in self.param_groups:
                name = group["name"] 
                if "lora_" in name:
                    continue

                if "base_layer" in name:
                    for p in group["params"]:
                        assert len(group["params"]) == 1
                        grad = p.grad
                        assert grad is not None
                        lora_A_name = ".".join(name.split(".")[:-2]) + ".lora_A.default.weight"
                        lora_B_name = ".".join(name.split(".")[:-2]) + ".lora_B.default.weight"

                        step_size = group["lr"]
                        if group["correct_bias"]:  # No bias correction for Bert
                            bias_correction1 = 1.0 - beta1 ** state["step"]
                            bias_correction2 = 1.0 - beta2 ** state["step"]
                            step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                        lora_A_w, lora_A_norm_grad = lora_ABs_norm_grad[lora_A_name].data, lora_ABs_norm_grad[lora_A_name]
                        lora_B_w, lora_B_norm_grad = lora_ABs_norm_grad[lora_B_name].data, lora_ABs_norm_grad[lora_B_name]
                        norm_grad = lora_B_w @ lora_A_norm_grad + lora_B_norm_grad @ lora_A_w
                        p.add_(norm_grad, alpha=-step_size)

                        """
                        lora_A_w, lora_A_exp_avg, lora_A_exp_avg_sq = lora_ABs_data[lora_A_name].data, lora_ABs_exp_avg[lora_A_name], lora_ABs_exp_avg_sq[lora_A_name]
                        lora_B_w, lora_B_exp_avg, lora_B_exp_avg_sq = lora_ABs_data[lora_B_name].data, lora_ABs_exp_avg[lora_B_name], lora_ABs_exp_avg_sq[lora_B_name]

                        exp_avg = torch.mul(lora_B_w, lora_B_exp_avg_sq.sqrt()) @ lora_A_exp_avg + lora_B_exp_avg @ torch.mul(lora_A_w, lora_A_exp_avg_sq.sqrt())
                        exp_avg_sq = lora_B_exp_avg_sq @ lora_A_exp_avg_sq

                        beta1, beta2 = group["betas"]
                        exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                        denom = exp_avg_sq.sqrt().add_(group["eps"])

                        norm_grad = exp_avg / denom
                        p.add_(norm_grad, alpha=-step_size)
                        """
                        if group["weight_decay"] > 0.0:
                            p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))
                else:
                    for p in group["params"]:
                        if p.grad is None:
                                continue
                            
                        grad = p.grad
                        if grad.is_sparse:
                            raise RuntimeError(
                                "Adam does not support sparse gradients, please consider SparseAdam instead"
                            )
                                
                        state = self.state[p]
                        
                        if "step" not in state:
                            state["step"] = 0
                            
                        if "exp_avg" not in state:
                            # Exponential moving average of gradient values
                            state["exp_avg"] = torch.zeros_like(p)
                            # Exponential moving average of squared gradient values
                            state["exp_avg_sq"] = torch.zeros_like(p)

                        exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                        beta1, beta2 = group["betas"]

                        state["step"] += 1

                        # Decay the first and second moment running average coefficient
                        # In-place operations to update the averages at the same time
                        exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                        denom = exp_avg_sq.sqrt().add_(group["eps"])

                        step_size = group["lr"]
                        if group["correct_bias"]:  # No bias correction for Bert
                            bias_correction1 = 1.0 - beta1 ** state["step"]
                            bias_correction2 = 1.0 - beta2 ** state["step"]
                            step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                        # compute norm gradient
                        norm_grad = exp_avg / denom
                        
                        # scaling constant
                        # self.scaling = self.lora_alpha / self.r
                        
                        # lora scaling ?
                        p.add_(norm_grad, alpha=-step_size)

                        # Just adding the square of the weights to the loss function is *not*
                        # the correct way of using L2 regularization/weight decay with Adam,
                        # since that will interact with the m and v parameters in strange ways.
                        #
                        # Instead we want to decay the weights in a manner that doesn't interact
                        # with the m/v parameters. This is equivalent to adding the square
                        # of the weights to the loss with plain (non-momentum) SGD.
                        # Add weight decay at the end (fixed version)
                        if group["weight_decay"] > 0.0:
                            p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))


        self.global_step += 1

        """
        Performs a single optimization step.

        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.

            
        loss = None
        if closure is not None:
            loss = closure()

        lora_ABs_norm_grad = {}
        lora_ABs_data = {}
        for group in self.param_groups:
            name = group["name"]
            if "lora_" not in name:
                continue

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                norm_grad = exp_avg / denom
                lora_ABs_norm_grad[name] = norm_grad
                assert len(group["params"]) == 1
                lora_ABs_data[name] = p


        for group in self.param_groups:
            name = group["name"] 
            if "lora_" in name:
                continue

            if "base_layer" in name:
                for p in group["params"]:
                    assert len(group["params"]) == 1
                    lora_A_name = ".".join(name.split(".")[:-2]) + ".lora_A.default.weight"
                    lora_B_name = ".".join(name.split(".")[:-2]) + ".lora_B.default.weight"

                    lora_A_w, lora_A_norm_grad = lora_ABs_data[lora_A_name].data, lora_ABs_norm_grad[lora_A_name]
                    lora_B_w, lora_B_norm_grad = lora_ABs_data[lora_B_name].data, lora_ABs_norm_grad[lora_B_name]
                    norm_grad = lora_B_w @ lora_A_norm_grad + lora_B_norm_grad @ lora_A_w
                    p.add_(norm_grad, alpha=-step_size)

                    if group["weight_decay"] > 0.0:
                        p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))
            else:
                for p in group["params"]:
                    if p.grad is None:
                            continue
                        
                    grad = p.grad
                    if grad.is_sparse:
                        raise RuntimeError(
                            "Adam does not support sparse gradients, please consider SparseAdam instead"
                        )
                            
                    state = self.state[p]
                    
                    if "step" not in state:
                        state["step"] = 0
                        
                    if "exp_avg" not in state:
                        # Exponential moving average of gradient values
                        state["exp_avg"] = torch.zeros_like(p)
                        # Exponential moving average of squared gradient values
                        state["exp_avg_sq"] = torch.zeros_like(p)

                    exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                    beta1, beta2 = group["betas"]

                    state["step"] += 1

                    # Decay the first and second moment running average coefficient
                    # In-place operations to update the averages at the same time
                    exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                    denom = exp_avg_sq.sqrt().add_(group["eps"])

                    step_size = group["lr"]
                    if group["correct_bias"]:  # No bias correction for Bert
                        bias_correction1 = 1.0 - beta1 ** state["step"]
                        bias_correction2 = 1.0 - beta2 ** state["step"]
                        step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                    # compute norm gradient
                    norm_grad = exp_avg / denom
                    
                    # scaling constant
                    # self.scaling = self.lora_alpha / self.r
                    
                    # lora scaling ?
                    p.add_(norm_grad, alpha=-step_size)

                    # Just adding the square of the weights to the loss function is *not*
                    # the correct way of using L2 regularization/weight decay with Adam,
                    # since that will interact with the m and v parameters in strange ways.
                    #
                    # Instead we want to decay the weights in a manner that doesn't interact
                    # with the m/v parameters. This is equivalent to adding the square
                    # of the weights to the loss with plain (non-momentum) SGD.
                    # Add weight decay at the end (fixed version)
                    if group["weight_decay"] > 0.0:
                        p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))

        """


        return loss