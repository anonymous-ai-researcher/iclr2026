# -*- coding: utf-8 -*-
"""
layers.py

Differentiable Fuzzy Logic Layers for DF-EL++.

This module implements the core fuzzy logic operators as differentiable
PyTorch functions, based on the choices described in Section 3 of the paper.

Key Components:
- Product T-Norm: For fuzzy conjunction (C ∩ D).
- Probabilistic Sum T-Conorm: For the existential quantifier (∃r.C),
  implemented with a numerically stable log-space evaluation as per
  Section 4.2.3.
- Göguen Implication: The R-implication corresponding to the Product T-Norm,
  used to define the satisfaction degree of axioms.
"""

import torch

def product_tnorm(a, b):
    """
    Computes the Product T-Norm for fuzzy conjunction.
    T(a, b) = a * b
    """
    return a * b

def probabilistic_sum_tconorm_stable(tensors, dim=-1, keepdim=False, epsilon=1e-12):
    """
    Computes the probabilistic sum t-conorm in a numerically stable way.
    This is used for the existential quantifier: 1 - Π(1 - x_i).
    The computation is done in log-space to prevent underflow, as described
    in Section 4.2.3.

    Args:
        tensors (torch.Tensor): Tensor of fuzzy membership values.
        dim (int): The dimension to reduce.
        keepdim (bool): Whether the output tensor has `dim` retained or not.
        epsilon (float): Small constant for numerical stability.

    Returns:
        torch.Tensor: The result of the t-conorm.
    """
    # Clamp values to avoid log(0)
    clamped_tensors = torch.clamp(tensors, 0.0, 1.0 - epsilon)
    
    # 1 - Π(1 - x_i) = 1 - exp(Σ log(1 - x_i))
    log_terms = torch.log(1.0 - clamped_tensors)
    sum_log_terms = torch.sum(log_terms, dim=dim, keepdim=keepdim)
    
    return 1.0 - torch.exp(sum_log_terms)

def goguen_implication(lhs, rhs):
    """
    Computes the Göguen Implication (R-implication for Product T-Norm).
    J_G(x, y) = 1 if x <= y, else y / x.

    Args:
        lhs (torch.Tensor): The membership degrees of the axiom's premise.
        rhs (torch.Tensor): The membership degrees of the axiom's conclusion.

    Returns:
        torch.Tensor: The satisfaction degree of the implication.
    """
    # Create a mask for the condition lhs <= rhs
    condition = (lhs <= rhs).float()
    
    # Calculate y / x for the case where lhs > rhs
    # Add a small epsilon to lhs to avoid division by zero
    division_val = rhs / (lhs + 1e-12)
    
    # Use the mask to select between 1.0 and y/x
    satisfaction = condition * 1.0 + (1.0 - condition) * division_val
    
    return satisfaction

