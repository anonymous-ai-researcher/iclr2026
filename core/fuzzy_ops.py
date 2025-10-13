# fuzzy_ops.py

import torch

def fuzzy_conjunction(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    [cite_start]Product t-norm for fuzzy conjunction (AND)[cite: 377].
    """
    return a * b

def fuzzy_disjunction(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    [cite_start]Probabilistic sum (dual to Product t-norm) for fuzzy disjunction (OR)[cite: 378].
    """
    return a + b - a * b

def fuzzy_negation(a: torch.Tensor) -> torch.Tensor:
    """
    [cite_start]Standard fuzzy negation[cite: 376].
    """
    return 1.0 - a

def fuzzy_implication_reichenbach(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Reichenbach S-implication for fuzzy implication (→), chosen for its
    [cite_start]superior gradient properties[cite: 336, 379].
    """
    return 1.0 - a + a * b

def differentiable_sup(tensor: torch.Tensor, dim: int, tau: float = 0.1) -> torch.Tensor:
    """
    [cite_start]Differentiable supremum (sup) using LogSumExp for the existential quantifier[cite: 328].
    This serves as a smooth approximation of the max() function.
    """
    return tau * torch.logsumexp(tensor / tau, dim=dim)

def differentiable_inf(tensor: torch.Tensor, dim: int, tau: float = 0.1) -> torch.Tensor:
    """
    [cite_start]Differentiable infimum (inf) using LogSumExp for the universal quantifier[cite: 328].
    This serves as a smooth approximation of the min() function.
    Note: inf(z) = -sup(-z)
    """
    return -differentiable_sup(-tensor, dim=dim, tau=tau)

def fuzzy_existential_quantifier(body_satisfaction: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    """
    [cite_start]Computes the satisfaction of ∃x(...) using differentiable sup[cite: 380].
    """
    return differentiable_sup(body_satisfaction, dim=-1, tau=tau)

def fuzzy_universal_quantifier(body_satisfaction: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    """
    [cite_start]Computes the satisfaction of ∀x(...) using differentiable inf[cite: 379].
    """
    return differentiable_inf(body_satisfaction, dim=-1, tau=tau)
