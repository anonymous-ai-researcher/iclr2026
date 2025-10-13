# guardnet/layers.py

import torch
import torch.nn as nn

class FuzzyLogicLayers:
    """
    Implements the differentiable fuzzy logic operators defined in the paper
    - Conjunction (AND): Product t-norm
    - Disjunction (OR): Probabilistic Sum (dual to Product)
    - Negation (NOT): Standard Negation
    - Implication (IMPLIES): Reichenbach S-implication
    - Quantifiers (FORALL, EXISTS): Differentiable approximation using LogSumExp (LSE)
    """
    def __init__(self, temperature: float = 0.1):
        """
        Initializes the fuzzy logic layers
        Args:
            temperature (float): The temperature parameter τ (tau) for LSE approximation
        """
        self.tau = temperature

    def negation(self, x: torch.Tensor) -> torch.Tensor:
        """ [[¬φ]] = 1 - [[φ]] """
        return 1 - x

    def conjunction(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """ [[φ ∧ ψ]] = [[φ]] * [[ψ]] (Product t-norm) """
        return x * y

    def disjunction(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """ [[φ ∨ ψ]] = [[φ]] + [[ψ]] - [[φ]]*[[ψ]] (Probabilistic Sum) """
        return x + y - x * y

    def implication(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """ [[φ → ψ]] = 1 - [[φ]] + [[φ]]*[[ψ]] (Reichenbach S-implication) """
        return 1 - x + x * y

    def sup(self, z: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """ Differentiable sup (for EXISTS) using LSE """
        if z.shape[dim] == 0:
            return torch.tensor(0.0, device=z.device) # Supremum of an empty set is 0
        return self.tau * torch.log(torch.sum(torch.exp(z / self.tau), dim=dim))

    def inf(self, z: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """ Differentiable inf (for FORALL) using LSE """
        if z.shape[dim] == 0:
            return torch.tensor(1.0, device=z.device) # Infimum of an empty set is 1
        return -self.tau * torch.log(torch.sum(torch.exp(-z / self.tau), dim=dim))


class PredicateMLP(nn.Module):
    """
    MLP for predicate grounding as described in Section 4.1 of the paper.
    Input is the concatenation of n term embeddings, output is a fuzzy truth value in [0, 1].
    """
    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int = 1):
        super().__init__()
        layers = []
        current_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Concatenated entity embeddings (batch_size, n * embedding_dim)
        Returns:
            torch.Tensor: Fuzzy truth value (batch_size, 1)
        """
        return torch.sigmoid(self.mlp(x)).squeeze(-1)
