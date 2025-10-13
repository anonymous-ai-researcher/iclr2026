# guardnet_model.py

import torch
import torch.nn as nn
from typing import Dict, Any

from neural_grounding import PredicateMLP

class GuardNet(nn.Module):
    """
    The GUARDNET Framework model.
    This class holds the learnable components of the system:
    1. [cite_start]Embeddings for all constants (entities) in the knowledge base[cite: 417].
    2. [cite_start]A dictionary of MLPs for all predicates[cite: 419].
    """
    def __init__(self, num_constants: int, embedding_dim: int, predicate_definitions: Dict[str, int], device):
        super().__init__()
        self.device = device
        
        # [cite_start]Each constant symbol is grounded as a learnable vector embedding[cite: 417].
        self.constant_embeddings = nn.Embedding(num_constants, embedding_dim).to(device)
        
        # [cite_start]Each n-ary predicate is grounded as a differentiable MLP[cite: 419].
        self.predicates = nn.ModuleDict({
            name: PredicateMLP(arity, embedding_dim).to(device)
            for name, arity in predicate_definitions.items()
        })

    def forward(self, formula: Dict[str, Any], parser, bindings: Dict[str, torch.Tensor], domain: torch.Tensor):
        """
        Computes the satisfaction of a formula by delegating to the formula parser.
        
        Args:
            formula: The logical formula to evaluate.
            parser: The FormulaParser instance.
            bindings: Initial variable bindings (e.g., for free variables).
            domain: The domain to quantify over.
            
        Returns:
            The fuzzy satisfaction degree of the formula.
        """
        return parser.get_satisfaction(formula, bindings, domain)
