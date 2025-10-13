
# neural_grounding.py

import torch
import torch.nn as nn
from typing import List

class PredicateMLP(nn.Module):
    """
    A Multi-Layer Perceptron (MLP) to ground an n-ary predicate, as
    [cite_start]described in Section 4.1 of the paper[cite: 419].
    It takes concatenated term embeddings as input and outputs a
    fuzzy truth value in the interval [0, 1].
    """
    def __init__(self, arity: int, embedding_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.arity = arity
        self.embedding_dim = embedding_dim
        input_dim = arity * embedding_dim

        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            [cite_start]nn.ReLU(),  # ReLU activation for hidden layers[cite: 426].
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            [cite_start]nn.Sigmoid() # Sigmoid activation for the output layer to map to [0, 1][cite: 427].
        )

    def forward(self, term_embeddings: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            term_embeddings: A list of tensors, where each tensor is the
                             embedding for one argument of the predicate.
                             Shape of each tensor: (batch_size, embedding_dim).

        Returns:
            A tensor of fuzzy truth values, shape: (batch_size,).
        """
        if len(term_embeddings) != self.arity:
            raise ValueError(f"Expected {self.arity} term embeddings, but got {len(term_embeddings)}")

        # [cite_start]Concatenation is used as a parameter-free, information-preserving operation[cite: 422].
        concatenated_embeddings = torch.cat(term_embeddings, dim=-1)
        return self.layers(concatenated_embeddings).squeeze(-1)
