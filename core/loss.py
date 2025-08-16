# -*- coding: utf-8 -*-
"""
loss.py

Implementation of the unified semantic loss function for DF-EL++.

This module defines the loss function derived directly from the Göguen
implication, as presented in Equation (1), Section 4.3 of the paper.
"""

import torch
import torch.nn as nn

class UnifiedSemanticLoss(nn.Module):
    """
    The unified, theoretically-grounded loss function for any axiom.
    
    L_axiom(α) = Σ_d max(0, 1 - RHS(d) / (LHS(d) + ε))

    This loss function minimizes the dissatisfaction of each axiom, where
    dissatisfaction is defined as 1 - J_G(LHS, RHS).
    """
    def __init__(self, epsilon=1e-12):
        """
        Args:
            epsilon (float): A small constant for numerical stability to avoid
                             division by zero.
        """
        super(UnifiedSemanticLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, lhs_membership, rhs_membership):
        """
        Computes the loss for a batch of axioms.

        Args:
            lhs_membership (torch.Tensor): A tensor containing the fuzzy
                membership degrees for the premise (LHS) of the axioms.
            rhs_membership (torch.Tensor): A tensor containing the fuzzy
                membership degrees for the conclusion (RHS) of the axioms.

        Returns:
            torch.Tensor: A scalar tensor representing the total loss for the batch.
        """
        # Dissatisfaction = 1 - (RHS / (LHS + ε))
        # We want to minimize this, which is equivalent to minimizing max(0, dissatisfaction)
        dissatisfaction = 1.0 - (rhs_membership / (lhs_membership + self.epsilon))
        
        # Apply relu to get max(0, dissatisfaction)
        loss_per_element = torch.relu(dissatisfaction)
        
        # The total loss is the sum over all individuals in the domain,
        # which corresponds to summing over all elements in the tensor.
        total_loss = torch.sum(loss_per_element)
        
        return total_loss

