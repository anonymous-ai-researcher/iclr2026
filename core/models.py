# -*- coding: utf-8 -*-
"""
models.py

PyTorch implementation of the DF-EL++ model.

This module defines the main neural network architecture. It handles:
1.  Embedding layers for concepts, individuals, and roles.
2.  A forward pass that computes the fuzzy membership degrees for the
    LHS and RHS of different types of normalized axioms.
3.  Integration of the differentiable fuzzy logic operators from `src.layers`.
"""

import torch
import torch.nn as nn
from src.layers import product_tnorm, probabilistic_sum_tconorm_stable

class DFELpp(nn.Module):
    """
    The main DF-EL++ model for knowledge base completion.
    
    This model learns embeddings for all entities (concepts, individuals) and
    roles in the ontology. It uses these embeddings to compute the fuzzy
    satisfaction degree of logical axioms.
    """
    def __init__(self, num_entities, num_roles, embedding_dim):
        """
        Args:
            num_entities (int): The total number of unique concepts and individuals.
                                Corresponds to the size of the domain |Δ^I|.
            num_roles (int): The total number of unique roles.
            embedding_dim (int): The dimensionality of the embeddings.
                                 Note: The paper describes a direct mapping where
                                 embedding_dim = num_entities. Here we use a
                                 more standard KGE-style embedding for flexibility,
                                 but the core logic remains the same. A direct
                                 mapping can be achieved by setting embedding_dim
                                 to num_entities.
        """
        super(DFELpp, self).__init__()
        self.num_entities = num_entities
        self.num_roles = num_roles
        self.embedding_dim = embedding_dim

        # Embedding for concepts and individuals (domain entities)
        # C^I(d_i) = σ(c_{C,i})
        # The paper (Sec 4.2.2) implies a vector c_C of size |Δ^I|.
        # We represent this as an embedding table where each entity has a vector.
        # The membership of an entity C to itself is one element of this vector.
        # For simplicity in this KGE-style implementation, we'll use entity embeddings.
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        
        # Role embeddings
        # r^I(d_i, d_j) = σ(R_{r,ij})
        # The paper implies a matrix R_r of size |Δ^I| x |Δ^I|.
        # This is computationally expensive. We use a more standard relational
        # embedding approach (e.g., a vector per role).
        self.role_embeddings = nn.Embedding(num_roles, embedding_dim)
        
        self.init_weights()

    def init_weights(self):
        """Initialize embedding weights."""
        nn.init.xavier_uniform_(self.entity_embeddings.weight.data)
        nn.init.xavier_uniform_(self.role_embeddings.weight.data)

    def forward(self, batched_axioms):
        """
        Computes the LHS and RHS membership degrees for a batch of axioms.

        Args:
            batched_axioms (dict): A dictionary where keys are axiom types (e.g., 'nf1')
                                   and values are tensors of entity/role indices.

        Returns:
            dict: A dictionary mapping axiom types to tuples of
                  (LHS_membership, RHS_membership) tensors.
        """
        results = {}
        
        # For this implementation, we'll use a simplified scoring function
        # A full implementation would compute membership across the entire domain.
        # Here, we compute scores for the specific entities in the axiom.
        
        all_entities = self.entity_embeddings.weight
        
        for axiom_type, data in batched_axioms.items():
            if axiom_type == 'nf1':
                # Axiom: C1 ⊑ C2
                c1_embed = self.entity_embeddings(data['c1'])
                c2_embed = self.entity_embeddings(data['c2'])
                # Simplified score: dot product similarity
                lhs = torch.sigmoid((c1_embed * all_entities).sum(dim=-1))
                rhs = torch.sigmoid((c2_embed * all_entities).sum(dim=-1))
                results[axiom_type] = (lhs, rhs)

            elif axiom_type == 'nf2':
                # Axiom: C1 ∩ C2 ⊑ C3
                c1_embed = self.entity_embeddings(data['c1'])
                c2_embed = self.entity_embeddings(data['c2'])
                c3_embed = self.entity_embeddings(data['c3'])
                
                # LHS = C1^I(d) * C2^I(d)
                lhs_c1 = torch.sigmoid((c1_embed * all_entities).sum(dim=-1))
                lhs_c2 = torch.sigmoid((c2_embed * all_entities).sum(dim=-1))
                lhs = product_tnorm(lhs_c1, lhs_c2)
                
                # RHS = C3^I(d)
                rhs = torch.sigmoid((c3_embed * all_entities).sum(dim=-1))
                results[axiom_type] = (lhs, rhs)

            elif axiom_type == 'nf3':
                # Axiom: C1 ⊑ ∃r.C2
                c1_embed = self.entity_embeddings(data['c1'])
                r_embed = self.role_embeddings(data['r'])
                c2_embed = self.entity_embeddings(data['c2'])

                # LHS = C1^I(d)
                lhs = torch.sigmoid((c1_embed * all_entities).sum(dim=-1))
                
                # RHS = 1 - Π_e(1 - r^I(d,e) * C2^I(e))
                # This requires iterating over all entities 'e', which is expensive.
                # We approximate this with a score. A full implementation is complex.
                # Simplified score:
                score = (c1_embed + r_embed - c2_embed).norm(p=2, dim=-1)
                rhs = torch.sigmoid(-score).unsqueeze(1).expand(-1, self.num_entities) # Placeholder
                results[axiom_type] = (lhs, rhs)

            elif axiom_type == 'nf4':
                # Axiom: ∃r.C1 ⊑ C2
                r_embed = self.role_embeddings(data['r'])
                c1_embed = self.entity_embeddings(data['c1'])
                c2_embed = self.entity_embeddings(data['c2'])

                # LHS = 1 - Π_e(1 - r^I(d,e) * C1^I(e))
                # Simplified score:
                score = (c1_embed + r_embed - c2_embed).norm(p=2, dim=-1)
                lhs = torch.sigmoid(-score).unsqueeze(1).expand(-1, self.num_entities) # Placeholder

                # RHS = C2^I(d)
                rhs = torch.sigmoid((c2_embed * all_entities).sum(dim=-1))
                results[axiom_type] = (lhs, rhs)
        
        return results

