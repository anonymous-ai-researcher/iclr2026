# formula_parser.py

import torch
from typing import Dict, Any

import fuzzy_ops as fops

class FormulaParser:
    """
    A recursive parser to compute the fuzzy satisfaction of a Guarded Fragment formula.
    It traverses the formula structure and applies the corresponding fuzzy operators.
    """
    def __init__(self, model, tau=0.1):
        self.model = model
        self.tau = tau

    def get_satisfaction(self, formula: Dict[str, Any], bindings: Dict[str, torch.Tensor], domain: torch.Tensor) -> torch.Tensor:
        """
        Recursively computes the satisfaction of a formula.

        Args:
            formula: The formula structure represented as a dictionary.
            bindings: A dictionary mapping variable names to their current embedding tensors.
            domain: A tensor of indices for the domain to quantify over.

        Returns:
            A tensor of satisfaction values. The shape depends on the formula.
        """
        formula_type = formula['type']

        if formula_type == 'atom':
            predicate = self.model.predicates[formula['name']]
            term_embeddings = []
            for arg in formula['args']:
                if isinstance(arg, str) and arg in bindings: # It's a bound variable
                    term_embeddings.append(bindings[arg])
                else: # It's a constant
                    const_idx = torch.tensor([arg], device=self.model.device)
                    emb = self.model.constant_embeddings(const_idx)
                    # Expand if necessary to match batch size from other bindings
                    if bindings:
                        batch_size = next(iter(bindings.values())).shape[0]
                        emb = emb.expand(batch_size, -1)
                    term_embeddings.append(emb)
            return predicate(term_embeddings)

        elif formula_type == 'conjunction':
            left_sat = self.get_satisfaction(formula['left'], bindings, domain)
            right_sat = self.get_satisfaction(formula['right'], bindings, domain)
            return fops.fuzzy_conjunction(left_sat, right_sat)

        elif formula_type == 'implication':
            left_sat = self.get_satisfaction(formula['left'], bindings, domain)
            right_sat = self.get_satisfaction(formula['right'], bindings, domain)
            return fops.fuzzy_implication_reichenbach(left_sat, right_sat)

        elif formula_type == 'forall':
            var_name = formula['var']
            
            # For quantification, we evaluate the body for each element in the domain.
            domain_embeddings = self.model.constant_embeddings(domain)
            
            # Vectorized implementation for efficiency
            # Replicate bindings for each element in the domain
            num_domain_elements = domain.shape[0]
            expanded_bindings = {}
            for key, val in bindings.items():
                # val shape: (batch_size, emb_dim) -> (batch_size * num_domain, emb_dim)
                expanded_bindings[key] = val.repeat_interleave(num_domain_elements, dim=0)

            # Add the quantified variable's bindings
            # domain_embeddings shape: (num_domain, emb_dim)
            # expanded_var shape: (batch_size * num_domain, emb_dim)
            batch_size = next(iter(bindings.values())).shape[0] if bindings else 1
            expanded_var = domain_embeddings.repeat(batch_size, 1)
            expanded_bindings[var_name] = expanded_var

            all_body_sats = self.get_satisfaction(formula['body'], expanded_bindings, domain)
            
            # Reshape to (batch_size, num_domain_elements) to apply quantifier
            all_body_sats = all_body_sats.view(batch_size, num_domain_elements)
            
            return fops.fuzzy_universal_quantifier(all_body_sats, self.tau)

        else:
            raise NotImplementedError(f"Formula type '{formula_type}' is not supported.")
