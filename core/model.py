# guardnet/model.py

import torch
import torch.nn as nn
from typing import Dict, List, Any

from layers import FuzzyLogicLayers, PredicateMLP
from dataloader import KnowledgeBase

class GuardNet(nn.Module):
    """
    The core GUARDNET model (Section 4)
    """
    def __init__(self, kb: KnowledgeBase, config: Dict):
        super().__init__()
        self.config = config
        self.kb = kb
        self.fuzzy_ops = FuzzyLogicLayers(temperature=config['lse_temperature'])

        # 1. Neural Grounding: Constants as learnable embeddings (Section 4.1)
        self.entity_embeddings = nn.Embedding(
            kb.num_entities,
            config['embedding_dim']
        )
        nn.init.xavier_uniform_(self.entity_embeddings.weight.data)

        # 2. Neural Grounding: Predicates as differentiable functions (MLPs) (Section 4.1)
        self.predicate_mlps = nn.ModuleDict()
        for pred_name, arity in kb.predicate_arities.items():
            input_dim = arity * config['embedding_dim']
            self.predicate_mlps[pred_name] = PredicateMLP(
                input_dim, config['predicate_mlp_hidden_dims']
            )

    def get_entity_embeddings(self, entity_ids: torch.Tensor) -> torch.Tensor:
        """Get entity embeddings"""
        return self.entity_embeddings(entity_ids)

    def forward(self, formulas: List[Any]) -> torch.Tensor:
        """
        Calculates the fuzzy satisfaction degree for a batch of formulas.
        Args:
            formulas (List[Any]): A list of parsed formulas.
        Returns:
            torch.Tensor: Satisfaction degree for each formula, shape: (batch_size,).
        """
        satisfactions = []
        for formula in formulas:
            # Each formula starts with an empty variable assignment
            satisfactions.append(self.satisfaction(formula, {}))
        return torch.stack(satisfactions)

    def satisfaction(self, formula: Any, var_assignments: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Recursively calculates the fuzzy satisfaction degree for a single formula.
        Args:
            formula (Any): A parsed formula (e.g., ('atom', 'Predicate', ('c1', 'x'))).
            var_assignments (Dict): The current mapping from variables to entity ID tensors.
        """
        op = formula[0]

        if op == 'atom':
            _, pred_name, terms = formula
            term_embeds = []
            for term in terms:
                if term in self.kb.entity_to_id: # It's a constant
                    entity_id = self.kb.entity_to_id[term]
                    embed = self.entity_embeddings(torch.tensor(entity_id, device=self.config['device']))
                    term_embeds.append(embed)
                elif term in var_assignments: # It's a bound variable
                    entity_ids = var_assignments[term]
                    embed = self.entity_embeddings(entity_ids)
                    term_embeds.append(embed)
                else: # Error: Unbound variable
                    raise ValueError(f"Unbound variable {term} in formula {formula}")

            # Concatenate embeddings to feed into the MLP
            # Note: Broadcasting needs to be handled to match batch dimensions
            shapes = [e.shape[0] for e in term_embeds if e.dim() > 1]
            batch_size = max(shapes) if shapes else 1
            
            processed_embeds = []
            for e in term_embeds:
                if e.dim() == 1 and batch_size > 1:
                    processed_embeds.append(e.unsqueeze(0).expand(batch_size, -1))
                else:
                    processed_embeds.append(e)

            concatenated_embeds = torch.cat(processed_embeds, dim=-1)
            
            return self.predicate_mlps[pred_name](concatenated_embeds)

        elif op == 'negation':
            return self.fuzzy_ops.negation(self.satisfaction(formula[1], var_assignments))

        elif op == 'conjunction':
            return self.fuzzy_ops.conjunction(
                self.satisfaction(formula[1], var_assignments),
                self.satisfaction(formula[2], var_assignments)
            )
            
        elif op == 'disjunction':
            return self.fuzzy_ops.disjunction(
                self.satisfaction(formula[1], var_assignments),
                self.satisfaction(formula[2], var_assignments)
            )

        elif op == 'forall':
            # ∀x(α → ψ)
            _, var, (guard, body) = formula
            
            # Key: The source of GF's efficiency. We only iterate over facts that satisfy the guard.
            # Here we simplify by evaluating over all entities; an optimized implementation would use kb.get_facts_for_guard
            all_entities = torch.arange(self.kb.num_entities, device=self.config['device'])
            
            new_assignments = var_assignments.copy()
            new_assignments[var] = all_entities
            
            guard_sat = self.satisfaction(guard, new_assignments)
            body_sat = self.satisfaction(body, new_assignments)
            
            implication_values = self.fuzzy_ops.implication(guard_sat, body_sat)
            
            # Use inf to aggregate all instances
            return self.fuzzy_ops.inf(implication_values)

        # Handling for 'exists' can also be added
        # elif op == 'exists': ...

        else:
            raise NotImplementedError(f"Operator {op} not implemented.")
