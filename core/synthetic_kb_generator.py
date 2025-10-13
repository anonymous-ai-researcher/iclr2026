# synthetic_kb_generator.py

import random
from typing import List, Dict, Any

class SyntheticKB:
    """
    Generates a synthetic Knowledge Base (KB) containing constants, predicates,
    facts, and axioms. This allows the project to be run without requiring
    large, external datasets.
    """
    def __init__(self, num_constants=100, core_domain_ratio=0.2, num_predicates=3, num_facts=200, num_axioms=10):
        self.num_constants = num_constants
        self.num_predicates = num_predicates
        self.num_facts = num_facts
        self.num_axioms = num_axioms
        
        self.constants = list(range(num_constants))
        # [cite_start]The Core Domain consists of constants explicitly mentioned in the KB[cite: 434, 437].
        self.core_domain = self.constants[:int(num_constants * core_domain_ratio)]
        
        self.predicate_definitions = {f"P{i}": random.randint(1, 2) for i in range(num_predicates)}
        
        self.facts = self._generate_facts()
        self.axioms = self._generate_axioms()

    def _generate_facts(self) -> List[Dict[str, Any]]:
        """Generates random ground atoms (facts)."""
        facts = []
        for _ in range(self.num_facts):
            pred_name = random.choice(list(self.predicate_definitions.keys()))
            arity = self.predicate_definitions[pred_name]
            # Use tuple to make facts hashable for uniqueness check
            fact_args = tuple(random.sample(self.constants, arity))
            fact = {
                'type': 'atom',
                'name': pred_name,
                'args': fact_args
            }
            facts.append(fact)
        # Return unique facts
        return [dict(t) for t in {tuple(d.items()) for d in facts}]

    def _generate_axioms(self) -> List[Dict[str, Any]]:
        """
        Generates simple guarded axioms, e.g., ∀x (Guard(c,x) → Body(x)).
        [cite_start]The guard relativizes the quantification to a local neighborhood[cite: 209].
        """
        axioms = []
        for _ in range(self.num_axioms):
            binary_preds = [p for p, a in self.predicate_definitions.items() if a == 2]
            if len(binary_preds) < 1: continue # Need at least one binary predicate for a guard

            unary_preds = [p for p, a in self.predicate_definitions.items() if a == 1]
            if not unary_preds: continue # Need a unary predicate for the body

            guard_pred = random.choice(binary_preds)
            body_pred_unary = random.choice(unary_preds)
            
            # Ground one variable of the guard with a constant from the core domain
            c = random.choice(self.core_domain)

            axiom = {
                'type': 'forall',
                'var': 'x',
                'body': {
                    'type': 'implication',
                    'left': {'type': 'atom', 'name': guard_pred, 'args': ['c', 'x']}, # Guard atom
                    'right': {'type': 'atom', 'name': body_pred_unary, 'args': ['x']} # Body
                },
                'grounding_constants': {'c': c} # Store constant for grounding during training
            }
            axioms.append(axiom)
        return axioms

    def info(self):
        print("--- Synthetic Knowledge Base ---")
        print(f"Total Constants: {self.num_constants}")
        print(f"Core Domain Size: {len(self.core_domain)}")
        print(f"Predicate Definitions: {self.predicate_definitions}")
        print(f"Number of Unique Facts: {len(self.facts)}")
        print(f"Number of Axioms: {len(self.axioms)}")
        print("--------------------------------")
