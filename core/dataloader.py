# guardnet/dataloader.py

import torch
from torch.utils.data import Dataset, DataLoader
import random
from typing import List, Dict, Tuple, Any

class KnowledgeBase:
    """A simple container for the knowledge base"""
    def __init__(self, facts: List[Tuple], rules: List[str]):
        self.entities = sorted(list(set(term for fact in facts for term in fact[1:] if isinstance(term, str))))
        self.predicates = sorted(list(set(fact[0] for fact in facts)))
        
        self.entity_to_id = {e: i for i, e in enumerate(self.entities)}
        self.id_to_entity = {i: e for i, e in enumerate(self.entities)}
        
        self.predicate_to_id = {p: i for i, p in enumerate(self.predicates)}
        
        self.facts = facts
        self.rules_str = rules
        self.parsed_rules = [self.parse_formula(rule) for rule in self.rules_str]

        self.predicate_arities = self._get_arities(facts)
        
        print(f"Knowledge Base Initialized:")
        print(f"  - {self.num_entities} entities")
        print(f"  - {self.num_predicates} predicates")
        print(f"  - {len(self.facts)} facts")
        print(f"  - {len(self.rules_str)} rules")

    @property
    def num_entities(self):
        return len(self.entities)

    @property
    def num_predicates(self):
        return len(self.predicates)

    def _get_arities(self, facts: List[Tuple]) -> Dict[str, int]:
        arities = {}
        # Adjust for facts like ('IsExpert', 'Bob')
        for fact in facts:
            pred = fact[0]
            arity = len(fact) - 1
            if pred not in arities:
                arities[pred] = arity
        return arities

    def parse_formula(self, formula_str: str) -> Any:
        """
        A very basic S-expression parser for GF formulas.
        Example: (forall x (implies (Follows Alice x) (Trusts Alice x)))
        """
        tokens = formula_str.replace('(', ' ( ').replace(')', ' ) ').split()
        
        def parse(token_list):
            token = token_list.pop(0)
            if token == '(':
                exp = []
                while token_list[0] != ')':
                    exp.append(parse(token_list))
                token_list.pop(0) # pop ')'
                
                # Convert to our internal format
                op = exp[0]
                if op in ['forall', 'exists']:
                    # (quantifier, var, body)
                    return (op, exp[1], exp[2])
                elif op in ['implies', 'conjunction', 'disjunction']:
                    # (operator, arg1, arg2)
                    # Note: 'implies' is not a standard operator in our fuzzy logic,
                    # it is part of the forall structure. We adjust the parser.
                    if op == 'implies':
                        return ('implies', exp[1], exp[2])
                    return (op, exp[1], exp[2])
                elif op in ['negation']:
                    return (op, exp[1])
                else: # This is an atom
                    # ('atom', pred_name, (term1, term2, ...))
                    return ('atom', exp[0], tuple(exp[1:]))
            else:
                return token
        
        parsed_expr = parse(tokens)
        
        # Post-process to handle the forall->implies structure
        def transform_implies(expr):
            if isinstance(expr, tuple):
                op = expr[0]
                if op == 'forall':
                    var, body = expr[1], expr[2]
                    if body[0] == 'implies':
                        return ('forall', var, (body[1], body[2]))
                return tuple(transform_implies(e) for e in expr)
            return expr
            
        return transform_implies(parsed_expr)


class GuardNetDataset(Dataset):
    """A PyTorch Dataset to create training data for GUARDNET"""
    def __init__(self, kb: KnowledgeBase):
        self.kb = kb
        # In this simplified version, we only train on the rules
        self.formulas = self.kb.parsed_rules

    def __len__(self):
        return len(self.formulas)

    def __getitem__(self, idx):
        return self.formulas[idx]

def load_dummy_data() -> KnowledgeBase:
    """Creates a simple dummy knowledge base for demonstration"""
    # Facts: (predicate, term1, term2, ...)
    facts = [
        ('Follows', 'Alice', 'Bob'),
        ('Follows', 'Alice', 'Charlie'),
        ('Trusts', 'Alice', 'Bob'),
        ('IsExpert', 'Bob')
    ]
    
    # Rules: Strings using S-expression syntax
    rules = [
        "(forall x (implies (Follows Alice x) (Trusts Alice x)))"
    ]
    
    return KnowledgeBase(facts, rules)

def create_dataloader(kb: KnowledgeBase, batch_size: int, shuffle: bool = True):
    dataset = GuardNetDataset(kb)
    # Custom collate_fn to handle a list of formulas
    def collate_fn(batch):
        return batch
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
