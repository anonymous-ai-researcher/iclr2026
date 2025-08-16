# -*- coding: utf-8 -*-
"""
datasets.py

PyTorch Dataset and DataLoader implementation for DF-EL++.

This module handles:
1.  Parsing normalized axiom files.
2.  Mapping entities (concepts, roles, individuals) to integer indices.
3.  Creating a PyTorch Dataset for training, validation, and testing.
4.  Implementing a custom collate function for negative sampling during training.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import random
from src.utils import load_axioms, load_entities

class KBCDataset(Dataset):
    """
    A PyTorch Dataset for Knowledge Base Completion with EL++ axioms.
    """
    def __init__(self, data_path, split='train'):
        """
        Args:
            data_path (str): Path to the directory with preprocessed data.
            split (str): The data split to load ('train', 'valid', or 'test').
        """
        print(f"Loading {split} data from {data_path}...")
        self.axioms_str = load_axioms(data_path, split)
        
        concepts, roles, individuals = load_entities(data_path)
        
        # The full domain includes all concepts and individuals
        # In this implementation, we treat concepts and individuals as part of the same domain space for KBC ranking
        self.domain_entities = sorted(concepts + individuals)
        self.roles = sorted(roles)

        self.entity_to_idx = {entity: i for i, entity in enumerate(self.domain_entities)}
        self.role_to_idx = {role: i for i, role in enumerate(self.roles)}
        
        self.num_entities = len(self.domain_entities)
        self.num_roles = len(self.roles)

        self.axioms = self.parse_axioms(self.axioms_str)
        print(f"Loaded {len(self.axioms)} axioms for {split} split.")

    def __len__(self):
        return len(self.axioms)

    def __getitem__(self, idx):
        return self.axioms[idx]

    def parse_axioms(self, axioms_str):
        """
        Parses string axioms into a structured format with integer indices.
        """
        parsed = []
        for axiom_str in axioms_str:
            parts = axiom_str.split()
            # This is a simplified parser. A real implementation would be more robust.
            if "SubClassOf" in axiom_str:
                # NF1: A ⊑ B
                if len(parts) == 3 and parts[1] == "SubClassOf":
                    c1 = self.entity_to_idx.get(parts[0])
                    c2 = self.entity_to_idx.get(parts[2])
                    if c1 is not None and c2 is not None:
                        parsed.append({'type': 'nf1', 'c1': c1, 'c2': c2})
                # NF2: A1 ∩ A2 ⊑ B
                elif len(parts) == 5 and parts[1] == "And" and parts[3] == "SubClassOf":
                    c1 = self.entity_to_idx.get(parts[0])
                    c2 = self.entity_to_idx.get(parts[2])
                    c3 = self.entity_to_idx.get(parts[4])
                    if c1 is not None and c2 is not None and c3 is not None:
                        parsed.append({'type': 'nf2', 'c1': c1, 'c2': c2, 'c3': c3})
                # NF3: A ⊑ ∃r.B
                elif len(parts) == 5 and parts[1] == "SubClassOf" and parts[2] == "Some":
                    c1 = self.entity_to_idx.get(parts[0])
                    r = self.role_to_idx.get(parts[3])
                    c2 = self.entity_to_idx.get(parts[4])
                    if c1 is not None and r is not None and c2 is not None:
                        parsed.append({'type': 'nf3', 'c1': c1, 'r': r, 'c2': c2})
                # NF4: ∃r.A ⊑ B
                elif len(parts) == 5 and parts[0] == "Some" and parts[3] == "SubClassOf":
                    r = self.role_to_idx.get(parts[1])
                    c1 = self.entity_to_idx.get(parts[2])
                    c2 = self.entity_to_idx.get(parts[4])
                    if r is not None and c1 is not None and c2 is not None:
                        parsed.append({'type': 'nf4', 'r': r, 'c1': c1, 'c2': c2})
            # Add parsers for other axiom types (role inclusion, abox) if needed
        return parsed

class KBCCollator:
    """
    Custom collate function for the DataLoader to handle negative sampling.
    """
    def __init__(self, num_entities, num_negative_samples):
        self.num_entities = num_entities
        self.num_negative_samples = num_negative_samples

    def __call__(self, batch):
        positive_samples = batch
        negative_samples = []

        for axiom in positive_samples:
            for _ in range(self.num_negative_samples):
                neg_axiom = axiom.copy()
                
                # Corrupt either head or tail (simplified for NF1)
                if neg_axiom['type'] == 'nf1':
                    if random.random() < 0.5: # Corrupt head
                        neg_axiom['c1'] = random.randint(0, self.num_entities - 1)
                    else: # Corrupt tail
                        neg_axiom['c2'] = random.randint(0, self.num_entities - 1)
                # Extend corruption for other normal forms
                
                negative_samples.append(neg_axiom)
        
        # Group axioms by type for efficient batch processing in the model
        batched_axioms = {}
        all_samples = positive_samples + negative_samples
        for axiom in all_samples:
            axiom_type = axiom['type']
            if axiom_type not in batched_axioms:
                batched_axioms[axiom_type] = {k: [] for k in axiom.keys() if k != 'type'}
            for key, value in axiom.items():
                if key != 'type':
                    batched_axioms[axiom_type][key].append(value)
        
        # Convert lists to tensors
        for axiom_type, data in batched_axioms.items():
            for key, value in data.items():
                batched_axioms[axiom_type][key] = torch.tensor(value, dtype=torch.long)

        return batched_axioms

