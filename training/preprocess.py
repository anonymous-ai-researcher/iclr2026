# -*- coding: utf-8 -*-
"""
preprocess.py

Ontology Normalization and Preprocessing Script for DF-EL++.

This script implements the normalization procedure described in Section 4.1
of the paper "Fast and Faithful: Scalable Neuro-Symbolic Learning and
Reasoning with Differentiable Fuzzy EL++".

It takes an OWL ontology as input, parses it, applies a series of
transformation rules (NR1-NR8) to decompose complex axioms into one of the
six normal forms, and saves the result into a set of text files suitable for
training.

Key functionalities:
1.  Loads an OWL ontology using the owlready2 library.
2.  Iteratively applies normalization rules to decompose complex axioms.
3.  Handles the creation of fresh concept names for decomposition.
4.  Collects all entities (concepts, roles, individuals).
5.  Saves the normalized axioms and entity lists to disk.
6.  Performs a train/validation/test split of the axioms.

Usage:
    python preprocess.py --input_ontology <path_to_owl> --output_dir <path_to_output>
"""

import os
import argparse
import random
from collections import defaultdict
from owlready2 import *
from tqdm import tqdm

# Set a seed for reproducibility of the data splits
random.seed(42)

def is_complex(concept):
    """
    Checks if a concept is complex (i.e., not an atomic concept name or Top).
    Corresponds to C_hat or D_hat in the paper's normalization rules.
    """
    return not (isinstance(concept, ThingClass) and not isinstance(concept, (And, Or, Not, Restriction))) and concept is not Thing

def normalize_ontology(onto):
    """
    Applies the normalization rules NR1-NR8 from Section 4.1 of the paper.

    Args:
        onto (owlready2.Ontology): The ontology to be normalized.

    Returns:
        list: A list of strings, where each string represents a normalized axiom.
    """
    axioms = []
    # Collect all original subclass axioms
    for axiom in onto.classes():
        for superclass in axiom.is_a:
            if isinstance(superclass, ThingClass): # Ignore non-class constructs
                 axioms.append((axiom, superclass))

    normalized_axioms = []
    processing_queue = axioms
    
    # A counter for creating fresh concept names (A_new in the paper)
    new_concept_counter = 0

    with onto:
        while processing_queue:
            sub, sup = processing_queue.pop(0)
            
            sub_is_complex = is_complex(sub)
            sup_is_complex = is_complex(sup)

            # Rule NR1: C_hat ⊑ D_hat => C_hat ⊑ A_new, A_new ⊑ D_hat
            if sub_is_complex and sup_is_complex:
                new_concept_name = f"FreshConcept_{new_concept_counter}"
                A_new = types.new_class(new_concept_name, (Thing,))
                new_concept_counter += 1
                processing_queue.append((sub, A_new))
                processing_queue.append((A_new, sup))
                continue

            # Rule NR8: B ⊑ D ∩ E <=> B ⊑ D, B ⊑ E
            if isinstance(sup, And):
                for conjunct in sup.Classes:
                    processing_queue.append((sub, conjunct))
                continue

            # Rules for complex LHS
            if isinstance(sub, And):
                # Assuming And has two components for simplicity, as in the paper
                c1, c2 = sub.Classes
                # Rule NR2: C_hat ∩ D ⊑ B
                if is_complex(c1):
                    new_concept_name = f"FreshConcept_{new_concept_counter}"
                    A_new = types.new_class(new_concept_name, (Thing,))
                    new_concept_counter += 1
                    processing_queue.append((c1, A_new))
                    processing_queue.append((And([A_new, c2]), sup))
                # Rule NR3: D ∩ C_hat ⊑ B
                elif is_complex(c2):
                    new_concept_name = f"FreshConcept_{new_concept_counter}"
                    A_new = types.new_class(new_concept_name, (Thing,))
                    new_concept_counter += 1
                    processing_queue.append((c2, A_new))
                    processing_queue.append((And([c1, A_new]), sup))
                else: # NF2: A1 ∩ A2 ⊑ B
                    normalized_axioms.append(f"{c1.name} And {c2.name} SubClassOf {sup.name}")
                continue
            
            if isinstance(sub, Restriction) and sub.type == SOME:
                prop, filler = sub.property, sub.value
                # Rule NR4: ∃r.C_hat ⊑ B
                if is_complex(filler):
                    new_concept_name = f"FreshConcept_{new_concept_counter}"
                    A_new = types.new_class(new_concept_name, (Thing,))
                    new_concept_counter += 1
                    processing_queue.append((filler, A_new))
                    processing_queue.append((prop.some(A_new), sup))
                else: # NF4: ∃r.A ⊑ B
                    normalized_axioms.append(f"Some {prop.name} {filler.name} SubClassOf {sup.name}")
                continue

            # Rules for complex RHS
            if isinstance(sup, Restriction) and sup.type == SOME:
                prop, filler = sup.property, sup.value
                # Rule NR5: B ⊑ ∃r.C_hat
                if is_complex(filler):
                    new_concept_name = f"FreshConcept_{new_concept_counter}"
                    A_new = types.new_class(new_concept_name, (Thing,))
                    new_concept_counter += 1
                    processing_queue.append((A_new, filler))
                    processing_queue.append((sub, prop.some(A_new)))
                else: # NF3: A ⊑ ∃r.B
                    normalized_axioms.append(f"{sub.name} SubClassOf Some {prop.name} {filler.name}")
                continue
            
            # If none of the above rules match, it should be in NF1: A ⊑ B
            if isinstance(sub, ThingClass) and isinstance(sup, ThingClass):
                 normalized_axioms.append(f"{sub.name} SubClassOf {sup.name}")
            else:
                 print(f"Warning: Could not normalize axiom: {sub} SubClassOf {sup}")

    # Add Role Inclusions (already in normal form)
    for role in onto.object_properties():
        for super_role in role.is_a:
            if isinstance(super_role, ObjectPropertyClass):
                normalized_axioms.append(f"{role.name} SubPropertyOf {super_role.name}")
    
    # Add ABox assertions (already in normal form)
    for ind in onto.individuals():
        for cls in ind.is_a:
            if isinstance(cls, ThingClass):
                normalized_axioms.append(f"{ind.name} Type {cls.name}")
        for prop in onto.object_properties():
            for val in prop[ind]:
                if isinstance(val, Individual):
                    normalized_axioms.append(f"{ind.name} {prop.name} {val.name}")

    return normalized_axioms

def save_to_file(data, filepath):
    """Saves a list of strings to a file, one per line."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(f"{item}\n")

def main(args):
    print(f"Loading ontology from: {args.input_ontology}")
    # It is recommended to use a dedicated world for each ontology
    world = World()
    onto = world.get_ontology(args.input_ontology).load()

    print("Starting ontology normalization...")
    normalized_axioms = normalize_ontology(onto)
    print(f"Normalization complete. Found {len(normalized_axioms)} normalized axioms.")

    # Collect all entities
    concepts = {c.name for c in onto.classes()}
    roles = {r.name for r in onto.object_properties()}
    individuals = {i.name for i in onto.individuals()}
    
    # Add 'Thing' (Top concept)
    concepts.add("Thing")

    print(f"Found {len(concepts)} concepts, {len(roles)} roles, {len(individuals)} individuals.")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Saving processed files to: {args.output_dir}")

    # Save entity lists
    save_to_file(sorted(list(concepts)), os.path.join(args.output_dir, 'concepts.txt'))
    save_to_file(sorted(list(roles)), os.path.join(args.output_dir, 'roles.txt'))
    save_to_file(sorted(list(individuals)), os.path.join(args.output_dir, 'individuals.txt'))

    # Split axioms into train, validation, and test sets (80/10/10 split from paper)
    random.shuffle(normalized_axioms)
    total_axioms = len(normalized_axioms)
    train_end = int(total_axioms * 0.8)
    valid_end = int(total_axioms * 0.9)

    train_axioms = normalized_axioms[:train_end]
    valid_axioms = normalized_axioms[train_end:valid_end]
    test_axioms = normalized_axioms[valid_end:]

    save_to_file(train_axioms, os.path.join(args.output_dir, 'train.txt'))
    save_to_file(valid_axioms, os.path.join(args.output_dir, 'valid.txt'))
    save_to_file(test_axioms, os.path.join(args.output_dir, 'test.txt'))
    
    print("Splitting axioms into train/validation/test sets:")
    print(f"  - Train: {len(train_axioms)} axioms")
    print(f"  - Validation: {len(valid_axioms)} axioms")
    print(f"  - Test: {len(test_axioms)} axioms")
    print("Preprocessing finished successfully.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DF-EL++ Ontology Preprocessing Script")
    parser.add_argument('--input_ontology', type=str, required=True, help='Path to the input OWL ontology file.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the processed files.')
    
    args = parser.parse_args()
    main(args)
