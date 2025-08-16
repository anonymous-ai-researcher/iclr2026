# -*- coding: utf-8 -*-
"""
evaluate_kbc.py

Evaluation script for the Knowledge Base Completion (KBC) task.

This script loads a trained DF-EL++ model and evaluates its performance on
the test set using a standard filtered ranking protocol.

Key functionalities:
1.  Loads a trained model checkpoint.
2.  Loads the test set and the full set of known axioms for filtering.
3.  For each test axiom, it generates corrupted candidates (head and tail).
4.  Scores all candidates and ranks the true axiom.
5.  Applies filtering to remove other known true axioms from the ranking.
6.  Calculates and reports Hits@1, Hits@10, Hits@100, and MRR.
"""
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from src.datasets import KBCDataset
from src.models import DFELpp
from src.utils import load_checkpoint

def get_ranks(model, axiom, all_entities_tensor, device, filter_triples):
    """
    Computes the rank of a single test axiom against all possible corruptions.
    This is a simplified example for NF1 axioms (C1 ⊑ C2). A full implementation
    would handle all normal forms.
    """
    model.eval()
    
    # Ensure axiom components are single integer values
    c1_idx = axiom['c1'].item()
    c2_idx = axiom['c2'].item()
    
    # --- Head Prediction (corrupt c1) ---
    c1_corrupted = all_entities_tensor
    c2_fixed = torch.tensor([c2_idx] * model.num_entities, device=device)
    
    # For a more faithful implementation, the model's forward pass should be
    # adapted to score single axioms or batches of corrupted triples.
    # Here, we simulate this with a simplified scoring mechanism.
    c1_embeds = model.entity_embeddings(c1_corrupted)
    c2_embed = model.entity_embeddings(c2_fixed)
    
    # A simple score (e.g., dot product) for demonstration.
    # The actual scoring should reflect the fuzzy semantics.
    head_scores = torch.sum(c1_embeds * c2_embed, dim=1)
    
    # Filtering: remove scores of other known true axioms
    target_score = head_scores[c1_idx].item()
    
    # Create a copy of scores for filtering
    filtered_head_scores = head_scores.clone()
    
    filter_mask = torch.zeros(model.num_entities, dtype=torch.bool, device=device)
    for i in range(model.num_entities):
        if (i, c2_idx) in filter_triples and i != c1_idx:
            filter_mask[i] = True
            
    filtered_head_scores[filter_mask] = -np.inf # Set scores of known triples to -inf
    
    head_rank = (filtered_head_scores > target_score).sum().item() + 1

    # --- Tail Prediction (corrupt c2) ---
    c1_fixed = torch.tensor([c1_idx] * model.num_entities, device=device)
    c2_corrupted = all_entities_tensor
    
    c1_embed = model.entity_embeddings(c1_fixed)
    c2_embeds = model.entity_embeddings(c2_corrupted)
    
    tail_scores = torch.sum(c1_embed * c2_embeds, dim=1)
    
    target_score = tail_scores[c2_idx].item()
    filtered_tail_scores = tail_scores.clone()

    filter_mask = torch.zeros(model.num_entities, dtype=torch.bool, device=device)
    for i in range(model.num_entities):
        if (c1_idx, i) in filter_triples and i != c2_idx:
            filter_mask[i] = True
            
    filtered_tail_scores[filter_mask] = -np.inf
    
    tail_rank = (filtered_tail_scores > target_score).sum().item() + 1
    
    return head_rank, tail_rank


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")

    # --- 1. Load Data and Model ---
    test_dataset = KBCDataset(args.data_path, 'test')
    
    # Load all axioms for the filtering step
    print("Loading all axioms for filtering...")
    train_axioms = KBCDataset(args.data_path, 'train').axioms
    valid_axioms = KBCDataset(args.data_path, 'valid').axioms
    all_known_axioms = train_axioms + valid_axioms + test_dataset.axioms
    
    # Create a set of (head, tail) tuples for fast filtering (simplified for NF1)
    filter_triples = set()
    for ax in all_known_axioms:
        if ax['type'] == 'nf1':
            filter_triples.add((ax['c1'], ax['c2']))

    model = DFELpp(
        num_entities=test_dataset.num_entities,
        num_roles=test_dataset.num_roles,
        embedding_dim=args.embedding_dim
    ).to(device)

    model, _, _, _ = load_checkpoint(model, None, args.model_path)
    print("Model loaded successfully.")

    # --- 2. Evaluation Loop ---
    ranks = []
    all_entities_tensor = torch.arange(model.num_entities).to(device)
    
    # Use a DataLoader for batching during evaluation for efficiency
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    print("Starting evaluation on the test set...")
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            # This simplified version processes one axiom at a time.
            # A fully optimized version would process batches.
            for i in range(len(batch)):
                axiom = {k: v[i] for k, v in batch.items()}
                if axiom['type'] == 'nf1':
                    head_rank, tail_rank = get_ranks(model, axiom, all_entities_tensor, device, filter_triples)
                    ranks.append(head_rank)
                    ranks.append(tail_rank)

    ranks = np.array(ranks)
    
    # --- 3. Calculate and Report Metrics ---
    if len(ranks) == 0:
        print("No compatible axioms (e.g., NF1) found in the test set to evaluate.")
        return
        
    hits_at_1 = np.mean(ranks <= 1) * 100
    hits_at_10 = np.mean(ranks <= 10) * 100
    hits_at_100 = np.mean(ranks <= 100) * 100
    mrr = np.mean(1.0 / ranks)
    
    print("\n--- KBC Evaluation Results ---")
    print(f"Hits@1:   {hits_at_1:.2f}%")
    print(f"Hits@10:  {hits_at_10:.2f}%")
    print(f"Hits@100: {hits_at_100:.2f}%")
    print(f"MRR:      {mrr:.4f}")
    print("------------------------------")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DF-EL++ KBC Evaluation Script")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model checkpoint (.pt file).')
    parser.add_argument('--data_path', type=str, required=True, help='Path to preprocessed data directory.')
    parser.add_argument('--embedding_dim', type=int, default=200, help='Dimensionality of embeddings (must match the trained model).')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for evaluation.')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA even if available.')
    
    args = parser.parse_args()
    main(args)
