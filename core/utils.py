# -*- coding: utf-8 -*-
"""
utils.py

Utility functions for the DF-EL++ project.

This module contains helper functions for tasks such as loading data,
setting random seeds for reproducibility, and managing model checkpoints.
"""

import os
import random
import numpy as np
import torch

def set_seed(seed):
    """
    Sets the random seed for reproducibility across all relevant libraries.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_entities(data_path):
    """
    Loads concepts, roles, and individuals from text files.
    
    Args:
        data_path (str): Path to the directory containing processed data.

    Returns:
        tuple: Three lists containing concepts, roles, and individuals.
    """
    def read_file(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f.readlines()]

    concepts = read_file(os.path.join(data_path, 'concepts.txt'))
    roles = read_file(os.path.join(data_path, 'roles.txt'))
    individuals = read_file(os.path.join(data_path, 'individuals.txt'))
    
    return concepts, roles, individuals

def load_axioms(data_path, split='train'):
    """
    Loads axioms for a specific data split (train, valid, or test).
    
    Args:
        data_path (str): Path to the directory containing processed data.
        split (str): The data split to load ('train', 'valid', or 'test').

    Returns:
        list: A list of strings, where each string is a normalized axiom.
    """
    filepath = os.path.join(data_path, f'{split}.txt')
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()]

def save_checkpoint(model, optimizer, epoch, mrr, save_path):
    """
    Saves a model checkpoint.
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'mrr': mrr,
    }, save_path)
    print(f"Saved model checkpoint to {save_path}")

def load_checkpoint(model, optimizer, load_path):
    """
    Loads a model checkpoint.
    """
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    mrr = checkpoint['mrr']
    print(f"Loaded model checkpoint from {load_path} at epoch {epoch} with MRR {mrr:.4f}")
    return model, optimizer, epoch, mrr

