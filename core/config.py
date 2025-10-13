# guardnet/config.py

import torch

def get_config():
    """Get all hyperparameters and configurations"""
    config = {
        # --- Environment and Device Config ---
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "seed": 42,

        # --- Model Architecture (Appendix E.2) ---
        "embedding_dim": 200,
        "predicate_mlp_hidden_dims": [256, 256],

        # --- Training and Optimization (Appendix E.2) ---
        "optimizer": "AdamW",
        "learning_rate": 5e-4,
        "weight_decay": 5e-5,
        "batch_size": 512,
        "num_epochs": 100, # Can be adjusted based on the dataset size

        # --- Learning Rate Scheduler ---
        "lr_scheduler_factor": 0.5,
        "lr_scheduler_patience": 5,

        # --- Early Stopping ---
        "early_stopping_patience": 15,

        # --- Hybrid Domain and Loss Curriculum (Appendix E.2) ---
        "lambda_start": 0.9, # Initially, 90% weight is on the fidelity loss
        "lambda_end": 0.4,   # At the end, 40% weight is on the fidelity loss

        # --- Fuzzy Semantics Fixed Parameters (Appendix E.2) ---
        "lse_temperature": 0.1, # The temperature τ for LogSumExp

        # --- Data and Paths ---
        "dataset_name": "SNOMED_CT_SAMPLE", # Example dataset name
        "data_path": "./data/",
        "checkpoint_path": "./checkpoints/",
        "log_path": "./logs/"
    }
    return config
