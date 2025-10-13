# main.py

import torch
from utils import set_seed
from synthetic_kb_generator import SyntheticKB
from guardnet_model import GuardNet
from trainer import Trainer

def main():
    """
    Main execution script for the GUARDNET project.
    """
    # Set seed for reproducibility
    set_seed(42)

    # Configuration based on the paper's appendix and common practices
    config = {
        [cite_start]'embedding_dim': 200,   # [cite: 1085]
        [cite_start]'lr': 5e-4,             # [cite: 1088]
        [cite_start]'wd': 5e-5,             # [cite: 1088]
        'batch_size': 16,       # Adjusted for demonstration purposes
        'epochs': 50,
        [cite_start]'tau': 0.1,             # Temperature for LSE approximation of quantifiers [cite: 331, 1096]
    }

    # Parameters for the Synthetic Knowledge Base
    kb_config = {
        'num_constants': 100,
        'core_domain_ratio': 0.2,
        'num_predicates': 5,
        'num_facts': 200,
        'num_axioms': 20
    }

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Generate Synthetic Knowledge Base
    print("Generating Synthetic Knowledge Base...")
    kb = SyntheticKB(**kb_config)
    kb.info()
    
    # 2. Initialize the GUARDNET Model
    model = GuardNet(
        num_constants=kb.num_constants,
        embedding_dim=config['embedding_dim'],
        predicate_definitions=kb.predicate_definitions,
        device=device
    )
    
    # 3. Initialize the Trainer
    trainer = Trainer(model, kb, config)
    
    # 4. Start the Training Process
    trainer.train()

if __name__ == '__main__':
    main()
