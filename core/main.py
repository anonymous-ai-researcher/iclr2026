# guardnet/main.py

import os
from config import get_config
from utils import set_seed, setup_logger
from dataloader import load_dummy_data
from model import GuardNet
from train import Trainer

def main():
    # 1. Load configuration
    config = get_config()
    
    # 2. Set up logger and random seed
    os.makedirs(config['log_path'], exist_ok=True)
    logger = setup_logger("Main", os.path.join(config['log_path'], "main.log"))
    set_seed(config['seed'])
    logger.info("Configuration loaded and seed set.")
    logger.info(f"Using device: {config['device']}")

    # 3. Load data and knowledge base
    # TODO: Replace with actual data loading logic
    logger.info("Loading knowledge base...")
    kb = load_dummy_data()
    
    # 4. Initialize the model
    logger.info("Initializing GUARDNET model...")
    model = GuardNet(kb, config)
    logger.info(model)
    
    # 5. Initialize the trainer and start training
    trainer = Trainer(model, kb, config)
    trainer.run()
    
    logger.info("Process finished.")

if __name__ == "__main__":
    main()
