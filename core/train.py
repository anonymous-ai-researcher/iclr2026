# guardnet/train.py

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
import os

from model import GuardNet
from dataloader import create_dataloader, KnowledgeBase
from utils import setup_logger

class Trainer:
    def __init__(self, model: GuardNet, kb: KnowledgeBase, config: dict):
        self.model = model
        self.kb = kb
        self.config = config
        self.logger = setup_logger("GuardNetTrainer", os.path.join(config['log_path'], "training.log"))
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='max', # We want to maximize satisfaction
            factor=config['lr_scheduler_factor'],
            patience=config['lr_scheduler_patience']
        )
        self.device = config['device']
        self.model.to(self.device)

    def train_epoch(self, dataloader: torch.utils.data.DataLoader, current_lambda: float):
        self.model.train()
        total_loss = 0.0
        
        for batch in tqdm(dataloader, desc="Training Epoch"):
            self.optimizer.zero_grad()
            
            # In this simplified version, we do not differentiate between fidelity and generalization loss.
            # A complete implementation would need to sample constants according to the hybrid domain strategy.
            
            satisfactions = self.model(batch)
            
            # The loss is the unsatisfaction (1 - satisfaction)
            loss = torch.mean(1.0 - satisfactions)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(dataloader)

    def evaluate(self, dataloader: torch.utils.data.DataLoader) -> float:
        self.model.eval()
        total_satisfaction = 0.0
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                satisfactions = self.model(batch)
                total_satisfaction += satisfactions.mean().item()
        
        avg_satisfaction = total_satisfaction / len(dataloader)
        # In a real KBC task, MRR and Hits@K would be calculated here.
        # Here, we use average satisfaction as the performance metric.
        return avg_satisfaction

    def run(self):
        self.logger.info("Starting training...")
        best_metric = -1.0
        epochs_no_improve = 0
        
        # Create data loaders
        train_loader = create_dataloader(self.kb, self.config['batch_size'], shuffle=True)
        # Ideally, there should be a separate validation set
        val_loader = create_dataloader(self.kb, self.config['batch_size'], shuffle=False)
        
        for epoch in range(self.config['num_epochs']):
            # Update lambda according to the curriculum learning schedule
            progress = epoch / self.config['num_epochs']
            current_lambda = self.config['lambda_start'] * (1 - progress) + self.config['lambda_end'] * progress

            avg_loss = self.train_epoch(train_loader, current_lambda)
            val_metric = self.evaluate(val_loader)
            
            self.logger.info(
                f"Epoch {epoch+1}/{self.config['num_epochs']} | "
                f"Loss: {avg_loss:.4f} | "
                f"Validation Metric (Satisfaction): {val_metric:.4f} | "
                f"Lambda: {current_lambda:.4f}"
            )
            
            self.scheduler.step(val_metric)

            if val_metric > best_metric:
                best_metric = val_metric
                epochs_no_improve = 0
                self.save_checkpoint('best_model.pth')
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= self.config['early_stopping_patience']:
                self.logger.info(f"Early stopping triggered after {epoch+1} epochs.")
                break
                
        self.logger.info("Training finished.")

    def save_checkpoint(self, filename: str):
        path = os.path.join(self.config['checkpoint_path'], filename)
        os.makedirs(self.config['checkpoint_path'], exist_ok=True)
        torch.save(self.model.state_dict(), path)
        self.logger.info(f"Checkpoint saved to {path}")
