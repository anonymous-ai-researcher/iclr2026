# trainer.py

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import random

from guardnet_model import GuardNet
from formula_parser import FormulaParser

class Trainer:
    """
    Encapsulates the training logic for the GUARDNET model, implementing
    the Hybrid Domain Strategy described in Section 4.2.
    """
    def __init__(self, model: GuardNet, kb, config: dict):
        self.model = model
        self.kb = kb
        self.config = config
        self.device = model.device
        
        self.parser = FormulaParser(model, tau=config['tau'])
        
        # [cite_start]The paper uses the AdamW optimizer[cite: 647, 1088].
        self.optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['wd'])
        # [cite_start]The scheduler reduces LR on plateau, monitoring validation MRR in the paper[cite: 1089].
        # Here we monitor the training loss for simplicity.
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=5, factor=0.5)
        
        self.full_domain_tensor = torch.tensor(self.kb.constants, dtype=torch.long, device=self.device)

    def calculate_loss(self, axiom_batch: list, lambda_fidelity: float) -> torch.Tensor:
        """
        [cite_start]Calculates the total loss by combining fidelity and generalization losses[cite: 503].
        """
        batch_size = len(axiom_batch)
        
        # [cite_start]--- Fidelity Loss on Core Domain [cite: 564] ---
        # [cite_start]The Core Domain ensures the model is faithful to known entities[cite: 434].
        core_constants = torch.tensor(random.choices(self.kb.core_domain, k=batch_size), device=self.device)
        core_embeddings = self.model.constant_embeddings(core_constants)
        
        fidelity_satisfactions = []
        for i, axiom in enumerate(axiom_batch):
            initial_bindings = {'c': core_embeddings[i].unsqueeze(0)}
            satisfaction = self.model(axiom, self.parser, initial_bindings, self.full_domain_tensor)
            fidelity_satisfactions.append(satisfaction)
        
        avg_fidelity_sat = torch.mean(torch.stack(fidelity_satisfactions))
        [cite_start]fidelity_loss = 1.0 - avg_fidelity_sat  # Dissatisfaction is 1 - satisfaction[cite: 505].

        # [cite_start]--- Generalization Loss on Latent Domain [cite: 566] ---
        # [cite_start]The Latent Domain drives generalization to unseen entities[cite: 441].
        # We approximate this by sampling from the entire set of constants.
        latent_constants = torch.randint(0, self.kb.num_constants, (batch_size,), device=self.device)
        latent_embeddings = self.model.constant_embeddings(latent_constants)
        
        generalization_satisfactions = []
        for i, axiom in enumerate(axiom_batch):
            initial_bindings = {'c': latent_embeddings[i].unsqueeze(0)}
            satisfaction = self.model(axiom, self.parser, initial_bindings, self.full_domain_tensor)
            generalization_satisfactions.append(satisfaction)
            
        avg_generalization_sat = torch.mean(torch.stack(generalization_satisfactions))
        generalization_loss = 1.0 - avg_generalization_sat
        
        # [cite_start]The final training objective is a weighted combination[cite: 503].
        total_loss = (lambda_fidelity * fidelity_loss + 
                      (1 - lambda_fidelity) * generalization_loss)
        return total_loss

    def train(self):
        """
        The main training loop.
        """
        print("Starting GUARDNET training...")
        for epoch in range(self.config['epochs']):
            self.model.train()
            
            # [cite_start]Dynamic curriculum for lambda, linearly annealing from 0.9 to 0.4[cite: 1094].
            lambda_fidelity = np.linspace(0.9, 0.4, self.config['epochs'])[epoch]
            
            epoch_loss = 0.0
            
            axiom_loader = DataLoader(self.kb.axioms, batch_size=self.config['batch_size'], shuffle=True)
            
            progress_bar = tqdm(axiom_loader, desc=f"Epoch {epoch+1}/{self.config['epochs']}", leave=False)
            for axiom_batch in progress_bar:
                self.optimizer.zero_grad()
                loss = self.calculate_loss(axiom_batch, lambda_fidelity)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                progress_bar.set_postfix(loss=f"{loss.item():.4f}", lambda_f=f"{lambda_fidelity:.2f}")

            avg_epoch_loss = epoch_loss / len(axiom_loader)
            self.scheduler.step(avg_epoch_loss)
            
            print(f"Epoch {epoch+1}/{self.config['epochs']} | Average Loss: {avg_epoch_loss:.4f} | LR: {self.optimizer.param_groups[0]['lr']:.1e}")
        
        print("Training complete.")
