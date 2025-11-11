import torch
import torch.nn as nn
from typing import List

from typing import Optional
import numpy as np
from tqdm.auto import trange
from torch.utils.data import DataLoader, TensorDataset
import os
import random

def set_seed(seed: int):
    """Sets the seed for reproducibility across multiple libraries."""
    os.environ['PYTHONHASHSEED'] = str(seed)  # Python hash seed
    random.seed(seed)                        # Python's random module
    np.random.seed(seed)                     # NumPy
    torch.manual_seed(seed)                  # PyTorch CPU
    torch.cuda.manual_seed(seed)             # PyTorch CUDA (for a single GPU)
    torch.cuda.manual_seed_all(seed)         # PyTorch CUDA (for all GPUs)
    
    # Configure PyTorch for deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int]):
        """Feature extractor network."""
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim

        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class Critic(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int]):
        """Domain critic network."""
        super().__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))  # Output scalar for Wasserstein distance

        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    

class WDGRL():
    def __init__(
            self,
            input_dim: int=2,
            encoder_hidden_dims: List[int]=[10], 
            critic_hidden_dims: List[int]=[10, 10],
            _lr_encoder: float = 1e-4, 
            _lr_critic: float = 1e-4, 
            device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
            seed = None
            ):


        self.device = device
        
        if seed is not None:
            set_seed(seed)

        self.encoder = Encoder(input_dim, encoder_hidden_dims).to(self.device)
        self.critic = Critic(encoder_hidden_dims[-1], critic_hidden_dims).to(self.device)
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=_lr_encoder)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=_lr_critic)
    

    def compute_gradient_penalty(
            self, 
            source_data: torch.Tensor, 
            target_data: torch.Tensor
            ) -> torch.Tensor:
        
        alpha = torch.rand(source_data.size(0), 1).to(self.device)
        # print(target_data.shape)
        # print(source_data.shape)
        differences = target_data - source_data 
        interpolates = source_data + (alpha * differences)
        
        interpolates = torch.stack([interpolates, source_data, target_data]).requires_grad_()


        preds = self.critic(interpolates)
        gradients = torch.autograd.grad(preds, interpolates,
                     grad_outputs=torch.ones_like(preds),
                     retain_graph=True, create_graph=True)[0]
        
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty = ((gradient_norm - 1)**2).mean()
        return gradient_penalty


    def train(
            self, 
            source_dataset: TensorDataset, 
            target_dataset: TensorDataset, 
            num_epochs: int = 100, 
            gamma: float = 10.0,
            dc_iter: int = 5,
            batch_size: int = 32,
            ) -> List[float]:
        
        self.encoder.train()
        self.critic.train()
        losses = []
        source_critic_scores = []
        target_critic_scores = []


        
        source_size = len(source_dataset)
        target_size = len(target_dataset)

        for epoch in trange(num_epochs, desc='Epoch'):
            loss = 0
            total_loss = 0
            # for source_data, target_data in zip(source_loader, target_loader):
            
            # Randomly sample m from source
            source_indices = torch.randint(0, source_size, (batch_size,))
            source_batch = torch.stack([source_dataset[i][0] for i in source_indices])

            # Randomly sample m from target (with replacement if smaller)
            target_indices = torch.randint(0, target_size, (batch_size,))
            target_batch = torch.stack([target_dataset[i][0] for i in target_indices])

            source_data, target_data = source_batch.to(self.device), target_batch.to(self.device)

            # Train domain critic
            for _ in range(dc_iter):
                self.critic_optimizer.zero_grad()
                
                with torch.no_grad():
                    source_features = self.encoder(source_data).view(source_data.size(0), -1)
                    target_features = self.encoder(target_data).view(target_data.size(0), -1)
                
                # Compute empirical Wasserstein distance
                dc_source = self.critic(source_features)
                dc_target = self.critic(target_features)
                wasserstein_distance = (dc_source.mean() - dc_target.mean())

                # Gradient penalty
                gradient_penalty = self.compute_gradient_penalty(source_features, target_features)

                # Domain critic loss
                dc_loss = - wasserstein_distance + gamma * gradient_penalty
                # print(f'- iteration #{_} / {dc_iter} | source critic: {dc_source.mean().item()} | target critic: {dc_target.mean().item()} | wasserstein distance: {wasserstein_distance.item()} | gradient penalty: {gradient_penalty.item()}')
                dc_loss.backward()
                self.critic_optimizer.step()
                with torch.no_grad():
                    total_loss += wasserstein_distance.item()
            # print('-------------------------------')
            # Train feature extractor
            self.encoder_optimizer.zero_grad()
            source_features = self.encoder(source_data)
            target_features = self.encoder(target_data)
            dc_source = self.critic(source_features)
            dc_target = self.critic(target_features)
            wasserstein_distance = (dc_source.mean() - dc_target.mean())
            wasserstein_distance.backward()
            self.encoder_optimizer.step()

            with torch.no_grad():
                loss += wasserstein_distance.item()
                
                    
            # source_critic_scores.append(self.criticize(source_loader.dataset.tensors[0].to(self.device)))
            # target_critic_scores.append(self.criticize(target_loader.dataset.tensors[0].to(self.device)))
            # losses.append(loss/len(source_loader))
            # print(f'Epoch {epoch + 1}/{num_epochs} | Loss: {loss/len(source_loader)}')
            # print('--------------------------------')
        return losses, source_critic_scores, target_critic_scores

    @torch.no_grad()
    def extract_feature(
        self, 
        x: torch.Tensor
        ) -> torch.Tensor:
        
        self.encoder.eval()
        return self.encoder(x)
    
    @torch.no_grad()
    def criticize(self, x: torch.Tensor) -> float:
        self.encoder.eval()
        self.critic.eval()
        return self.critic(self.encoder(x)).mean().item()