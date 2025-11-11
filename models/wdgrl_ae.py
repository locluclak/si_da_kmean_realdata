import torch
import torch.nn as nn
from typing import List
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Optional
import numpy as np
from tqdm.auto import trange
from torch.utils.data import DataLoader, TensorDataset
import os
import random
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
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
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if i < len(hidden_dims) - 1:  # only add ReLU for intermediate layers
                layers.append(nn.ReLU())
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
    
class Decoder(nn.Module):
    def __init__(
            self, 
            input_dim: int,
            hidden_dims: List[int],
            output_dim: int,
            ):
        
        super().__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim)) 

        self.net = nn.Sequential(*layers)
        self.criterion = nn.MSELoss()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
        



class WDGRL():
    def __init__(
            self,
            input_dim: int,
            encoder_hidden_dims: List[int]=[10], 
            critic_hidden_dims: List[int]=[10, 10],
            use_decoder: bool = False,
            decoder_hidden_dims: List[int] = [10],
            alpha1: float = 1e-4, 
            alpha2: float = 1e-4,
            device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
            reallabel= None,
            seed = None
            ):

        
        self.device = device
        
        if seed is not None:
            set_seed(seed)

        self.encoder = Encoder(input_dim, encoder_hidden_dims).to(self.device)
        self.critic = Critic(encoder_hidden_dims[-1], critic_hidden_dims).to(self.device)
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=alpha2)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=alpha1)

        if use_decoder:
            self.decoderS = Decoder(encoder_hidden_dims[-1], decoder_hidden_dims,output_dim=input_dim).to(self.device)
            self.decoder_optimizerS = torch.optim.Adam(self.decoderS.parameters(), lr= alpha2)
            self.decoderT = Decoder(encoder_hidden_dims[-1], decoder_hidden_dims,output_dim=input_dim).to(self.device)
            self.decoder_optimizerT = torch.optim.Adam(self.decoderT.parameters(), lr= alpha2)
    
        self.reallabel = reallabel

    def check_metric(
            self,
            Xs,
            Xt,
            n_cluster:int =2,
            ):
        if self.reallabel is None:
            return -1
        ns = len(Xs)


        xs_hat = self.extract_feature(Xs.tensors[0].to(self.device))
        xt_hat = self.extract_feature(Xt.tensors[0].to(self.device))
        xs_hat = xs_hat.cpu().numpy()
        xt_hat = xt_hat.cpu().numpy()

        x_comb = np.vstack((xs_hat, xt_hat))

        kmeans = KMeans(n_clusters=n_cluster, random_state=42)
        predicted_labels = kmeans.fit_predict(x_comb)

        ari = adjusted_rand_score(self.reallabel, predicted_labels[ns:])
        sil = silhouette_score(x_comb, predicted_labels)

        kmean2 = KMeans(n_clusters=n_cluster, random_state=42)
        pre_label_onlyT = kmean2.fit_predict(xt_hat)
        ariT = adjusted_rand_score(self.reallabel, pre_label_onlyT)
        silT = silhouette_score(xt_hat, pre_label_onlyT)
        return {
            "ari_comb": ari,
            "silhouette_comb": sil,
            "ari_Tonly": ariT,
            "sil_Tonly": silT,
        }

    def compute_gradient_penalty(
            self, 
            source_data: torch.Tensor, 
            target_data: torch.Tensor
            ) -> torch.Tensor:
        
        alpha = torch.rand(source_data.size(0), 1).to(self.device)

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
            with_decoder: bool = False,
            num_epochs: int = 100, 
            gamma: float = 1.0,
            lambda_: float = 1.0,
            delta: float = 0.5,
            dc_iter: int = 5,
            batch_size: int = 32,
            verbose: bool = False,
            early_stopping: bool = True,
            patience: int = 20,
            min_delta: float = 1e-5,
            check_ari: bool = True,
            ) -> List[float]:
        
        self.encoder.train()
        self.critic.train()
        losses = []
        source_critic_scores = []
        target_critic_scores = []

        reconstruction_loss = []
        
        source_size = len(source_dataset)
        target_size = len(target_dataset)

        best_objective = None
        best_epoch = 0
        wait = 0
        best_state = None
        log_ari = []
        
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
                dc_loss.backward()
                self.critic_optimizer.step()
                with torch.no_grad():
                    total_loss += wasserstein_distance.item()
            
            # Train feature extractor
            self.encoder_optimizer.zero_grad()
            source_features = self.encoder(source_data)
            target_features = self.encoder(target_data)

            dc_source = self.critic(source_features)
            dc_target = self.critic(target_features)

            wasserstein_distance = (dc_source.mean() - dc_target.mean())
            
            if with_decoder:
                reconstruct_source = self.decoderS(source_features)
                decoderS_loss  = self.decoderS.criterion(reconstruct_source, source_data)

                reconstruct_target = self.decoderT(target_features)
                decoderT_loss  = self.decoderT.criterion(reconstruct_target, target_data)

                objective_function = decoderS_loss + delta*decoderT_loss + lambda_*wasserstein_distance

                reconstruction_loss.append(decoderS_loss.detach().cpu().numpy().item())

                objective_function.backward()
                self.decoder_optimizerS.step() 
                self.decoder_optimizerT.step() 
                self.encoder_optimizer.step()
            else:
                objective_function = wasserstein_distance

                objective_function.backward()
                self.encoder_optimizer.step()

            with torch.no_grad():
                loss += objective_function.item()
            if check_ari:
                log_ari.append(self.check_metric(source_dataset, target_dataset, n_cluster=2))
            # Early stopping logic
            current_objective = wasserstein_distance.item()
            if (epoch > 50):# and (objective_function < 3):
                if (best_objective is None or (best_objective - current_objective) > min_delta):
                    best_objective = current_objective
                    best_epoch = epoch
                    wait = 0
                    # Save best model state
                    best_state = {
                        'encoder': self.encoder.state_dict(),
                        'critic': self.critic.state_dict(),
                    }
                    if with_decoder:
                        best_state['decoderS'] = self.decoderS.state_dict()
                        best_state['decoderT'] = self.decoderT.state_dict()
                else:
                    wait += 1
                    if early_stopping and wait >= patience:
                        # if verbose:
                        print(f"Early stopping at epoch {epoch+1}. Best objective: {best_objective} at epoch {best_epoch+1}")
                        # Restore best model state
                        self.encoder.load_state_dict(best_state['encoder'])
                        self.critic.load_state_dict(best_state['critic'])
                        if with_decoder and 'decoderS' in best_state:
                            self.decoderS.load_state_dict(best_state['decoderS'])
                        if with_decoder and 'decoderT' in best_state:
                            self.decoderT.load_state_dict(best_state['decoderT'])
                        break
                
            source_critic_scores.append(self.criticize(source_data))
            target_critic_scores.append(self.criticize(target_data))
            losses.append(loss)#/len(source_data))

            
            if verbose:
                print(f'Epoch {epoch + 1}/{num_epochs} | Loss: {loss/len(source_data)}')

        return {
            "loss": losses,
            "decoder_loss": reconstruction_loss,
            "log_ari": log_ari,
            }
    def save(self, folder_path):
        os.makedirs(folder_path, exist_ok=True)

        torch.save(self.feature_extractor.state_dict(),
                   os.path.join(folder_path, "feature_extractor.pth"))
        torch.save(self.classifier.state_dict(),
                   os.path.join(folder_path, "classifier.pth"))
        torch.save(self.domain_discriminator.state_dict(),
                   os.path.join(folder_path, "domain_discriminator.pth"))

        print(f"✅ WDGRL models saved in folder: {folder_path}")

    def load(self, folder_path):
        self.feature_extractor.load_state_dict(
            torch.load(os.path.join(folder_path, "feature_extractor.pth"), map_location="cpu")
        )
        self.classifier.load_state_dict(
            torch.load(os.path.join(folder_path, "classifier.pth"), map_location="cpu")
        )
        self.domain_discriminator.load_state_dict(
            torch.load(os.path.join(folder_path, "domain_discriminator.pth"), map_location="cpu")
        )

        print(f"🔄 WDGRL models loaded from folder: {folder_path}")
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
    
    @torch.no_grad()
    def mseloss(self, x: torch.Tensor) -> float:
        # self.encoder.eval()
        self.decoder.eval()
        return self.decoder.criterion(self.decod)
