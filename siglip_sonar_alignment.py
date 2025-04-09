

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import AutoModel, AutoProcessor
from datasets import load_dataset
import numpy as np
from PIL import Image
import io
import logging
import os
from typing import Optional, Dict, Any, Tuple, List
import wandb
from datetime import datetime



class SIGLIPSOLCMAligner(nn.Module):
    def __init__(self, 
                 visual_dim=768,        
                 sonar_dim=1024,        
                 projection_dim=1024,   
                 hidden_dim=1536,       
                 num_heads=8,          
                 dropout=0.1,          
                 num_layers=2,         
                 temperature=0.07):    
        
        super().__init__()
        
        self.visual_projector = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, projection_dim),
            nn.LayerNorm(projection_dim)
        )
        
        self.refinement_blocks = nn.ModuleList([
            RefinementBlock(
                dim=projection_dim,
                sonar_dim=sonar_dim,
                num_heads=num_heads,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        self.final_projection = nn.Sequential(
            nn.Linear(projection_dim, projection_dim),
            nn.LayerNorm(projection_dim)
        )
        
        self.temperature = nn.Parameter(torch.tensor(temperature))
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights with small values for stable training"""
        if isinstance(module, nn.Linear):
            torch.nn.init.kaiming_normal_(module.weight, a=0.1, mode='fan_in', nonlinearity='leaky_relu')
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(self, 
                visual_embeddings: torch.Tensor, 
                sonar_embeddings: torch.Tensor):
       
        if sonar_embeddings.dim() > 2:
            sonar_embeddings = sonar_embeddings.squeeze(1)
        
        visual_features = self.visual_projector(visual_embeddings)
        
        for block in self.refinement_blocks:
            visual_features = block(visual_features, sonar_embeddings)
        
        projected_visual = self.final_projection(visual_features)
        
        visual_norm = nn.functional.normalize(projected_visual, p=2, dim=1)
        sonar_norm = nn.functional.normalize(sonar_embeddings, p=2, dim=1)
        
        return projected_visual, (visual_norm, sonar_norm)
    
    def encode_image(self, visual_embeddings):
        """
        Encode only an image to SONAR space for inference
        """
        visual_features = self.visual_projector(visual_embeddings)
        

        for block in self.refinement_blocks:
            visual_features = block.forward_image_only(visual_features)
        
        projected_visual = self.final_projection(visual_features)
        
        return nn.functional.normalize(projected_visual, p=2, dim=1)
    
    def encode_text(self, sonar_embeddings):
        """
        Process SONAR text embeddings for similarity matching
        """
        if sonar_embeddings.dim() > 2:
            sonar_embeddings = sonar_embeddings.squeeze(1)
        
        return nn.functional.normalize(sonar_embeddings, p=2, dim=1)
    
    def similarity(self, visual_embeddings, sonar_embeddings):
        """
        Compute similarity between visual and SONAR embeddings
        """
        visual_norm = self.encode_image(visual_embeddings)
        sonar_norm = self.encode_text(sonar_embeddings)
        
        similarity = torch.matmul(visual_norm, sonar_norm.t()) / self.temperature
        
        return similarity


class RefinementBlock(nn.Module):
    """
    Cross-modal attention and self-attention
    for alignment 
    """
    def __init__(self, dim, sonar_dim, num_heads=8, dropout=0.1):
        super().__init__()
        
        self.visual_query = nn.Linear(dim, dim)
        self.sonar_key = nn.Linear(sonar_dim, dim)
        self.sonar_value = nn.Linear(sonar_dim, dim)
        
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm1 = nn.LayerNorm(dim)
        
        self.self_attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm2 = nn.LayerNorm(dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        
        self.norm3 = nn.LayerNorm(dim)
        
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
    
    def forward(self, visual_features, sonar_embeddings):
        visual_query = self.visual_query(visual_features).unsqueeze(1)
        sonar_key = self.sonar_key(sonar_embeddings).unsqueeze(1)
        sonar_value = self.sonar_value(sonar_embeddings).unsqueeze(1)
        
        cross_attn_output, _ = self.cross_attention(
            query=visual_query,
            key=sonar_key,
            value=sonar_value
        )
        cross_attn_output = cross_attn_output.squeeze(1)
        visual_enhanced = self.norm1(visual_features + cross_attn_output)
        
        visual_seq = visual_enhanced.unsqueeze(1)
        self_attn_output, _ = self.self_attention(
            query=visual_seq,
            key=visual_seq,
            value=visual_seq
        )
        self_attn_output = self_attn_output.squeeze(1)
        
        visual_refined = self.norm2(visual_enhanced + self_attn_output)
        
        ffn_output = self.ffn(visual_refined)
        
        visual_output = self.norm3(visual_refined + ffn_output)
        gate_input = torch.cat([visual_features, visual_output], dim=1)
        gate_value = self.gate(gate_input)
        gated_output = gate_value * visual_output + (1 - gate_value) * visual_features
        
        return gated_output
    
    def forward_image_only(self, visual_features):
        visual_seq = visual_features.unsqueeze(1)
        self_attn_output, _ = self.self_attention(
            query=visual_seq,
            key=visual_seq,
            value=visual_seq
        )
        self_attn_output = self_attn_output.squeeze(1)
        
        visual_refined = self.norm2(visual_features + self_attn_output)
        
        ffn_output = self.ffn(visual_refined)
        visual_output = self.norm3(visual_refined + ffn_output)
        
        return visual_output


class StreamingCC12MDataset(torch.utils.data.IterableDataset):
    def __init__(
        self, 
        dataset,
        siglip_model_name, 
        siglip_processor_name,  
        sonar_text_encoder_name,  
        max_samples=None,
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
   
        self.dataset = dataset
        self.siglip_model_name = siglip_model_name
        self.siglip_processor_name = siglip_processor_name
        self.sonar_text_encoder_name = sonar_text_encoder_name
        self.max_samples = max_samples
        self.device = device
    
    def __iter__(self):

        from transformers import SiglipVisionModel, SiglipImageProcessor, SiglipModel
        from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
        
        
        device = torch.device(self.device)

        siglip_model = SiglipModel.from_pretrained(
            self.siglip_model_name, 
            device_map="auto"
        ).eval()
        
        siglip_processor = SiglipImageProcessor.from_pretrained(
            self.siglip_processor_name
        )
        
        sonar_text_encoder = TextToEmbeddingModelPipeline(
            encoder="text_sonar_basic_encoder",
            tokenizer="text_sonar_basic_encoder",
            device = device,
        )
        
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            dataset_iter = iter(self.dataset)
        else:
            dataset_iter = iter(self.dataset)
        
        sample_count = 0
        
        for item in dataset_iter:
            if self.max_samples and sample_count >= self.max_samples:
                break
            
            try:
                if isinstance(item['jpg'], (bytes, bytearray)):
                    image_bytes = item['jpg']
                else:
                    buffer = io.BytesIO()
                    item['jpg'].save(buffer, format="JPEG")
                    image_bytes = buffer.getvalue()
                
                image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                
                inputs = siglip_processor(
                    images=[image], 
                    return_tensors="pt"
                ).to(self.device)
                
                with torch.no_grad():
                    visual_embedding = siglip_model.get_image_features(**inputs)
                
                text = item['txt'].decode('utf-8') if isinstance(item['txt'], bytes) else item['txt']
                
                sonar_embedding = sonar_text_encoder.predict([text], source_lang="eng_Latn")

                if isinstance(sonar_embedding, torch.Tensor):
                    sonar_tensor = sonar_embedding.clone().detach().cpu()
                else:
                    sonar_tensor = torch.tensor(sonar_embedding, device='cpu')
                
                yield (
                    visual_embedding.cpu().squeeze(), 
                    sonar_tensor,
                )
                
                sample_count += 1
            
            except Exception as e:
                logging.error(f"Error processing item: {e}")
                continue


def sigmoid_contrastive_loss(similarity_matrix, temperature=1.0):
    labels = torch.eye(similarity_matrix.shape[0], device=similarity_matrix.device)
    
    logits = similarity_matrix / temperature
    
    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
    
    return loss


class SIGLIPSOLCMTrainer:
    def __init__(self, 
                 model, 
                 train_loader, 
                 val_loader=None,
                 learning_rate=1e-4,
                 weight_decay=0.01,
                 device="cuda" if torch.cuda.is_available() else "cpu",
                 config = None):
      
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=100
        )
        self.use_wandb = config.get('use_wandb', False)
        if self.use_wandb and wandb.run is None:
            wandb.init(
                project=config.get('wandb_project', 'siglip-sonar-alignment'),
                name=config.get('wandb_run_name', f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"),
                config=config
            )
            wandb.watch(self.model)


    
    def train_epoch(self, epoch, num_epochs):
        self.model.train()
        total_loss = 0.0
        batch_count = 0
        
        from tqdm import tqdm
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for visual_emb, sonar_emb in progress_bar:
            visual_emb = visual_emb.to(self.device)
            sonar_emb = sonar_emb.to(self.device)
            
            self.optimizer.zero_grad()
            
            _, (visual_norm, sonar_norm) = self.model(visual_emb, sonar_emb)
            
            similarity_matrix = torch.matmul(visual_norm, sonar_norm.t())
            loss = sigmoid_contrastive_loss(similarity_matrix, self.model.temperature)
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            batch_count += 1
            total_loss += loss.item()

            progress_bar.set_postfix({"loss": loss.item()})
            
            if self.use_wandb:
                wandb.log({
                    "batch_loss": loss.item(),
                    "temperature": self.model.temperature.item(),
                    "learning_rate": self.optimizer.param_groups[0]['lr']
                })
        
        self.scheduler.step()
        
        avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
        
        return avg_loss
    
        def evaluate(self):
            if self.val_loader is None:
                return None
                
            self.model.eval()
            total_loss = 0.0
            batch_count = 0
            
            all_similarities = []
            
            with torch.no_grad():
                for visual_emb, sonar_emb in self.val_loader:
                    visual_emb = visual_emb.to(self.device)
                    sonar_emb = sonar_emb.to(self.device)
                    
                    _, (visual_norm, sonar_norm) = self.model(visual_emb, sonar_emb)
                    
                    similarity_matrix = torch.matmul(visual_norm, sonar_norm.t())
                    
                    pair_similarities = torch.diagonal(similarity_matrix).cpu().numpy()
                    all_similarities.extend(pair_similarities)
                    
                    loss = sigmoid_contrastive_loss(similarity_matrix, self.model.temperature)
                    
                    batch_count += 1
                    total_loss += loss.item()
            
            avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
            
            if self.use_wandb:
                similarity_hist = wandb.Histogram(all_similarities)
                wandb.log({
                    "val_loss": avg_loss,
                    "val_similarity_histogram": similarity_hist,
                    "val_mean_similarity": np.mean(all_similarities),
                    "val_median_similarity": np.median(all_similarities)
                })
            
            return avg_loss
        
    def train(self, num_epochs=50, log_interval=10, save_path=None):
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(epoch, num_epochs)
            
            val_loss = self.evaluate() if self.val_loader else None

            if self.use_wandb:
                log_dict = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                }
                if val_loss is not None:
                    log_dict["val_loss"] = val_loss
                wandb.log(log_dict)

            if epoch % log_interval == 0 or epoch == num_epochs - 1:
                val_msg = f", Val Loss: {val_loss:.4f}" if val_loss is not None else ""
                print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}{val_msg}")
            
            if save_path:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'projection_dim': self.model.visual_projector[0].out_features
                }, save_path)
                print(f"Saved best model checkpoint to {save_path}")
                if self.use_wandb:
                    artifact = wandb.Artifact(
                        name=f"model-{wandb.run.id}", 
                        type="model",
                        description=f"Model checkpoint at epoch {epoch+1}"
                    )
                    artifact.add_file(save_path)
                    wandb.log_artifact(artifact)
        if self.use_wandb:
            wandb.finish()



def load_cc12m_dataset(streaming=True):
    dataset = load_dataset("pixparse/cc12m-wds", streaming=streaming)
    return dataset


def create_data_loaders(dataset, siglip_model_name, siglip_processor_name, sonar_text_encoder_name, config):
    if isinstance(dataset, dict) or hasattr(dataset, 'keys'):
        train_dataset = dataset["train"]
        val_dataset = dataset.get("validation", None)
    else:
        train_test_split = dataset.train_test_split(test_size=0.05)
        train_dataset = train_test_split["train"]
        val_dataset = train_test_split["test"]
    
    train_streaming_dataset = StreamingCC12MDataset(
        train_dataset,
        siglip_model_name=siglip_model_name,
        siglip_processor_name=siglip_processor_name,
        sonar_text_encoder_name=sonar_text_encoder_name,
        max_samples=config.get('max_train_samples')
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_streaming_dataset,
        batch_size=config.get('batch_size', 64),
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )
    
    val_loader = None
    if val_dataset is not None:
        val_streaming_dataset = StreamingCC12MDataset(
            val_dataset,
            siglip_model_name=siglip_model_name,
            siglip_processor_name=siglip_processor_name,
            sonar_text_encoder_name=sonar_text_encoder_name,
            max_samples=config.get('max_val_samples')
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_streaming_dataset,
            batch_size=config.get('batch_size', 64),
            num_workers=config.get('num_workers', 4),
            pin_memory=True
        )
    
    return train_loader, val_loader

def main():


    config = {
        'siglip_model_name': "google/siglip2-so400m-patch16-224",
        'sonar_model_name': "facebook/sonar-base",
        'visual_dim': 768,  # SigLIP2 vision dimension
        'sonar_dim': 1024,
        'projection_dim': 1024,
        'batch_size': 512,
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'num_epochs': 10,
        #'max_train_samples': 1000,  # Set to a small number for testing, None for full dataset
        'max_val_samples': 100,
        'num_workers': 4,
        'save_path': './checkpoints/siglip_sonar_alignment.pth',
        'log_interval': 1,
        'use_wandb': True,
        'wandb_project': 'siglip-sonar-alignment',
        'wandb_run_name': f"siglip-sonar-{datetime.now().strftime('%Y%m%d-%H%M%S')}",

    }
    
    os.makedirs(os.path.dirname(config['save_path']), exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    if config['use_wandb']:
        wandb.login(key='ff0b1ce4ff7d189b7ca08c950241de290ecff918')
        wandb.init(
            project=config['wandb_project'],
            name=config['wandb_run_name'],
            config=config,
        )


    
    logging.info(f"Loading SigLIP2 model: {config['siglip_model_name']}")
    from transformers import SiglipVisionModel, SiglipImageProcessor

  
    siglip_model_name = "google/siglip2-base-patch16-224"
    siglip_processor_name = "google/siglip2-base-patch16-224"
    sonar_text_encoder_name = "text_sonar_basic_encoder"
    
    logging.info("Loading CC12M dataset")
    dataset = load_cc12m_dataset(streaming=True)
    
    logging.info("Creating data loaders")
    train_loader, val_loader = create_data_loaders(
        dataset,
        siglip_model_name,
        siglip_processor_name,
        sonar_text_encoder_name,
        config
    )
    
    logging.info("Initializing alignment model")
    alignment_model = SIGLIPSOLCMAligner(
        visual_dim=config['visual_dim'],
        sonar_dim=config['sonar_dim'],
        projection_dim=config['projection_dim']
    )
    
    logging.info("Creating trainer")
    trainer = SIGLIPSOLCMTrainer(
        model=alignment_model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        config = config,
    )
    
    logging.info("Starting training")
    trainer.train(
        num_epochs=config['num_epochs'],
        log_interval=config['log_interval'],
        save_path=config['save_path']
    )
    
    logging.info("Training complete")
    
    torch.save({
        'model_state_dict': alignment_model.state_dict(),
        'projection_dim': alignment_model.visual_projector[0].out_features
    }, config['save_path'].replace('.pth', '_final.pth'))

    if config['use_wandb']:
        artifact = wandb.Artifact(
            name=f"final-model-{wandb.run.id}", 
            type="model",
            description="Final model checkpoint"
        )
        artifact.add_file(final_model_path)
        wandb.log_artifact(artifact)
        
        wandb.finish()


    logging.info(f"Final model saved to {config['save_path'].replace('.pth', '_final.pth')}")


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    main()