import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import torch
import torch.nn as nn
from transformers import AutoModel
import config

class RegiBERT(nn.Module):
    def __init__(self, freeze=True):
        super().__init__()
        self.camembert = AutoModel.from_pretrained(config.MODEL_NAME)
        
        if freeze:
            for param in self.camembert.parameters():
                param.requires_grad = False

        # nouvelle tete lineaire ajoutée par dessus pour la classification en 3 classes (logits)
        self.classifier = nn.Linear(self.camembert.config.hidden_size, 3)
        
        # logsoftmax pour les probas log requises pour KL divergence loss
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def mean_pooling(self, model_output, attention_mask):
        """
        calcule la moyenne des embeddings des tokens en ignorant le padding
        """
        token_embeddings = model_output[0] # forme : [batch_size, seq_len, 768]

        # forme du attention_mask : [batch_size, seq_len]
        # expansion du masque pour correspondre à la taille des embeddings
        input_mask_expanded = (attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()) 
        # forme après expansion : [batch_size, seq_len, 768]

        # somme des embeddings des tokens
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        
        # nombre de vrais tokens par phrase avec un minimum pour éviter la division par zéro
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        # vecteur 768D moyen par phrase
        return sum_embeddings / sum_mask

    def forward(self, input_ids, attention_mask):
        # passage dans CamemBERT
        outputs = self.camembert(input_ids=input_ids, attention_mask=attention_mask)
        
        embeddings = self.mean_pooling(outputs, attention_mask)
        logits = self.classifier(embeddings)
        logprobs = self.log_softmax(logits)
        
        return logprobs, embeddings