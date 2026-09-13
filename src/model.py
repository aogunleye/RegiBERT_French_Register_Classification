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

        self.classifier = nn.Linear(self.camembert.config.hidden_size, 3)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = (attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()) 
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        return sum_embeddings / sum_mask

    def forward(self, input_ids, attention_mask):
        outputs = self.camembert(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = self.mean_pooling(outputs, attention_mask)
        logits = self.classifier(embeddings)
        logprobs = self.log_softmax(logits)
        
        return logprobs, embeddings