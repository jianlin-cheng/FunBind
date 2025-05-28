import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingEncoder(nn.Module):
    def __init__(self, in_features, hidden_dims):
        super(EmbeddingEncoder, self).__init__()
        
        self.in_dropout = nn.Dropout(p=0.1)

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features if i == 0 else hidden_dims[i-1], hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(p=0.2) if i < len(hidden_dims) - 1 else nn.Identity()
            )
            for i, hidden_dim in enumerate(hidden_dims)
        ])
        
        self.skip = nn.Linear(in_features, hidden_dims[-1])
        self.final = nn.Linear(hidden_dims[-1], hidden_dims[-1])
        
    def forward(self, x):
        x = self.in_dropout(x)
        residual = self.skip(x)
        out = x

        for layer in self.layers:
            out = layer(out)
            
        out = out + residual
        out = self.final(out)
        
        return out


class ExpertHead(nn.Module):
    def __init__(self, in_features, expert_configs):
        super().__init__()
        
        self.num_experts = len(expert_configs)

        self.experts = nn.ModuleList([
            EmbeddingEncoder(in_features, hidden_dims) for hidden_dims in expert_configs
        ])

        self.gate = nn.Sequential(
            nn.Linear(in_features, self.num_experts),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        gate_outputs = self.gate(x) 
        expert_outputs = [expert(x) for expert in self.experts]
        stacked_experts = torch.stack(expert_outputs, dim=-1)
        x = torch.sum(stacked_experts * gate_outputs.unsqueeze(1), dim=-1)
        return x, expert_outputs
    
    def get_expert_usage(self, x):
        gate_outputs = self.gate(x)
        stats = {
            'mean_activation': gate_outputs.mean(0),
            'max_activation': gate_outputs.max(0)[0],
            'sparsity': (gate_outputs < 0.01).float().mean(0)
        }
        return stats
        

class SeqBindBase(nn.Module):
    def __init__(self, configs):
        super(SeqBindBase, self).__init__()

        self.modality_encoder = nn.ModuleDict({
            f'{modality}_modality': ExpertHead(_configs['input_dim'], _configs['expert_configs'])
            for modality, _configs in configs.items()
        })
        self.modalities = list(self.modality_encoder.keys())

    def encode_modality(self, modality, value):
        if modality not in self.modality_encoder:
            raise ValueError(f"Modality '{modality}' not found in modality encoders.")
        return self.modality_encoder[modality](value)



class SeqBindPretrain(SeqBindBase):
    def __init__(self, pretrain_config=None):

        super(SeqBindPretrain, self).__init__(pretrain_config)

    def forward(self, inputs):
        encoder_outputs, expert_outputs  = {}, {}
        for modality_key, modality_value in inputs.items():
            if modality_key in self.modalities:
                modality_output, modality_experts = self.encode_modality(modality=modality_key, value=modality_value.squeeze(1))
                encoder_outputs[modality_key] = F.normalize(modality_output, dim=-1)
                expert_outputs[modality_key] = modality_experts
        return encoder_outputs, expert_outputs



class SeqBindClassifier(SeqBindPretrain):
    def __init__(self, pretrain_config, classifier_config):

        super(SeqBindClassifier, self).__init__(pretrain_config=pretrain_config)

        self.classifier =  nn.ModuleDict({
            f'{modality}_modality': ExpertHead(pretrain_config[modality]['output_dim'], _expert_config)
            for modality, _expert_config in classifier_config.items()
        })


        self.sigmoid = nn.Sigmoid()

        
    def forward(self, inputs):

        encoder_outputs, classification_outputs, expert_outputs  = {}, {}, {}

        for modality_key, modality_value in inputs.items():

            if modality_key in self.modalities:
                modality_output, modality_experts = self.encode_modality(modality=modality_key, value=modality_value.squeeze(1))
                encoder_outputs[modality_key] = F.normalize(modality_output, dim=-1)
                modality_output, modality_experts = self.classifier[modality_key](modality_output)
                classification_outputs[modality_key] = self.sigmoid(modality_output)
                expert_outputs[modality_key] = modality_experts
        
        return encoder_outputs, classification_outputs, expert_outputs
