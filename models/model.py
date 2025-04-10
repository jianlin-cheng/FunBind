import torch
import torch.nn as nn
import torch.nn.functional as F


expert_configs = {
    "MF": {
        'Sequence': [(3072, 1600, 1529), (2048, 1600, 1529), (1600, 1529)],
        'Structure': [(1280, 1400, 1529), (1056, 1400, 1529), (1280, 1529)],
        'Text': [(2048, 1600, 1529), (3072, 1600, 1529), (1600, 1529)],
        'Interpro': [(2048, 1600, 1529), (3072, 1600, 1529), (1600, 1529)]
    },
    "CC": {
        'Sequence': [(2048, 1280, 1043), (1440, 1280, 1043), (1280, 1043)],
        'Structure': [(512, 768, 1043), (512, 768, 1043), (768, 1043)],
        'Text': [(2048, 1280, 1043), (1440, 1280, 1043), (1280, 1043)],
        'Interpro': [(2048, 1280, 1043), (1440, 1280, 1043), (1280, 1043)]
    },
    "BP": {
        'Sequence': [(3072, 1664, 1631), (2560, 1664, 1631), (1664, 1631)],
        'Structure': [(1280, 1600, 1631), (1536, 1600, 1631), (1600, 1631)],
        'Text': [(2560, 1664, 1631), (2048, 1664, 1631), (1664, 1631)],
        'Interpro': [(2560, 1664, 1631), (2048, 1664, 1631), (1664, 1631)]
    }
}


expert_configs = {
    "MF": {
        'Sequence': [(1440, 1504, 1529),],
        'Structure':[(1440, 1504, 1529),],
        'Text':     [(1440, 1504, 1529),],
        'Interpro': [(1440, 1504, 1529),]
    },
    "CC": {
        'Sequence':  [(1120, 1056, 1043)],
        'Structure': [(1120, 1056, 1043)],
        'Text':      [(1120, 1056, 1043)],
        'Interpro':  [(1120, 1056, 1043)]
    },
    "BP": {
        'Sequence':  [(1440, 1568, 1631)],
        'Structure': [(1440, 1568, 1631)],
        'Text':      [(1440, 1568, 1631)],
        'Interpro':  [(1440, 1568, 1631)]
    }
}



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
    def __init__(self, config, expert_configs=None):
        super(SeqBindBase, self).__init__()

        self.modality_encoder = nn.ModuleDict({
            f'{modality}_modality': ExpertHead(config[modality]['input_dim'], expert_configs)
            for modality, expert_configs in expert_configs.items()
        })
        self.modalities = list(self.modality_encoder.keys())

    def encode_modality(self, modality, value):
        if modality not in self.modality_encoder:
            raise ValueError(f"Modality '{modality}' not found in modality encoders.")
        return self.modality_encoder[modality](value)



class SeqBindPretrain(SeqBindBase):
    def __init__(self, config, expert_configs=None):
        expert_configs = {
            'Sequence': [(3072, 1440, 1280), (1600, 1440, 1280), (1440, 1280)],
            #'Sequence': [(1088, 1152, 1280), (1600, 1248, 1280), (1024, 1280)],
            'Structure': [(1088, 1152, 1280), (1600, 1248, 1280), (1024, 1280)],
            'Text': [(2048, 1440, 1280), (1536, 1440, 1280), (1440, 1280)],
            #'Text': [(1088, 1152, 1280), (1600, 1248, 1280), (1024, 1280)],
            'Interpro': [(2048, 1440, 1280), (1536, 1440, 1280), (1440, 1280)],
            #'Interpro': [(1088, 1152, 1280), (1600, 1248, 1280), (1024, 1280)],
            'Ontology': [(2048, 1440, 1280), (1536, 1440, 1280), (1440, 1280)]
        }

        super(SeqBindPretrain, self).__init__(config, expert_configs)

    def forward(self, inputs):
        encoder_outputs, expert_outputs  = {}, {}
        for modality_key, modality_value in inputs.items():
            if modality_key in self.modalities:
                modality_output, modality_experts = self.encode_modality(modality=modality_key, value=modality_value.squeeze(1))
                encoder_outputs[modality_key] = F.normalize(modality_output, dim=-1)
                expert_outputs[modality_key] = modality_experts
        return encoder_outputs, expert_outputs



class SeqBindClassifier(SeqBindPretrain):
    def __init__(self, config, go_ontology=None, pretrained_model_path=None):

        expert_config = expert_configs[go_ontology]
        super(SeqBindClassifier, self).__init__(config)

        self.classifier =  nn.ModuleDict({
            f'{modality}_modality': ExpertHead(1280, _expert_config)
            for modality, _expert_config in expert_config.items()
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
