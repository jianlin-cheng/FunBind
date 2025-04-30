import torch
from torch.utils.data import Dataset
from data_processing.utils import load_pickle
from Const import BASE_DATA_DIR


class CustomDataset(Dataset):
    def __init__(self, modality_pair, config, data_path=None, data_list=None, validation=False):

        self.modality_1, self.modality_2 = modality_pair.split("_")
        self.config = config
        self.base_path = data_path or BASE_DATA_DIR

        if data_list is not None:
            self.data_list = data_list
            
        else:
            train_valid_data = load_pickle(f"{self.base_path}/data/train_valid")
            
            if validation:
                self.data_list = train_valid_data["Validation"]
            elif self.modality_2 == "Ontology":
                self.data_list = train_valid_data[self.modality_1]
            else:
                self.data_list = train_valid_data[self.modality_2]
            
            self.data_list = list(self.data_list)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample = torch.load(
            f'{self.base_path}/data/dataset/{self.data_list[idx]}.pt',
            weights_only=False
        )

        prot = sample['Protein']
        seq = sample[self.modality_1][self.config[self.modality_1]['model']]
        mod_pair = sample[self.modality_2][self.config[self.modality_2]['model']]
        return (prot, seq, mod_pair)
    


class CustomDataCollator:
    def __init__(self, modality_pair, device):

        self.modality_1 = modality_pair.split("_")[0]
        self.modality_2 = modality_pair.split("_")[1]
        self.device = device

    def _move_to_device(self, input_ids):
        return torch.stack(input_ids, 0).to(self.device)


    def __call__(self, batch):

        input_ids_1, input_ids_2, input_ids_3 = [], [], []
        for item in batch:
            input_ids_1.append(item[0])
            input_ids_2.append(item[1])
            input_ids_3.append(item[2])

        modality_1 = self._move_to_device(input_ids_2)
        modality_2 = self._move_to_device(input_ids_3)

        return  {
                'protein': input_ids_1, 
                '{}_modality'.format(self.modality_1): modality_1,
                '{}_modality'.format(self.modality_2): modality_2
            }
