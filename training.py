import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from Loss import ContrastiveLoss
from preprocessing.data_processor import CustomDataset, CustomDataCollator
from model import SeqBind
from torch.utils.data import DataLoader
from peft import get_peft_model, LoraConfig
from utils import count_parameters, print_all_modules, print_trainable_modules



def train_model(model, dataloader, criterion, optimizer, num_epochs=25):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, input_ids, attention_mask in dataloader:
            images, input_ids, attention_mask = images.cuda(), input_ids.cuda(), attention_mask.cuda()
            optimizer.zero_grad()
            image_features, text_features = model(images, input_ids, attention_mask)
            loss = criterion(image_features, text_features)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        epoch_loss = running_loss / len(dataloader.dataset)
        print(f'Epoch {epoch}/{num_epochs - 1}, Loss: {epoch_loss:.4f}')


def train_model(model, ss_dataloader, st_dataloader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()

        running_loss = 0.0
        mlm_ss_loss, mlm_st_loss, ctr_ss_loss, ctr_st_loss = 0.0, 0.0, 0.0, 0.0

        ss_iter = iter(ss_dataloader)
        st_iter = iter(st_dataloader)
        num_batches = max(len(ss_dataloader), len(st_dataloader))

        for _i in range(num_batches):
            optimizer.zero_grad()
            
            # Process Sequence-Structure batch
            try:
                ss_batch = next(ss_iter)
                ss_outputs, _mlm_ss_loss = model(ss_batch)
                _ctr_ss_loss =  criterion(ss_outputs['sequence_modality'], ss_outputs['structure_modality'])

                ss_loss = _mlm_ss_loss + _ctr_ss_loss
                ss_loss.backward()
                optimizer.step()

                running_loss += ss_loss.item() * len(ss_batch)
                mlm_ss_loss += _mlm_ss_loss.item() * len(ss_batch)
                ctr_ss_loss += _ctr_ss_loss.item() * len(ss_batch)
            except StopIteration:
                pass
            
            # Process Sequence-Text batch
            try:
                st_batch = next(st_iter)
                st_outputs, _mlm_st_loss = model(st_batch)
                _ctr_st_loss =  criterion(st_outputs['sequence_modality'], st_outputs['text_modality'])

                st_loss = _mlm_st_loss + _ctr_st_loss
                st_loss.backward()
                optimizer.step()

                running_loss += st_loss.item() * len(st_batch)
                mlm_st_loss += _mlm_st_loss.item() * len(st_batch)
                ctr_st_loss += _ctr_st_loss.item() * len(st_batch)
            except StopIteration:
                pass

            print(_i)

        epoch_loss = running_loss / (len(ss_dataloader.dataset) + len(st_dataloader.dataset))
        epoch_mlm_ss_loss = mlm_ss_loss / len(ss_dataloader.dataset)
        epoch_ctr_ss_loss = ctr_ss_loss / len(ss_dataloader.dataset)
        epoch_mlm_st_loss = mlm_st_loss / len(st_dataloader.dataset)
        epoch_ctr_st_loss = ctr_st_loss / len(st_dataloader.dataset)

        print(f'Epoch {epoch}/{num_epochs - 1}, Total loss: {epoch_loss:.4f}, '
        f'Mask Loss SS: {epoch_mlm_ss_loss:.4f}, Contrastive Loss SS: {epoch_ctr_ss_loss:.4f}, '
        f'Mask Loss ST: {epoch_mlm_st_loss:.4f}, Contrastive Loss ST: {epoch_ctr_st_loss:.4f}')



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Pre-training', add_help=False)
    args = parser.parse_args()

    args.modality_pair = "ss"
    ss_dataset = CustomDataset(args)
    custom_data_collator = CustomDataCollator(matching_modality=args.modality_pair)
    ss_dataloader = DataLoader(ss_dataset, batch_size=100, shuffle=True, collate_fn=custom_data_collator)


    args.modality_pair = "st"
    st_dataset = CustomDataset(args)
    custom_data_collator = CustomDataCollator(matching_modality=args.modality_pair)
    st_dataloader = DataLoader(st_dataset, batch_size=100, shuffle=True, collate_fn=custom_data_collator)


    model = SeqBind().cuda()

    # print_trainable_modules(model)
    
    print(count_parameters(model))

    exit()

    # Initialize LoRA configuration
    peft_config = LoraConfig(  
        r=16,  # Rank
        lora_alpha=16,
        target_modules=["query", "value"],
        lora_dropout=0.1,
        bias="none",
        # modules_to_save=["classifier"],
    )

    # Apply the PEFT configuration to the model
    # model = get_peft_model(model, peft_config)

    # print(count_parameters(model))
    # model.print_trainable_parameters()
    
    # exit()

    criterion = ContrastiveLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Train both datasets together
    train_model(model, ss_dataloader, st_dataloader, criterion, optimizer, num_epochs=100)