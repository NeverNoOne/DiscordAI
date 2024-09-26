import torch
import torch.nn as nn
from torch.utils.data  import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
import model as m

MODEL_PATH = "M:/AI_Models/model_full.pt"
#DATASET_PATH = "C:/Users/maxhe/Downloads/dialog/dialog.csv"
DATASET_PATH = "M:/AI_DataSets/cornell_movie_dialog/corpus.csv"
VOCAB_PATH = "M:/AI_Vocab/vocab_full.json"

class ChatDataset(Dataset):
    def __init__(self, input_sequences, target_sequences):
        self.input_sequences = input_sequences
        self.target_sequences = target_sequences

    def __len__(self):
        return len(self.input_sequences)
    
    def __getitem__(self,idx):
        input_seq = self.input_sequences[idx]
        target_seq = self.target_sequences[idx]
        
        # Return the input and target as a dictionary or tuple
        return {
            'input': torch.tensor(input_seq, dtype=torch.long),
            'target': torch.tensor(target_seq, dtype=torch.long)
        }


def collate_fn(batch):
    input_batch = [item['input'] for item in batch]
    target_batch = [item['target'] for item in batch]
    
    # Pad sequences to the maximum length in the batch
    input_batch_padded = pad_sequence(input_batch, batch_first=True, padding_value=0)  # Assume 0 is your PAD token
    target_batch_padded = pad_sequence(target_batch, batch_first=True, padding_value=0)
    
    return {
        'input': input_batch_padded,
        'target': target_batch_padded
    }
from sequences import GetSequences, GetSequences_max, GetSequences_random

# Assuming input_sequences and target_sequences are lists of tokenized sequences
input_sequences = [...]  # List of tokenized input sequences
target_sequences = [...]  # List of tokenized target sequences
input_sequences, target_sequences, vocab_size, word_index = GetSequences_random(DATASET_PATH, 10000)

#input_sequences, target_sequences, vocab_size, word_index = GetSequences_random(DATASET_PATH, 5000)

import os
import json
#loading vocab if exists
vocab = []
if os.path.exists(VOCAB_PATH):
    with open(VOCAB_PATH, 'r') as f:
        vocab = json.load(f)
    vocab = word_index | vocab
else:
    vocab = word_index
vocab_size = len(vocab) + 1
with open(VOCAB_PATH, 'w') as f:
    json.dump(vocab, f)

# Create the dataset
chat_dataset = ChatDataset(input_sequences, target_sequences)

# Create the DataLoader
train_loader = DataLoader(
    dataset=chat_dataset,
    batch_size=16,        # Set the batch size
    shuffle=True,         # Shuffle the dataset
    collate_fn=collate_fn # Handle padding within each batch,
)

#vocab_size = 10000
model, device = m.getModel(vocab_size=vocab_size, embed_size=150, hidden_size=256, num_layers=2)
print(f"Model loaded with device {device}")

criterion = nn.CrossEntropyLoss()  #(ignore_index=pad_token)  # Assuming you have a padding token
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001) # type: ignore

num_epochs = 10

import time 
from torch import autocast, GradScaler
scaler = GradScaler()
#TODO how to handle the vocab (extending)
#TODO optimizing for large datasets
for epoch in range(num_epochs):
    model.train()
    bcount = 0
    total_loss = 0
    begin = time.time()
    for batch in train_loader:
        bcount += 1
        input_seq = batch['input'].to(device)
        target_seq = batch['target'].to(device)
        
        with autocast(device.type):
            # Forward pass and loss calculation
            output = model(input_seq, target_seq[:, :-1])  # Exclude <EOS> token from target input
            loss = criterion(output.view(-1, vocab_size), target_seq[:, 1:].reshape(-1))  # Target starts from the next token
        
        # Backward pass and optimization
        #optimizer.zero_grad()
        #loss.backward()
        #optimizer.step()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        total_loss += loss.item()
        if bcount % 10 == 0: print(f"finished batch: {bcount} took: {time.time()-begin}"); begin = time.time()
    #if epoch % 5 == 0:
    print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {total_loss / len(train_loader):.4f}')
    

torch.save(model.state_dict(), MODEL_PATH)
#from keras_preprocessing.text import Tokenizer
#input_text = "How are you?"
#tokenizer = Tokenizer(filters='', lower=True, oov_token="<OOV>")  # Keeps punctuation for better meaning
#tokenizer.fit_on_texts(vocab)
#word_index is empty???
#output = model.predict(input_text, tokenizer, vocab_size, "<START>", "<END>")
#print(f"Bot's response: {output}")