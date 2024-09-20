import torch
import torch.nn as nn
from torch.utils.data  import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import model as m

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
from sequences import GetSequences

# Assuming input_sequences and target_sequences are lists of tokenized sequences
input_sequences = [...]  # List of tokenized input sequences
target_sequences = [...]  # List of tokenized target sequences
input_sequences, target_sequences, vocab_size, word_index = GetSequences()

# Create the dataset
chat_dataset = ChatDataset(input_sequences, target_sequences)

# Create the DataLoader
train_loader = DataLoader(
    dataset=chat_dataset,
    batch_size=32,        # Set the batch size
    shuffle=True,         # Shuffle the dataset
    collate_fn=collate_fn # Handle padding within each batch
)

vocab_size = 10000
model, device = m.getModel(vocab_size=vocab_size)
print(f"Model loaded with device {device}")

criterion = nn.CrossEntropyLoss()  #(ignore_index=pad_token)  # Assuming you have a padding token
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001) # type: ignore

num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    
    for batch in train_loader:
        input_seq = batch['input'].to(device)
        target_seq = batch['target'].to(device)
        
        # Forward pass and loss calculation
        output = model(input_seq, target_seq[:, :-1])  # Exclude <EOS> token from target input
        loss = criterion(output.view(-1, vocab_size), target_seq[:, 1:].reshape(-1))  # Target starts from the next token
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

torch.save(model.state_dict(), "M:/AI_Models/model1.pt")

