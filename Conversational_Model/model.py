import torch
import torch.nn as nn
import torch.nn.functional as F
from keras_preprocessing.sequence import pad_sequences
from sequences import clean_text

class ChatbotModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size, num_layers, device):
        super(ChatbotModel, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # Encoder
        self.encoder = Encoder(embed_size, hidden_size, num_layers)
        
        # Attention mechanism
        self.attention = Attention(hidden_size)
        
        # Decoder
        self.decoder = Decoder(embed_size, hidden_size, output_size, num_layers, self.attention)
        
        self.device = device

    def forward(self, input_seq, target_seq):
        # Embedding the input sequence
        embedded_input = self.embedding(input_seq)
        
        # Encoding the input sequence
        encoder_outputs, hidden, cell = self.encoder(embedded_input)
        
        # Embedding the target sequence
        embedded_target = self.embedding(target_seq)
        
        # Decoding with attention mechanism
        output, _, _ = self.decoder(embedded_target, hidden, cell, encoder_outputs)
        
        return output

class Encoder(nn.Module):
    def __init__(self, embed_size, hidden_size, num_layers):
        super(Encoder, self).__init__()
        
        # LSTM layer for encoding
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        
    def forward(self, input_seq):
        encoder_outputs, (hidden, cell) = self.lstm(input_seq)
        return encoder_outputs, hidden, cell


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        
        # Attention layers
        self.attn = nn.Linear(hidden_size * 2, hidden_size)  # Compute attention score
        self.v = nn.Parameter(torch.rand(hidden_size))  # Trainable attention weight
        
    def forward(self, hidden, encoder_outputs):
        # Repeat the hidden state for each encoder output step
        hidden_repeated = hidden.unsqueeze(1).repeat(1, encoder_outputs.size(1), 1)
        
        # Concatenate the hidden state and encoder outputs
        combined = torch.cat((hidden_repeated, encoder_outputs), 2)
        
        # Compute attention scores
        energy = torch.tanh(self.attn(combined))
        
        # Apply the attention weight
        attn_weights = torch.sum(energy * self.v, dim=2)
        
        # Apply softmax to get normalized attention scores
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=1)
        
        # Compute the context vector as a weighted sum of encoder outputs
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        
        return context, attn_weights


class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, output_size, num_layers, attention):
        super(Decoder, self).__init__()
        
        # Attention mechanism
        self.attention = attention
        
        # LSTM layer for decoding
        self.lstm = nn.LSTM(embed_size + hidden_size, hidden_size, num_layers, batch_first=True)
        
        # Fully connected layer for generating the final output token
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, input_seq, hidden, cell, encoder_outputs):
        # Compute the attention context vector
        context, attn_weights = self.attention(hidden[-1], encoder_outputs)
        
        # Concatenate the context vector with the input sequence embedding
        context_repeated = context.unsqueeze(1).repeat(1, input_seq.size(1), 1)
        input_combined = torch.cat((input_seq, context_repeated), 2)
        
        # Decode the sequence
        output, (hidden, cell) = self.lstm(input_combined, (hidden, cell))
        
        # Predict the next token
        output = self.fc(output)
        
        return output, hidden, cell


## Example usage:
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
## Define hyperparameters
#vocab_size = 10000  # Size of your vocabulary
#embed_size = 300    # Embedding size
#hidden_size = 512   # LSTM hidden size
#output_size = vocab_size  # Output size, which is the same as vocab size
#num_layers = 2      # Number of LSTM layers
#
## Initialize the model
#model = ChatbotModel(vocab_size, embed_size, hidden_size, output_size, num_layers, device).to(device)
#
## Example input and target sequences
#input_seq = torch.randint(0, vocab_size, (32, 10)).to(device)  # Batch of 32 sequences, each 10 tokens long
#target_seq = torch.randint(0, vocab_size, (32, 10)).to(device)
#
## Forward pass
#output = model(input_seq, target_seq)
#
#print(output.shape)  # Output shape should be (batch_size, seq_len, vocab_size)


def getModel(
        # Define hyperparameters
        vocab_size = 10000  # Size of your vocabulary
        ,embed_size = 300    # Embedding size
        ,hidden_size = 512   # LSTM hidden size
        ,output_size = 10000  # Output size, which is the same as vocab size
        ,num_layers = 2      # Number of LSTM layers
):
    # Example usage:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    

    # Initialize the model
    model = ChatbotModel(vocab_size, embed_size, hidden_size, output_size, num_layers, device).to(device)

    return model, device
