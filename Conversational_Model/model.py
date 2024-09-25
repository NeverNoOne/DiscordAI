import torch
import torch.nn as nn
import torch.nn.functional as F
from keras_preprocessing.sequence import pad_sequences
from sequences import clean_text

class ChatbotModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size, num_layers, device):
        """
        Initializes the ChatbotModel with embedding, encoder, attention, and decoder layers.
        
        Args:
            vocab_size (int): Size of the vocabulary.
            embed_size (int): Dimension of the embedding vectors.
            hidden_size (int): Number of features in the hidden state of the LSTM.
            output_size (int): Size of the output vocabulary.
            num_layers (int): Number of recurrent layers in the LSTM.
            device (torch.device): Device to run the model on (CPU or GPU).
        """
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
        self.max_length = 200  # Maximum length of the output sequence

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
    
    def predict(self, input_text, tokenizer, vocab_size, sos_token_idx:str, eos_token_idx:str):
        sos_token_idx = tokenizer.word_index[sos_token_idx.lower()]

        eos_token_idx = tokenizer.word_index[eos_token_idx.lower()]
        # Preprocess and tokenize the input text
        input_seq = tokenizer.texts_to_sequences([input_text])
        input_seq = pad_sequences(input_seq, maxlen=self.max_length, padding='post')
        input_seq = torch.LongTensor(input_seq).to(self.device)
        
        # Encode the input sequence
        embedded_input = self.embedding(input_seq)
        encoder_outputs, hidden, cell = self.encoder(embedded_input)
        
        # Start the decoding with the start-of-sequence token
        decoder_input = torch.LongTensor([[sos_token_idx]]).to(self.device)
        
        decoded_tokens = []
        for _ in range(self.max_length):
            # Embed the decoder input
            embedded_input = self.embedding(decoder_input)
            
            # Decode the token with attention
            output, hidden, cell = self.decoder(embedded_input, hidden, cell, encoder_outputs)
            
            # Get the token with the highest probability
            predicted_token = output.argmax(2).item()
            
            # Append the token to the result
            decoded_tokens.append(predicted_token)
            
            # If the model predicts the end-of-sequence token, stop decoding
            if predicted_token == eos_token_idx:
                break
            
            # Prepare the next input to the decoder (previous prediction)
            decoder_input = torch.LongTensor([[predicted_token]]).to(self.device)
        
        # Convert tokens back to words
        predicted_words = [tokenizer.index_word[token] for token in decoded_tokens if token in tokenizer.index_word]
        
        return ' '.join(predicted_words)

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

def getModel(
        # Define hyperparameters
        vocab_size = 10000  # Size of your vocabulary
        ,embed_size = 300    # Embedding size
        ,hidden_size = 512   # LSTM hidden size
        ,output_size = 10000  # Output size, which is the same as vocab size
        ,num_layers = 2      # Number of LSTM layers
):
    # Example usage:
    output_size = vocab_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    

    # Initialize the model
    model = ChatbotModel(vocab_size, embed_size, hidden_size, output_size, num_layers, device).to(device)

    return model, device
