import torch
from model import ChatbotModel, getModel

VOCAB_PATH = "M:/AI_Vocab/vocab_discord.json"

import os
import json
#loading vocab if exists
vocab = None
vocab_size = 0
if os.path.exists(VOCAB_PATH):
    with open(VOCAB_PATH, 'r') as f:
        vocab = json.load(f)
    vocab_size = len(vocab) + 1
else:
    print("Couldn't find a vocab!")
    exit()

model, device = getModel(vocab_size=vocab_size, num_layers=4)
model.load_state_dict(torch.load("M:/AI_Models/model_discord.pt", weights_only=True))
model.eval()
from keras_preprocessing.text import Tokenizer
input_text = "Hello"
tokenizer = Tokenizer(filters='', lower=True, oov_token="<OOV>")  # Keeps punctuation for better meaning
tokenizer.word_index = vocab
tokenizer.index_word = {index: word for word, index in vocab.items()}
output = model.predict(input_text, tokenizer, vocab_size, "<START>", "<END>")
print(output)

is_runnning = True
while is_runnning:
    input_text = input()
    if input_text.lower() == "exit":
         is_runnning = False
    output = model.predict(input_text, tokenizer, vocab_size, "<START>", "<END>")
    print(output)