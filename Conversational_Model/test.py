import torch
from model import ChatbotModel, getModel

VOCAB_PATH = "M:/AI_Vocab/vocab_full.json"

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

model, device = getModel(vocab_size=vocab_size, embed_size=150, hidden_size=256, num_layers=2)
model.load_state_dict(torch.load("M:/AI_Models/model_full.pt", weights_only=True))
model.eval()
from keras_preprocessing.text import Tokenizer
input_text = "Hello"
tokenizer = Tokenizer(filters='', lower=True, oov_token="<OOV>")  # Keeps punctuation for better meaning
tokenizer.word_index = vocab
#TODO RuntimeWarning: assigning None to unbound local 'index'
#TODO for some reason the output is always the same?????
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