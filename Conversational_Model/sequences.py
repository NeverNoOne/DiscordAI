import re
# 1. Data Preprocessing: Clean and Tokenize
def clean_text(text):
    # Lowercase and remove special characters (except punctuation marks important in language)
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9,?.!']+", " ", text)
    return text

def GetSequences(PATH):
    entries = [[]]

    with open(PATH, 'r') as f:
        text = f.readlines()

    text.pop(0)
    entries.pop(0)
    for t in text:
        s = t.split(",", 2)
        entries.append(s)

    from nltk.tokenize import word_tokenize
    from keras_preprocessing.text import Tokenizer
    from keras_preprocessing.sequence import pad_sequences

    # Sample data
    data = [
        [0, "hi, how are you doing?", "i'm fine. how about yourself?"],
        [1, "i'm fine. how about yourself?", "i'm pretty good. thanks for asking."],
        [2, "i'm pretty good. thanks for asking.", "no problem. so how have you been?"],
        # Add remaining rows from your dataset
        # ...
    ]
    data = entries
    

    # Clean the dataset
    for i in range(len(data)):
        data[i][1] = clean_text(data[i][1])  # Clean input text
        data[i][2] = clean_text(data[i][2])  # Clean target text

    # Extract input (user) and target (bot) messages
    input_texts = [item[1] for item in data]
    target_texts = ['<START> ' + item[2] + ' <END>' for item in data]

    # 2. Tokenization using Keras
    tokenizer = Tokenizer(filters='', lower=True, oov_token="<OOV>")  # Keeps punctuation for better meaning
    tokenizer.fit_on_texts(input_texts + target_texts)

    # Convert texts to sequences
    input_sequences = tokenizer.texts_to_sequences(input_texts)
    target_sequences = tokenizer.texts_to_sequences(target_texts)

    # 3. Padding: Ensure all sequences are the same length
    max_sequence_length = max(max(len(seq) for seq in input_sequences),
                              max(len(seq) for seq in target_sequences))

    input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='post')
    target_sequences = pad_sequences(target_sequences, maxlen=max_sequence_length, padding='post')

    # 4. Tokenizer Information
    vocab_size = len(tokenizer.word_index) + 1  # +1 for padding token
    word_index = tokenizer.word_index  # Dictionary mapping words to their index

    # 5. Print formatted input/output sequences
    print("Sample Input Sequences:", input_sequences[:2])
    print("Sample Target Sequences:", target_sequences[:2])
    print("Vocabulary Size:", vocab_size)
    return input_sequences, target_sequences, vocab_size, word_index


def GetSequences_max(PATH:str, max_size:int):
    entries = [[]]

    with open(PATH, 'r') as f:
        text = f.readlines()

    text.pop(0)
    entries.pop(0)
    for t in text:
        s = t.split(",", 2)
        entries.append(s)

    from nltk.tokenize import word_tokenize
    from keras_preprocessing.text import Tokenizer
    from keras_preprocessing.sequence import pad_sequences

    # Sample data
    data = [
        [0, "hi, how are you doing?", "i'm fine. how about yourself?"],
        [1, "i'm fine. how about yourself?", "i'm pretty good. thanks for asking."],
        [2, "i'm pretty good. thanks for asking.", "no problem. so how have you been?"],
        # Add remaining rows from your dataset
        # ...
    ]
    data = entries
    

    # Clean the dataset
    for i in range(len(data)):
        data[i][1] = clean_text(data[i][1])  # Clean input text
        data[i][2] = clean_text(data[i][2])  # Clean target text

    # Extract input (user) and target (bot) messages
    input_texts = [item[1] for item in data]
    target_texts = ['<START> ' + item[2] + ' <END>' for item in data]

    # 2. Tokenization using Keras
    tokenizer = Tokenizer(filters='', lower=True, oov_token="<OOV>")  # Keeps punctuation for better meaning
    tokenizer.fit_on_texts(input_texts + target_texts)

    # Convert texts to sequences
    input_sequences = tokenizer.texts_to_sequences(input_texts)
    target_sequences = tokenizer.texts_to_sequences(target_texts)

    # 3. Padding: Ensure all sequences are the same length
    max_sequence_length = max(max(len(seq) for seq in input_sequences),
                              max(len(seq) for seq in target_sequences))

    input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='post')
    target_sequences = pad_sequences(target_sequences, maxlen=max_sequence_length, padding='post')

    # 4. Tokenizer Information
    vocab_size = len(tokenizer.word_index) + 1  # +1 for padding token
    word_index = tokenizer.word_index  # Dictionary mapping words to their index

    # 5. Print formatted input/output sequences
    print("Sample Input Sequences:", input_sequences[:2])
    print("Sample Target Sequences:", target_sequences[:2])
    print("Vocabulary Size:", vocab_size)
    return input_sequences, target_sequences, vocab_size, word_index