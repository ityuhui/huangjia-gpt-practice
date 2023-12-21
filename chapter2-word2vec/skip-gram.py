sentences = ["Kage is Teacher", "Mazong is Boss", "Niuzhong is Boss", "Xiaobing is Student", "Xiaoxue is Student"]
words = ' '.join(sentences).split()
word_list = list(set(words))
word_to_idx = { word: idx for idx, word in enumerate(word_list) }
idx_to_word = { idx: word for idx, word in enumerate(word_list) }
voc_size = len(word_list)
print("Vocabulary: ", word_list)
print("From word to index: ", word_to_idx)
print("From index to word: ", idx_to_word)
print("Size of vocabulary: ", voc_size)

def create_skipgram_dataset(sentences, window_size=2):
    data = []
    for sentence in sentences:
        sentence = sentence.split()
        for idx, word in enumerate(word_list):
            for neighbor in sentence[max(idx - window_size, 0):min(idx + window_size + 1, len(sentence))]:
                if neighbor != word:
                    data.append((neighbor, word))
    return data

skipgram_data = create_skipgram_dataset(sentences)
print("Skip-Gram data example (Not encoded): ", skipgram_data[:3])

import torch
def one_hot_encoding(word, word_to_idx):
    tensor = torch.zeros(len(word_to_idx))
    tensor[word_to_idx[word]] = 1
    return tensor

word_example = "Teacher"
print("Word before One-Hot encoding: ", word_example)
print("One-Hot encoded vector: ", one_hot_encoding(word_example, word_to_idx))

print("Skip-Gram data example (Encoded): ", [(one_hot_encoding(context, word_to_idx), word_to_idx[target]) for context, target in skipgram_data[:3]])
