from gensim.models import Word2Vec
import numpy as np

# Sample dataset (tokenized sentences)
sentences = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "dog", "barked", "loudly"],
    ["cats", "and", "dogs", "are", "good", "friends"]
]

# Step 1: Train Word2Vec model
# vector_size = dimension of embedding, window = context window
model = Word2Vec(sentences, vector_size=50, window=5, min_count=1, workers=4)

# Step 2: Function to compute sentence embedding (average of word vectors)
def sentence_embedding(sentence, model):
    vectors = [model.wv[word] for word in sentence if word in model.wv]
    if not vectors:
        return np.zeros(model.vector_size)  # handle empty case
    return np.mean(vectors, axis=0)

# Step 3: Get embeddings for each sentence
for sent in sentences:
    emb = sentence_embedding(sent, model)
    print(f"Sentence: {' '.join(sent)}")
    print(f"Embedding shape: {emb.shape}")
    print(f"Embedding (first 5 dims): {emb[:5]}")
    print("-" * 50)
