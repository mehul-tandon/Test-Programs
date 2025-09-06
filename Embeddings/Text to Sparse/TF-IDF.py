from sklearn.feature_extraction.text import TfidfVectorizer

# Sample text documents
docs = [
    "the cat sat on the mat",
    "the dog barked loudly",
    "cats and dogs are good friends"
]

# Step 1: Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer()

# Step 2: Fit and transform the documents into sparse matrix
sparse_matrix = vectorizer.fit_transform(docs)

# Step 3: Inspect (Sparse Representation)
print("Vocabulary:", vectorizer.get_feature_names_out())
print("Sparse Matrix Shape:", sparse_matrix.shape)

# If you want to see non-zero values & positions
print("\nSparse Matrix (CSR format):\n", sparse_matrix)

