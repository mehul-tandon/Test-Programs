from sklearn.feature_extraction.text import CountVectorizer

# Sample documents
docs = [
    "the cat sat on the mat",
    "the dog barked loudly",
    "cats and dogs are friends"
]

# Step 1: Initialize BoW vectorizer
vectorizer = CountVectorizer()

# Step 2: Fit and transform the documents into sparse matrix
sparse_matrix = vectorizer.fit_transform(docs)

# Step 3: Inspect results
print("Vocabulary:", vectorizer.get_feature_names_out())  # List of words
print("Sparse Matrix Shape:", sparse_matrix.shape)        # (docs, vocab size)
print("Sparse Matrix:\n", sparse_matrix.toarray())        # Dense view for clarity
