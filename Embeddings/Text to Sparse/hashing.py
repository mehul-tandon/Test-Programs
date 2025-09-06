from sklearn.feature_extraction.text import HashingVectorizer

# Sample text documents
docs = [
    "the cat sat on the mat",
    "the dog barked loudly",
    "cats and dogs are good friends"
]

# Step 1: Initialize HashingVectorizer
# n_features = number of dimensions (default 2^20 ~ 1M, here we use smaller for clarity)
vectorizer = HashingVectorizer(n_features=16, norm=None, alternate_sign=False)

# Step 2: Transform the documents
sparse_matrix = vectorizer.transform(docs)

# Step 3: Sparse matrix info
print("Sparse Matrix Shape:", sparse_matrix.shape)

# Step 4: Inspect values (row, col, value)
print("\nSparse Matrix (row, col, value):")
coo = sparse_matrix.tocoo()
for row, col, val in zip(coo.row, coo.col, coo.data):
    print(f"({row}, {col}) -> {val}")
