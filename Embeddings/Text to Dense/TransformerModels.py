from sentence_transformers import SentenceTransformer

# Load a pre-trained transformer model (MiniLM is lightweight & fast)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Sample text data
sentences = [
    "Machine learning is amazing.",
    "Transformers are powerful for natural language processing.",
    "Embeddings convert text into numerical vectors."
]

# Convert text into dense embeddings
embeddings = model.encode(sentences)

# Show results
print("Shape of embeddings:", embeddings.shape)
print("First embedding vector:\n", embeddings[0])
