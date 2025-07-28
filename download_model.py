# download_model.py
from sentence_transformers import SentenceTransformer

print("Downloading model…")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("Saving model locally…")
model.save("models/all-MiniLM-L6-v2")
print("Done.")
