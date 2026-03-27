import os
import time
from uuid import uuid4

# Set up logging for visibility
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    print(f"❌ Missing library: {e}")
    print("Please run: pip install sentence-transformers qdrant-client torch")
    exit(1)

def main():
    # 1. Configuration
    COLLECTION_NAME = "test_mxbai_collection"
    MODEL_NAME = "mixedbread-ai/mxbai-embed-large-v1"
    QDRANT_URL = "http://localhost:6333"  # Your local Docker Qdrant
    
    # 2. Sample Data
    paragraph = (
        "Project Vanguard was one of the United States' earliest 5 satellite programs. "
        "It was managed by the Naval Research Laboratory (NRL) and was intended to be "
        "the first American satellite effort before Sputnik."
    )
    
    # 3. Initialize Model
    logging.info(f"Loading embedding model: {MODEL_NAME}...")
    logging.info("(This may take a moment if it needs to download or load into memory)")
    
    # Load model and check device
    model = SentenceTransformer(MODEL_NAME)
    logging.info(f"✅ Model loaded successfully on device: {model.device}")
    
    # 4. Generate Embeddings
    logging.info("Generating embedding for the text paragraph...")
    start_time = time.time()
    
    # We pass it as a list of strings
    dense_vector = model.encode([paragraph])[0].tolist()
    
    elapsed = time.time() - start_time
    logging.info(f"✅ Generated embedding vector of dimension {len(dense_vector)} in {elapsed:.2f} seconds.")

    # 5. Connect to Qdrant
    logging.info(f"Connecting to Qdrant at {QDRANT_URL}...")
    client = QdrantClient(url=QDRANT_URL)
    
    # 6. Recreate Collection (Start fresh)
    logging.info(f"Recreating collection '{COLLECTION_NAME}'...")
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
        
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=len(dense_vector),  # Should be 1024 for mxbai
            distance=Distance.COSINE
        )
    )
    logging.info("✅ Collection created.")

    # 7. Upsert Data to Qdrant
    points = [
        PointStruct(
            id=str(uuid4()),
            vector=dense_vector,
            payload={
                "source": "NASA History",
                "text": paragraph
            }
        )
    ]
    
    logging.info("Uploading data to Qdrant...")
    operation_info = client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )
    
    logging.info("🎉 Success! Paragraph was embedded and stored in Qdrant.")
    print("\n--- Upsert Result ---")
    print(operation_info)
    
    # Verify by checking collection count
    count = client.count(COLLECTION_NAME).count
    print(f"\nTotal points in '{COLLECTION_NAME}': {count}")


if __name__ == "__main__":
    main()
