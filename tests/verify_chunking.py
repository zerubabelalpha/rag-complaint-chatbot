import sys
from pathlib import Path
import unittest
from langchain_core.documents import Document

# allow imports from project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.chunking import chunk_documents, get_chunk_stats

class TestChunking(unittest.TestCase):
    def test_chunk_documents(self):
        # Create a document with a long narrative
        text = "This is a long sentence. " * 30  # Approx 750 characters
        metadata = {
            "complaint_id": "12345",
            "product": "Credit card",
            "company": "Test Bank"
        }
        doc = Document(page_content=text, metadata=metadata)
        
        # Chunk the document (default size 500)
        chunks = chunk_documents([doc], chunk_size=300, chunk_overlap=50)
        
        # Verify chunks created
        self.assertGreater(len(chunks), 1)
        
        # Verify metadata preservation and chunk_index
        for i, chunk in enumerate(chunks):
            self.assertEqual(chunk.metadata["complaint_id"], "12345")
            self.assertEqual(chunk.metadata["product"], "Credit card")
            self.assertEqual(chunk.metadata["chunk_index"], i)
            
        print("✓ Chunking verification passed!")

    def test_chunk_stats(self):
        docs = [
            Document(page_content="short text", metadata={"complaint_id": "1"}),
            Document(page_content="very very long text" * 10, metadata={"complaint_id": "2"})
        ]
        chunks = chunk_documents(docs, chunk_size=50, chunk_overlap=10)
        stats = get_chunk_stats(chunks)
        
        self.assertIn("total_chunks", stats)
        self.assertIn("mean_length", stats)
        print(f"Chunk stats: {stats}")

if __name__ == "__main__":
    unittest.main()
