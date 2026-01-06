import sys
from pathlib import Path
import pandas as pd
import unittest

# allow imports from project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.docs import row_to_document, dataframe_to_documents

class TestDocs(unittest.TestCase):
    def test_row_to_document(self):
        row = pd.Series({
            "Complaint ID": "12345",
            "Product": "Credit card",
            "Issue": "Late fee",
            "Company": "Test Bank",
            "clean_narrative": "I was charged a late fee incorrectly."
        })
        
        doc = row_to_document(row)
        
        # Verify content
        self.assertEqual(doc.page_content, "I was charged a late fee incorrectly.")
        
        # Verify metadata (keys were lowercased in row_to_document)
        self.assertEqual(doc.metadata["complaint_id"], "12345")
        self.assertEqual(doc.metadata["product"], "Credit card")
        self.assertEqual(doc.metadata["company"], "Test Bank")
        
    def test_dataframe_to_documents(self):
        df = pd.DataFrame([
            {"Complaint ID": "1", "clean_narrative": "Text 1", "Product": "P1"},
            {"Complaint ID": "2", "clean_narrative": "Text 2", "Product": "P2"},
            {"Complaint ID": "3", "clean_narrative": "", "Product": "P3"} # Empty content
        ])
        
        docs = dataframe_to_documents(df)
        
        # Should only have 2 docs (empty one skipped)
        self.assertEqual(len(docs), 2)
        print("✓ Docs verification passed!")

if __name__ == "__main__":
    unittest.main()
