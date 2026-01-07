import sys
from pathlib import Path
import re

# Allow imports from src
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.preprocess import clean_complaint_text

def test_cleaning():
    print("--- Testing Advanced Text Cleaning ---")
    
    test_cases = [
        {
            "name": "Redaction Removal",
            "input": "I was charged XXXX dollars on XX/XX/XXXX and XXXXXX.",
            "expected_contains": ["charged", "dollar", "on"],
            "expected_not_contains": ["XXXX", "XX/XX/XXXX"]
        },
        {
            "name": "Boilerplate Removal",
            "input": "To whom it may concern, I am writing to file a complaint regarding my credit card. Sincerely, John Doe.",
            "expected_contains": ["credit card", "my"],
            "expected_not_contains": ["to whom it may concern", "sincerely"]
        },
        {
            "name": "Currency and Punctuation",
            "input": "I lost $500.00!!! and 5% interest... common.",
            "expected_contains": ["dollar", "percent", "interest"],
            "expected_not_contains": ["$"]
        },
        {
            "name": "Semantic Preservation",
            "input": "This is a sentence. This is another sentence.",
            "expected_contains": ["sentence. this"]
        }
    ]
    
    all_passed = True
    for case in test_cases:
        output = clean_complaint_text(case["input"])
        print(f"\n[Test: {case['name']}]")
        print(f"Input:  {case['input']}")
        print(f"Output: {output}")
        
        passed = True
        for skip in case.get("expected_not_contains", []):
            if skip.lower() in output.lower():
                print(f"❌ FAIL: Found unwanted text '{skip}'")
                passed = False
        
        for keep in case.get("expected_contains", []):
            if keep.lower() not in output.lower():
                print(f"❌ FAIL: Did not find '{keep}'")
                passed = False
        
        if passed:
            print("✅ Pass")
        else:
            all_passed = False
            
    if all_passed:
        print("\n🏆 All text cleaning tests passed!")
    else:
        print("\n❌ Some tests failed.")

if __name__ == "__main__":
    test_cleaning()
