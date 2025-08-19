#!/usr/bin/env python3
"""Simple data rewriting script for the pipeline"""

import os
import json
import argparse
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    parser = argparse.ArgumentParser(description="Simple data rewriting")
    parser.add_argument("--input", required=True, help="Input file path")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--rewrite_ratio", type=float, default=0.3, help="Rewrite ratio")
    parser.add_argument("--template", default="professional", help="Template type")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Read input data
    print(f"Reading input from: {args.input}")
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading input file: {e}")
        return 1
    
    # For now, just copy the data as-is (since we don't have the API key)
    # In a real scenario, this would call the rewriting API
    print(f"Processing {len(data) if isinstance(data, list) else 1} items...")
    print(f"Rewrite ratio: {args.rewrite_ratio}")
    print(f"Template: {args.template}")
    
    # Save output
    output_file = os.path.join(args.output, "rewritten_qa.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"Data saved to: {output_file}")
    print("Rewriting completed (placeholder mode - no actual rewriting performed)")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())