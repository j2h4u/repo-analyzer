#!/usr/bin/env python3

import google.generativeai as genai
import os
from dotenv import load_dotenv
import sys

# Load environment variables
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("Error: GOOGLE_API_KEY not found in environment variables.")
    sys.exit(1)

genai.configure(api_key=api_key)

print("Retrieving file list...")

try:
    # Convert generator to list to get count and reuse it
    files = list(genai.list_files())
except Exception as e:
    print(f"Error retrieving file list: {e}")
    sys.exit(1)

# 1. If no files exist
if not files:
    print("No files found in storage. Cleanup not required.")
    sys.exit(0)

# 2. If files exist, show list and metadata
print(f"\nFiles found: {len(files)}\n")
print(f"{'ID (Name)':<20} | {'Display Name':<25} | {'MIME Type':<15} | {'Size (Bytes)':<12} | {'State'}")
print("-" * 95)

for f in files:
    # Safely retrieve attributes in case of API changes
    f_name = getattr(f, 'name', 'N/A')
    f_display = getattr(f, 'display_name', 'N/A')
    f_mime = getattr(f, 'mime_type', 'N/A')
    f_size = getattr(f, 'size_bytes', '0')
    f_state = getattr(f, 'state', {}).name if hasattr(f, 'state') else 'UNKNOWN'
    
    print(f"{f_name:<20} | {f_display:<25} | {f_mime:<15} | {f_size:<12} | {f_state}")

print("-" * 95)

# 2.1. Ask for confirmation
confirm = input("\nAre you sure you want to delete ALL these files? (y/N): ").strip().lower()

# 2.2. If Y, delete
if confirm == 'y':
    print("\nStarting deletion...")
    deleted_count = 0
    for f in files:
        try:
            print(f"Deleting {f.display_name} ({f.name})...", end=" ")
            genai.delete_file(f.name)
            print("OK")
            deleted_count += 1
        except Exception as e:
            print(f"Error: {e}")
    
    print(f"\nDone. Files deleted: {deleted_count} of {len(files)}.")
else:
    print("\nOperation cancelled by user. No files were deleted.")
