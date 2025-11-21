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
    files = list(genai.list_files())
except Exception as e:
    print(f"Error retrieving file list: {e}")
    sys.exit(1)

# 1. If no files exist
if not files:
    print("No files found in storage. Cleanup not required.")
    sys.exit(0)

# Prepare data for the table to calculate widths
table_data = []
for f in files:
    table_data.append({
        "name": getattr(f, 'name', 'N/A'),
        "display": getattr(f, 'display_name', 'N/A'),
        "mime": getattr(f, 'mime_type', 'N/A'),
        "size": str(getattr(f, 'size_bytes', '0')), # Convert number to string
        "state": getattr(f, 'state', {}).name if hasattr(f, 'state') else 'UNKNOWN'
    })

# Define headers
headers = {
    "name": "ID (Name)",
    "display": "Display Name",
    "mime": "MIME Type",
    "size": "Size (Bytes)",
    "state": "State"
}

# Calculate max width for each column (Header vs Content)
w_name = max(len(headers["name"]), max((len(x["name"]) for x in table_data), default=0))
w_display = max(len(headers["display"]), max((len(x["display"]) for x in table_data), default=0))
w_mime = max(len(headers["mime"]), max((len(x["mime"]) for x in table_data), default=0))
w_size = max(len(headers["size"]), max((len(x["size"]) for x in table_data), default=0))
w_state = max(len(headers["state"]), max((len(x["state"]) for x in table_data), default=0))

# 2. Show list
print(f"\nFiles found: {len(files)}\n")

# Dynamic format string
row_fmt = f"{{:<{w_name}}} | {{:<{w_display}}} | {{:<{w_mime}}} | {{:<{w_size}}} | {{:<{w_state}}}"

# Print Header
print(row_fmt.format(headers["name"], headers["display"], headers["mime"], headers["size"], headers["state"]))

# Print Separator (sum of widths + 3 chars per separator * 4 separators)
total_width = w_name + w_display + w_mime + w_size + w_state + (3 * 4)
print("-" * total_width)

# Print Rows
for row in table_data:
    print(row_fmt.format(row["name"], row["display"], row["mime"], row["size"], row["state"]))

print("-" * total_width)

# 2.1. Ask for confirmation
confirm = input("\nAre you sure you want to delete ALL these files? (y/N): ").strip().lower()

# 2.2. If Y, delete
if confirm == 'y':
    print("\nStarting deletion...")
    deleted_count = 0
    for f in files:
        try:
            # Use original file object for deletion
            display = getattr(f, 'display_name', f.name)
            print(f"Deleting {display}...", end=" ")
            genai.delete_file(f.name)
            print("OK")
            deleted_count += 1
        except Exception as e:
            print(f"Error: {e}")
    
    print(f"\nDone. Files deleted: {deleted_count} of {len(files)}.")
else:
    print("\nOperation cancelled by user. No files were deleted.")
