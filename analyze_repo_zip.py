#!/usr/bin/env python3

import os
import re
import zipfile
import tempfile
import time
import hashlib
import google.generativeai as genai
from dotenv import load_dotenv

# 1. Load configuration
load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.0-flash")
ZIP_PATH = os.getenv("ZIP_PATH")
PROMPT_FILE = os.getenv("PROMPT_FILE")
SYSTEM_PROMPT_FILE = os.getenv("SYSTEM_PROMPT_FILE")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./output")

# Validation
required_vars = [API_KEY, ZIP_PATH, PROMPT_FILE, SYSTEM_PROMPT_FILE]
if not all(required_vars):
    print("Error: Missing variables in .env. Please check API_KEY, ZIP_PATH, PROMPT_FILE, and SYSTEM_PROMPT_FILE.")
    exit(1)

genai.configure(api_key=API_KEY)

def extract_repo_to_text(zip_path, output_path):
    """Extracts specific files from ZIP to a single flat text file."""
    valid_extensions = (
        '.py', '.js', '.ts', '.tsx', '.jsx', '.html', '.css', 
        '.json', '.md', '.yml', '.yaml', '.dockerfile', 'dockerfile', 
        '.txt', '.sh', '.java', '.go', '.rs', '.cpp', '.c', '.h', 
        '.sql', '.ini', '.toml', '.env.example'
    )
    ignore_dirs = ('.git', '__pycache__', 'node_modules', 'venv', '.idea', '.vscode', 'dist', 'build', 'coverage')

    with open(output_path, 'w', encoding='utf-8') as out_f:
        with zipfile.ZipFile(zip_path, 'r') as z:
            for filename in z.namelist():
                if filename.endswith('/') or any(d in filename for d in ignore_dirs):
                    continue
                
                if filename.lower().endswith(valid_extensions) or 'dockerfile' in filename.lower():
                    try:
                        content = z.read(filename).decode('utf-8')
                        out_f.write(f"--- START FILE: {filename} ---\n{content}\n--- END FILE: {filename} ---\n\n")
                    except Exception:
                        pass

def get_file_hash(filepath):
    """Computes MD5 hash to identify if the ZIP content has changed."""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def get_or_upload_file(local_path, mime_type="text/plain"):
    """
    Checks Gemini cache by listing files and matching 'display_name'.
    We cannot force the file 'name' (ID), so we rely on display_name containing the hash.
    """
    file_hash = get_file_hash(local_path)
    # Create a unique signature for the display name
    target_display_name = f"repo_context_{file_hash}"
    
    print(f"Computed context hash: {file_hash}")
    print("Checking for existing file in Gemini storage...")

    # Iterate over uploaded files to find a match by display_name
    # This is necessary because we cannot predict the server-assigned ID
    for f in genai.list_files():
        if f.display_name == target_display_name:
            print(f"Found cached file: {f.name} (URI: {f.uri})")
            return f

    # If not found, upload new
    print("File not found in cache. Uploading...")
    uploaded_file = genai.upload_file(
        path=local_path, 
        mime_type=mime_type, 
        display_name=target_display_name 
        # Note: We do NOT send 'name=' here, let Google assign the ID
    )
    
    # Wait for processing to complete
    while uploaded_file.state.name == "PROCESSING":
        print("Processing file on server...")
        time.sleep(2)
        uploaded_file = genai.get_file(uploaded_file.name)
        
    if uploaded_file.state.name == "FAILED":
        raise ValueError("File processing failed on Google side.")
        
    print(f"Upload complete: {uploaded_file.uri}")
    return uploaded_file

def save_generated_files(response_text):
    """
    Parses the LLM response looking for file delimiters and saves them.
    """
    pattern = re.compile(r"--- START OUTPUT: (.*?) ---\n(.*?)--- END OUTPUT: \1 ---", re.DOTALL)
    matches = pattern.findall(response_text)
    
    if not matches:
        print("\n--- NO FILES DETECTED IN RESPONSE ---")
        print("Printing raw response:\n")
        print(response_text)
        return

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

    print(f"\nFound {len(matches)} file(s) in the response.")

    for filename, content in matches:
        filename = filename.strip()
        filename = os.path.basename(filename) 
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        if os.path.exists(filepath):
            # Simple heuristic: if existing file content is identical, skip prompt
            # Otherwise, ask user
            pass # For now, we always ask or overwrite. Let's stick to asking.
            
            user_input = input(f"File '{filename}' already exists in {OUTPUT_DIR}. Overwrite? [y/N]: ")
            if user_input.lower() != 'y':
                print(f"Skipping {filename}.")
                continue
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content.strip())
            print(f"Saved: {filepath}")
        except Exception as e:
            print(f"Error saving {filename}: {e}")

# --- MAIN EXECUTION ---

for file_check in [ZIP_PATH, PROMPT_FILE, SYSTEM_PROMPT_FILE]:
    if not os.path.exists(file_check):
        print(f"Error: File not found: {file_check}")
        exit(1)

print("Reading prompt files...")
with open(SYSTEM_PROMPT_FILE, 'r', encoding='utf-8') as f:
    system_instruction = f.read()

with open(PROMPT_FILE, 'r', encoding='utf-8') as f:
    user_prompt_content = f.read()

print(f"Processing ZIP: {ZIP_PATH}...")
with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.txt', encoding='utf-8') as tmp_file:
    temp_txt_path = tmp_file.name

try:
    extract_repo_to_text(ZIP_PATH, temp_txt_path)
    gemini_file = get_or_upload_file(temp_txt_path)

    full_prompt = f"{system_instruction}\n\nUSER REQUEST:\n{user_prompt_content}"

    print(f"Sending request to {MODEL_NAME}...")
    model = genai.GenerativeModel(MODEL_NAME)
    
    response = model.generate_content(
        [gemini_file, full_prompt],
        request_options={"timeout": 600} 
    )

    print("Processing response...")
    save_generated_files(response.text)

finally:
    if os.path.exists(temp_txt_path):
        os.remove(temp_txt_path)
