#!/usr/bin/env python3

import os
import re
import zipfile
import tempfile
import time
import hashlib
import yaml
import google.generativeai as genai
from dotenv import load_dotenv
from halo import Halo  # For loading spinners

# 1. Load Secrets
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    print("Error: GOOGLE_API_KEY not found in .env file.")
    exit(1)

genai.configure(api_key=API_KEY)

# 2. Load Configuration
CONFIG_FILE = "config.yaml"

if not os.path.exists(CONFIG_FILE):
    print(f"Error: Configuration file '{CONFIG_FILE}' not found. Please copy config.yaml.example to config.yaml")
    exit(1)

with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

def get_required_cfg(path):
    """
    Retrieves config value. Exits strictly if key is missing.
    Path example: 'project.zip_path'
    """
    keys = path.split('.')
    val = config
    try:
        for key in keys:
            val = val[key]
        return val
    except KeyError:
        print(f"Error: Missing required configuration key: '{path}' in {CONFIG_FILE}")
        exit(1)

# Strict variable loading
ZIP_PATH = get_required_cfg('project.zip_path')
PROMPT_FILE = get_required_cfg('project.prompt_file')
SYSTEM_PROMPT_FILE = get_required_cfg('project.system_prompt_file')
OUTPUT_DIR = get_required_cfg('project.output_dir')
MODEL_NAME = get_required_cfg('model.name')
TIMEOUT = get_required_cfg('model.timeout')
PROCESSING_CFG = get_required_cfg('processing')

def extract_repo_to_text(zip_path, output_path, processing_config):
    """Extracts specific files from ZIP based on YAML configuration."""
    valid_exts = tuple(processing_config.get('valid_extensions', []))
    include_names = set(n.lower() for n in processing_config.get('include_filenames', []))
    ignore_dirs = processing_config.get('ignore_dirs', [])

    with open(output_path, 'w', encoding='utf-8') as out_f:
        with zipfile.ZipFile(zip_path, 'r') as z:
            for filename in z.namelist():
                if filename.endswith('/') or any(d in filename for d in ignore_dirs):
                    continue
                
                is_valid_ext = filename.lower().endswith(valid_exts)
                is_included_name = os.path.basename(filename).lower() in include_names
                
                if is_valid_ext or is_included_name:
                    try:
                        content = z.read(filename).decode('utf-8')
                        out_f.write(f"--- START FILE: {filename} ---\n{content}\n--- END FILE: {filename} ---\n\n")
                    except Exception:
                        pass

def get_file_hash(filepath):
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def get_or_upload_file(local_path, mime_type="text/plain"):
    file_hash = get_file_hash(local_path)
    target_display_name = f"repo_context_{file_hash}"
    
    print(f"Computed context hash: {file_hash}")

    # Check Cache
    with Halo(text='Checking Gemini cache...', spinner='dots'):
        for f in genai.list_files():
            if f.display_name == target_display_name:
                return f, True # Found

    # Upload
    spinner = Halo(text='Uploading file to Gemini...', spinner='dots')
    spinner.start()
    
    try:
        uploaded_file = genai.upload_file(
            path=local_path, 
            mime_type=mime_type, 
            display_name=target_display_name 
        )
        
        while uploaded_file.state.name == "PROCESSING":
            spinner.text = "Processing file on server..."
            time.sleep(2)
            uploaded_file = genai.get_file(uploaded_file.name)
            
        if uploaded_file.state.name == "FAILED":
            spinner.fail("File processing failed on Google side.")
            raise ValueError("Upload failed")
            
        spinner.succeed(f"Upload complete")
        return uploaded_file, False
    except Exception as e:
        spinner.fail(f"Upload error: {e}")
        raise e

def save_generated_files(response_text):
    pattern = re.compile(r"--- START OUTPUT: (.*?) ---\n(.*?)--- END OUTPUT: \1 ---", re.DOTALL)
    matches = pattern.findall(response_text)
    
    if not matches:
        print("\n--- NO FILES DETECTED IN RESPONSE ---")
        print(response_text)
        return

    print(f"\nFound {len(matches)} file(s).")

    for filename, content in matches:
        filename = filename.strip()
        
        # Security: Prevent Path Traversal (e.g. "../../etc/passwd")
        # We resolve the absolute path and ensure it starts with the output dir
        safe_output_dir = os.path.abspath(OUTPUT_DIR)
        full_path = os.path.abspath(os.path.join(safe_output_dir, filename))
        
        if not full_path.startswith(safe_output_dir):
            print(f"Security Warning: Skipping suspicious path '{filename}'")
            continue

        # Create subdirectories if needed (e.g. output/src/utils/)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        if os.path.exists(full_path):
            # Interactive check could be added here, but for automated flows usually we overwrite 
            # or we can ask using a cleaner UI library. For now, standard input:
            user_input = input(f"File '{filename}' exists. Overwrite? [y/N]: ")
            if user_input.lower() != 'y':
                print(f"Skipping {filename}.")
                continue
        
        try:
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content.strip())
            print(f"Saved: {filename}")
        except Exception as e:
            print(f"Error saving {filename}: {e}")

# --- MAIN EXECUTION ---

for p in [ZIP_PATH, PROMPT_FILE, SYSTEM_PROMPT_FILE]:
    if not os.path.exists(p):
        print(f"Error: File not found: {p}")
        exit(1)

print("Reading prompts...")
with open(SYSTEM_PROMPT_FILE, 'r', encoding='utf-8') as f:
    system_instruction = f.read()
with open(PROMPT_FILE, 'r', encoding='utf-8') as f:
    user_prompt_content = f.read()

# Processing ZIP with Spinner
with Halo(text=f'Extracting ZIP: {ZIP_PATH}...', spinner='dots'):
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.txt', encoding='utf-8') as tmp_file:
        temp_txt_path = tmp_file.name
    extract_repo_to_text(ZIP_PATH, temp_txt_path, PROCESSING_CFG)

try:
    gemini_file, is_cached = get_or_upload_file(temp_txt_path)
    if is_cached:
        print(f"Using cached file: {gemini_file.uri}")

    full_prompt = f"{system_instruction}\n\nUSER REQUEST:\n{user_prompt_content}"

    spinner = Halo(text=f'Generating content with {MODEL_NAME}...', spinner='dots')
    spinner.start()
    
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(
            [gemini_file, full_prompt],
            request_options={"timeout": TIMEOUT} 
        )
        spinner.succeed("Generation complete!")
    except Exception as e:
        spinner.fail(f"Generation failed: {e}")
        exit(1)

    save_generated_files(response.text)

finally:
    if os.path.exists(temp_txt_path):
        os.remove(temp_txt_path)
