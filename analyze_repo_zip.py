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
from halo import Halo

# --- 1. INIT & CONFIG LOADING ---

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    print("Error: GOOGLE_API_KEY not found in .env file.")
    exit(1)

genai.configure(api_key=API_KEY)

CONFIG_FILE = "config.yaml"

if not os.path.exists(CONFIG_FILE):
    print(f"Error: Configuration file '{CONFIG_FILE}' not found. Please copy config.yaml.example to config.yaml")
    exit(1)

with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

def get_required_cfg(path):
    keys = path.split('.')
    val = config
    try:
        for key in keys:
            val = val[key]
        return val
    except KeyError:
        print(f"Error: Missing required configuration key: '{path}' in {CONFIG_FILE}")
        exit(1)

# Load Variables
ZIP_PATH = get_required_cfg('project.zip_path')
PROMPT_FILE = get_required_cfg('project.prompt_file')
SYSTEM_PROMPT_FILE = get_required_cfg('project.system_prompt_file')
OUTPUT_DIR = get_required_cfg('project.output_dir')
REPORT_FILE = get_required_cfg('project.report_file') # New config
MODEL_NAME = get_required_cfg('model.name')
TIMEOUT = get_required_cfg('model.timeout')
PROCESSING_CFG = get_required_cfg('processing')

# --- 2. PRE-CHECKS ---

def check_model_availability(model_name):
    """Checks if the requested model exists in the API before doing heavy lifting."""
    spinner = Halo(text=f'Verifying model access: {model_name}...', spinner='dots')
    spinner.start()
    try:
        found = False
        # Google API usually returns names like 'models/gemini-1.5-flash'
        for m in genai.list_models():
            if m.name == model_name or m.name.endswith(f"/{model_name}"):
                found = True
                break
        
        if found:
            spinner.succeed(f"Model '{model_name}' is valid.")
        else:
            spinner.fail(f"Model '{model_name}' not found in your available models list.")
            print("  Tip: Check `genai.list_models()` to see available options.")
            exit(1)
    except Exception as e:
        spinner.fail(f"API Connection Error: {e}")
        exit(1)

# --- 3. CORE LOGIC ---

def extract_and_report(zip_path, output_path, report_path, processing_config):
    """
    Extracts text files to a single file AND generates a skipped files report.
    """
    valid_exts = tuple(processing_config.get('valid_extensions', []))
    include_names = set(n.lower() for n in processing_config.get('include_filenames', []))
    ignore_dirs = processing_config.get('ignore_dirs', [])

    # Reporting structures
    skipped_no_ext = []
    skipped_by_ext = {}

    with open(output_path, 'w', encoding='utf-8') as out_f, \
         zipfile.ZipFile(zip_path, 'r') as z:
        
        all_files = z.namelist()
        
        for filename in all_files:
            # 1. Filter out directories and ignored folders (Noise reduction)
            if filename.endswith('/') or any(d in filename for d in ignore_dirs):
                continue
            
            # 2. Determine inclusion
            is_valid_ext = filename.lower().endswith(valid_exts)
            is_included_name = os.path.basename(filename).lower() in include_names
            
            if is_valid_ext or is_included_name:
                # -- PROCESS VALID FILE --
                try:
                    content = z.read(filename).decode('utf-8')
                    out_f.write(f"--- START FILE: {filename} ---\n{content}\n--- END FILE: {filename} ---\n\n")
                except Exception:
                    # Binary or decoding error -> treat as skipped
                    ext = os.path.splitext(filename)[1].lower()
                    if ext:
                        skipped_by_ext.setdefault(ext, []).append(filename + " (Decode Error)")
                    else:
                        skipped_no_ext.append(filename + " (Decode Error)")
            else:
                # -- LOG SKIPPED FILE --
                ext = os.path.splitext(filename)[1].lower()
                if not ext:
                    skipped_no_ext.append(filename)
                else:
                    skipped_by_ext.setdefault(ext, []).append(filename)

    # Write Report
    with open(report_path, 'w', encoding='utf-8') as rep:
        rep.write(f"--- SKIPPED FILES REPORT ---\n")
        rep.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        rep.write(f"Ignored Folders (excluded completely): {', '.join(ignore_dirs)}\n\n")
        
        if skipped_no_ext:
            rep.write(f"### Misc / No Extension ({len(skipped_no_ext)} files)\n")
            for f in sorted(skipped_no_ext):
                rep.write(f" - {f}\n")
            rep.write("\n")
            
        rep.write("### Skipped by Extension\n")
        for ext, files in sorted(skipped_by_ext.items()):
            count = len(files)
            if count > 10:
                # Summary for noise reduction
                rep.write(f" - {ext}: {count} files omitted from report (too many)\n")
            else:
                # Full list for small amounts
                rep.write(f" - {ext} ({count}):\n")
                for f in sorted(files):
                    rep.write(f"    * {f}\n")

def get_file_hash(filepath):
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def get_or_upload_file(local_path, mime_type="text/plain"):
    file_hash = get_file_hash(local_path)
    target_display_name = f"repo_context_{file_hash}"
    
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
            spinner.fail("File processing failed.")
            raise ValueError("Upload failed state")
            
        spinner.succeed(f"Upload complete.")
        return uploaded_file, False
    except Exception as e:
        spinner.fail(f"Upload error: {e}")
        raise e

def save_generated_files(response_text):
    pattern = re.compile(r"--- START OUTPUT: (.*?) ---\n(.*?)--- END OUTPUT: \1 ---", re.DOTALL)
    matches = pattern.findall(response_text)
    
    if not matches:
        print("\n--- NO FILES DETECTED IN RESPONSE ---")
        # Optionally save raw response to debug
        return

    print(f"\nFound {len(matches)} file(s). Saving to '{OUTPUT_DIR}'...")

    for filename, content in matches:
        filename = filename.strip()
        safe_output_dir = os.path.abspath(OUTPUT_DIR)
        full_path = os.path.abspath(os.path.join(safe_output_dir, filename))
        
        if not full_path.startswith(safe_output_dir):
            print(f"Security Warning: Skipping suspicious path '{filename}'")
            continue

        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        if os.path.exists(full_path):
            # For automated runs, consider a flag to force overwrite. 
            # Currently asking user:
            user_input = input(f"  Overwrite '{filename}'? [y/N]: ")
            if user_input.lower() != 'y':
                print(f"  Skipped.")
                continue
        
        try:
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content.strip())
            print(f"  Saved: {filename}")
        except Exception as e:
            print(f"  Error saving {filename}: {e}")

# --- MAIN EXECUTION ---

# 1. Validate paths
for p in [ZIP_PATH, PROMPT_FILE, SYSTEM_PROMPT_FILE]:
    if not os.path.exists(p):
        print(f"Error: File not found: {p}")
        exit(1)

# 2. Validate API Model
check_model_availability(MODEL_NAME)

# 3. Prepare Prompts
print("\nReading prompts...")
with open(SYSTEM_PROMPT_FILE, 'r', encoding='utf-8') as f:
    system_instruction = f.read()
with open(PROMPT_FILE, 'r', encoding='utf-8') as f:
    user_prompt_content = f.read()

# 4. Extract & Report
with Halo(text=f'Processing ZIP: {ZIP_PATH}...', spinner='dots'):
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.txt', encoding='utf-8') as tmp_file:
        temp_txt_path = tmp_file.name
    extract_and_report(ZIP_PATH, temp_txt_path, REPORT_FILE, PROCESSING_CFG)

print(f"Diagnostic report saved to: {REPORT_FILE}")

# 5. Gemini Interaction
try:
    gemini_file, is_cached = get_or_upload_file(temp_txt_path)
    if is_cached:
        print(f"Using cached context hash: {gemini_file.display_name}")

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
