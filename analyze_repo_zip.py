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
from dataclasses import dataclass
from typing import List

# --- 1. CONFIGURATION STRUCTS ---

@dataclass
class ProjectConfig:
    zip_path: str
    prompt_file: str
    system_prompt_file: str
    output_dir: str
    report_file: str

@dataclass
class ModelConfig:
    name: str
    timeout: int

@dataclass
class ProcessingConfig:
    valid_extensions: List[str]
    include_filenames: List[str]
    ignore_dirs: List[str]

@dataclass
class AppConfig:
    project: ProjectConfig
    model: ModelConfig
    processing: ProcessingConfig

    @classmethod
    def load(cls, config_path: str) -> 'AppConfig':
        if not os.path.exists(config_path):
            print(f"Error: Configuration file '{config_path}' not found.")
            exit(1)
        with open(config_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return cls(
            project=ProjectConfig(**data['project']),
            model=ModelConfig(**data['model']),
            processing=ProcessingConfig(**data['processing'])
        )

# --- 2. SERVICE FUNCTIONS ---

def check_model_availability(model_name: str):
    spinner = Halo(text=f'Verifying model access: {model_name}...', spinner='dots')
    spinner.start()
    try:
        found = False
        for m in genai.list_models():
            if m.name == model_name or m.name.endswith(f"/{model_name}"):
                found = True
                break
        if found: spinner.succeed(f"Model '{model_name}' is valid.")
        else: 
            spinner.fail(f"Model '{model_name}' not found.")
            exit(1)
    except Exception as e:
        spinner.fail(f"API Connection Error: {e}")
        exit(1)

def extract_and_report(zip_path: str, output_path: str, report_path: str, cfg: ProcessingConfig):
    valid_exts = tuple(cfg.valid_extensions)
    include_names = set(n.lower() for n in cfg.include_filenames)
    ignore_dirs = cfg.ignore_dirs
    COLLAPSE_THRESHOLD = 50 
    
    skipped_no_ext = []
    skipped_by_ext = {}

    with open(output_path, 'w', encoding='utf-8') as out_f, \
         zipfile.ZipFile(zip_path, 'r') as z:
        
        for filename in z.namelist():
            if filename.endswith('/') or any(d in filename for d in ignore_dirs):
                continue
            
            is_valid = filename.lower().endswith(valid_exts) or \
                       os.path.basename(filename).lower() in include_names
            
            if is_valid:
                try:
                    content = z.read(filename).decode('utf-8')
                    out_f.write(f"--- START FILE: {filename} ---\n{content}\n--- END FILE: {filename} ---\n\n")
                except Exception:
                    ext = os.path.splitext(filename)[1].lower()
                    key = f"{filename} (Decode Error)"
                    if ext: skipped_by_ext.setdefault(ext, []).append(key)
                    else: skipped_no_ext.append(key)
            else:
                ext = os.path.splitext(filename)[1].lower()
                if not ext: skipped_no_ext.append(filename)
                else: skipped_by_ext.setdefault(ext, []).append(filename)

    # Initial Report Writing (Mode 'w')
    with open(report_path, 'w', encoding='utf-8') as rep:
        rep.write(f"--- EXECUTION REPORT ---\n")
        rep.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        rep.write(f"Source: {os.path.basename(zip_path)}\n\n")
        
        rep.write(f"--- SKIPPED FILES (Noise Reduction) ---\n")
        rep.write(f"Ignored Folders: {', '.join(ignore_dirs)}\n")
        
        if skipped_no_ext:
            rep.write(f"Misc / No Extension ({len(skipped_no_ext)}):\n")
            for f in sorted(skipped_no_ext): rep.write(f" - {f}\n")
            
        for ext, files in sorted(skipped_by_ext.items()):
            count = len(files)
            if count > COLLAPSE_THRESHOLD:
                rep.write(f" - {ext}: {count} files omitted\n")
            else:
                rep.write(f" - {ext} ({count}):\n")
                for f in sorted(files): rep.write(f"    * {f}\n")
        rep.write("\n")

def append_inference_stats(report_path: str, response, duration_sec: float, model_name: str):
    """
    Appends API usage statistics to the existing report.
    """
    with open(report_path, 'a', encoding='utf-8') as rep:
        rep.write(f"--- INFERENCE STATISTICS ---\n")
        rep.write(f"Model: {model_name}\n")
        rep.write(f"Duration: {duration_sec:.2f} seconds\n")
        
        # Token Usage
        if hasattr(response, 'usage_metadata'):
            usage = response.usage_metadata
            rep.write(f"Token Usage:\n")
            rep.write(f"  - Input (Context):  {usage.prompt_token_count}\n")
            rep.write(f"  - Output (Gen):     {usage.candidates_token_count}\n")
            rep.write(f"  - Total:            {usage.total_token_count}\n")
        else:
            rep.write("Token Usage: Not available in response.\n")

        # Finish Reason
        if response.candidates:
            reason = response.candidates[0].finish_reason
            # Enum to string map if needed, but usually str(reason) is readable like FinishReason.STOP
            rep.write(f"Finish Reason: {reason.name}\n") 
        
        # Speed calculation
        if hasattr(response, 'usage_metadata') and duration_sec > 0:
            out_tokens = response.usage_metadata.candidates_token_count
            tps = out_tokens / duration_sec
            rep.write(f"Speed: {tps:.2f} tokens/sec (output)\n")

def get_or_upload_file(local_path: str):
    def get_hash(fp):
        h = hashlib.md5()
        with open(fp, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""): h.update(chunk)
        return h.hexdigest()

    file_hash = get_hash(local_path)
    target_display_name = f"repo_context_{file_hash}"
    
    with Halo(text='Checking Gemini cache...', spinner='dots'):
        for f in genai.list_files():
            if f.display_name == target_display_name:
                return f, True

    spinner = Halo(text='Uploading to Gemini...', spinner='dots')
    spinner.start()
    try:
        uploaded = genai.upload_file(path=local_path, mime_type="text/plain", display_name=target_display_name)
        while uploaded.state.name == "PROCESSING":
            time.sleep(2)
            uploaded = genai.get_file(uploaded.name)
        if uploaded.state.name == "FAILED": raise ValueError("Upload failed")
        spinner.succeed("Upload complete.")
        return uploaded, False
    except Exception as e:
        spinner.fail(f"Upload error: {e}")
        raise e

def save_generated_files(response_text: str, target_dir: str):
    matches = re.findall(r"--- START OUTPUT: (.*?) ---\n(.*?)--- END OUTPUT: \1 ---", response_text, re.DOTALL)
    if not matches:
        print("\n--- NO FILES DETECTED ---")
        return

    print(f"\nFound {len(matches)} file(s). Saving to '{target_dir}'...")
    safe_dir = os.path.abspath(target_dir)

    for filename, content in matches:
        filename = filename.strip()
        full_path = os.path.abspath(os.path.join(safe_dir, filename))
        if not full_path.startswith(safe_dir): continue # Security skip
            
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        try:
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content.strip())
            print(f"  Saved: {filename}")
        except Exception as e:
            print(f"  Error: {e}")

# --- 3. MAIN EXECUTION ---

def main():
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key: exit(1)
    genai.configure(api_key=api_key)

    config = AppConfig.load("config.yaml")

    for p in [config.project.zip_path, config.project.prompt_file, config.project.system_prompt_file]:
        if not os.path.exists(p):
            print(f"Error: {p} not found"); exit(1)

    check_model_availability(config.model.name)

    # Setup Directories
    zip_name = os.path.splitext(os.path.basename(config.project.zip_path))[0]
    timestamp = time.strftime("%Y%m%d-%H%M")
    run_dir = os.path.join(config.project.output_dir, f"{zip_name}-{timestamp}")
    report_path = os.path.join(run_dir, os.path.basename(config.project.report_file))

    os.makedirs(run_dir, exist_ok=True)
    print(f"\nOutput directory created: {run_dir}")

    # Prepare content
    with open(config.project.system_prompt_file, 'r') as f: sys_prompt = f.read()
    with open(config.project.prompt_file, 'r') as f: user_prompt = f.read()

    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.txt', encoding='utf-8') as tmp:
        temp_txt = tmp.name

    try:
        # 1. Extract & Initial Report
        with Halo(text=f'Processing ZIP...', spinner='dots'):
            extract_and_report(config.project.zip_path, temp_txt, report_path, config.processing)

        # 2. Upload/Cache
        gemini_file, is_cached = get_or_upload_file(temp_txt)
        if is_cached: print(f"Using cached context: {gemini_file.display_name}")

        # 3. Inference & Timing
        spinner = Halo(text=f'Generating with {config.model.name}...', spinner='dots')
        spinner.start()
        start_time = time.time()
        try:
            model = genai.GenerativeModel(config.model.name)
            response = model.generate_content(
                [gemini_file, f"{sys_prompt}\n\nUSER REQUEST:\n{user_prompt}"],
                request_options={"timeout": config.model.timeout}
            )
            end_time = time.time()
            spinner.succeed("Generation complete!")
        except Exception as e:
            spinner.fail(f"Generation failed: {e}")
            exit(1)

        # 4. Append Stats & Save Files
        duration = end_time - start_time
        append_inference_stats(report_path, response, duration, config.model.name)
        print(f"Report updated: {report_path}")
        
        save_generated_files(response.text, run_dir)

    finally:
        if os.path.exists(temp_txt): os.remove(temp_txt)

if __name__ == "__main__":
    main()
