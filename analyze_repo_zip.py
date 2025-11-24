#!/usr/bin/env python3

import os
import re
import zipfile
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

# --- 2. HELPERS FOR REPORTING ---

def format_token_count(count: int) -> str:
    """Converts 300619 -> '301k'"""
    if count >= 1_000_000:
        return f"{count/1_000_000:.1f}M"
    if count >= 1_000:
        return f"{count/1000:.0f}k"
    return str(count)

def format_duration(seconds: float) -> str:
    """Converts 147.5 -> '2m 27s'"""
    m, s = divmod(int(seconds), 60)
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"

# --- 3. CORE LOGIC ---

def build_file_tree(all_files: list, included_files: set) -> str:
    """
    Builds a tree-style directory structure similar to the `tree` command.
    Files not in included_files are marked with comment-like "# not attached".
    """
    # Build hierarchical structure
    tree = {}
    
    for filepath in all_files:
        parts = filepath.split('/')
        current = tree
        
        # Navigate through directories
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        
        # Add the file with marker
        filename = parts[-1]
        marker = "" if filepath in included_files else " # not attached"
        current[filename] = marker  # String value indicates it's a file
    
    # Render the tree
    def render(node, prefix="", name=".", is_last=True):
        lines = []
        
        if isinstance(node, str):
            # It's a file (leaf node)
            return []
        
        # First line is the directory name
        if name != ".":
            connector = "└── " if is_last else "├── "
            lines.append(f"{prefix}{connector}{name}/")
            prefix += "    " if is_last else "│   "
        else:
            lines.append(".")
        
        # Get all items (dirs and files)
        items = sorted(node.items())
        
        for idx, (key, value) in enumerate(items):
            is_last_item = (idx == len(items) - 1)
            
            if isinstance(value, dict):
                # It's a directory
                lines.extend(render(value, prefix, key, is_last_item))
            else:
                # It's a file
                connector = "└── " if is_last_item else "├── "
                lines.append(f"{prefix}{connector}{key}{value}")
        
        return lines
    
    return "\n".join(render(tree))

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
    encountered_ignore_dirs = set()
    included_files = []  # Track files that will be included
    all_files = []  # Track ALL files (for tree generation)

    with zipfile.ZipFile(zip_path, 'r') as z:
        # First pass: collect all files and determine which are included
        for filename in z.namelist():
            if filename.endswith('/'): continue

            # Logic to track WHICH ignored dir matched
            matched_ignore = None
            for d in ignore_dirs:
                if d in filename: # Simple substring check for dir
                    matched_ignore = d
                    break

            if matched_ignore:
                encountered_ignore_dirs.add(matched_ignore)
                continue

            # Add to all_files (not ignored)
            all_files.append(filename)

            is_valid = filename.lower().endswith(valid_exts) or \
                       os.path.basename(filename).lower() in include_names

            if is_valid:
                try:
                    # Test if file can be decoded
                    z.read(filename).decode('utf-8')
                    included_files.append(filename)
                except Exception:
                    ext = os.path.splitext(filename)[1].lower()
                    key = f"{filename} (Decode Error)"
                    if ext: skipped_by_ext.setdefault(ext, []).append(key)
                    else: skipped_no_ext.append(key)
            else:
                ext = os.path.splitext(filename)[1].lower()
                if not ext: skipped_no_ext.append(filename)
                else: skipped_by_ext.setdefault(ext, []).append(filename)

        # Second pass: write tree and file contents
        with open(output_path, 'w', encoding='utf-8') as out_f:
            # Write file tree (tree-style, showing all files)
            out_f.write("=== PROJECT FILE TREE ===\n\n")
            tree_str = build_file_tree(all_files, set(included_files))
            out_f.write(tree_str)
            out_f.write(f"\n\nTotal files in archive: {len(all_files)}")
            out_f.write(f"\nAttached files: {len(included_files)}")
            out_f.write(f"\nNot attached: {len(all_files) - len(included_files)}\n")
            out_f.write("\n" + "="*50 + "\n\n")

            # Write file contents
            for filename in included_files:
                content = z.read(filename).decode('utf-8')
                out_f.write(f"--- START FILE: {filename} ---\n{content}\n--- END FILE: {filename} ---\n\n")

    # Initial Report Writing
    with open(report_path, 'w', encoding='utf-8') as rep:
        rep.write(f"--- EXECUTION REPORT ---\n")
        rep.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        rep.write(f"Source: {os.path.basename(zip_path)}\n\n")

        rep.write(f"--- SKIPPED FILES (Noise Reduction) ---\n")

        if encountered_ignore_dirs:
            # Format with trailing slash
            sorted_dirs = sorted([f"{d}/" for d in encountered_ignore_dirs])
            rep.write(f"Ignored Folders: {', '.join(sorted_dirs)}\n")
        else:
            rep.write(f"Ignored Folders: None encountered\n")

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
    Appends clean, human-readable stats to the report.
    """
    with open(report_path, 'a', encoding='utf-8') as rep:
        rep.write(f"--- INFERENCE STATS ---\n")
        rep.write(f"Model: {model_name}\n")
        rep.write(f"Duration: {format_duration(duration_sec)}\n")

        # 1. Token Usage
        if hasattr(response, 'usage_metadata'):
            usage = response.usage_metadata
            in_tok = usage.prompt_token_count
            out_tok = usage.candidates_token_count
            total_tok = usage.total_token_count

            # Visual alignment using fixed width (:>6)
            rep.write(f"Token Usage:\n")
            rep.write(f"  - Input (context):  {format_token_count(in_tok):>6} ({in_tok})\n")
            rep.write(f"  - Output (gen):     {format_token_count(out_tok):>6} ({out_tok})\n")
            rep.write(f"  - Total:            {format_token_count(total_tok):>6} ({total_tok})\n")

            if duration_sec > 0:
                speed = out_tok / duration_sec
                rep.write(f"Token Speed: {int(speed)} tokens/sec (output)\n")
        else:
            rep.write("Token Usage: N/A\n")

        # Finish Reason
        if response.candidates:
            reason = response.candidates[0].finish_reason
            # Enum to string map if needed, but usually str(reason) is readable like FinishReason.STOP
            if reason.name != "STOP":
                rep.write(f"Finish Reason: {reason.name}\n")

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
        if not full_path.startswith(safe_dir): continue

        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        try:
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content.strip())
            print(f"  Saved: {filename}")
        except Exception as e:
            print(f"  Error: {e}")

# --- 4. MAIN EXECUTION ---

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
    context_path = os.path.join(run_dir, "context.txt")

    os.makedirs(run_dir, exist_ok=True)
    print(f"\nOutput directory created: {run_dir}")

    with open(config.project.system_prompt_file, 'r') as f: sys_prompt = f.read()
    with open(config.project.prompt_file, 'r') as f: user_prompt = f.read()

    # Create context file directly (no temp file)
    with Halo(text=f'Processing ZIP...', spinner='dots'):
        extract_and_report(config.project.zip_path, context_path, report_path, config.processing)
    
    print(f"Context saved: {context_path}")

    gemini_file, is_cached = get_or_upload_file(context_path)
    if is_cached: print(f"Using cloud-cached context: {gemini_file.display_name}")

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

    duration = end_time - start_time
    append_inference_stats(report_path, response, duration, config.model.name)
    print(f"Report updated: {report_path}")

    # Check if response has valid content before accessing .text
    if response.candidates and response.candidates[0].finish_reason.name != "STOP":
        reason = response.candidates[0].finish_reason.name
        print(f"\n⚠️ Response finished with reason: {reason}")
    else:
        save_generated_files(response.text, run_dir)

if __name__ == "__main__":
    main()

