#!/usr/bin/env python3
"""Repository analyzer tool that processes zip archives and generates documentation using Gemini AI.

This script extracts and analyzes code from a zip archive, uploads it to Google's Gemini API,
and generates comprehensive documentation based on configurable prompts.
"""

import hashlib
import logging
import os
import re
import queue
import sys
import threading
import time
import zipfile
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from google.api_core import exceptions as google_exceptions
import google.generativeai as genai
from halo import Halo

# Logger will be configured in main() after loading config
logger = logging.getLogger(__name__)

# --- CONSTANTS ---
COLLAPSE_THRESHOLD = 50

# Output filenames
CONTEXT_FILENAME = "context.txt"
REPORT_FILENAME = "report.txt"
RESPONSE_FILENAME = "response.txt"
DEBUG_LOG_FILENAME = "debug.log"

# --- CUSTOM EXCEPTIONS ---
class RepoAnalyzerError(Exception):
    """Base exception for repo analyzer errors."""

class ConfigError(RepoAnalyzerError):
    """Configuration loading error."""

# --- 1. CONFIGURATION STRUCTS ---

@dataclass
class ProjectConfig:
    """Configuration for project-specific paths."""
    zip_path: Path
    prompt_file: Path
    system_prompt_file: Path
    output_dir: Path
    report_file: str

@dataclass
class ModelConfig:
    """Configuration for AI model settings."""
    name: str
    timeout: int
    validate_model: bool
    chunk_timeout: int = 60

@dataclass
class ProcessingConfig:
    """Configuration for file processing and filtering."""
    valid_extensions: list[str]
    include_filenames: list[str]
    ignore_dirs: list[str]

@dataclass
class InferenceStats:
    """Statistics collected during model inference."""
    model_name: str
    duration_seconds: float
    input_tokens: int
    output_tokens: int
    total_tokens: int
    finish_reason: str
    token_speed: float
    time_to_first_token: float
    chunk_count: int

@dataclass
class LoggingConfig:
    """Configuration for logging behavior."""
    level: str = "INFO"

def safe_load_dataclass(dclass_type, data: dict, section_name: str):
    """
    Safely loads a dataclass from a dictionary, ignoring unknown keys
    and logging warnings for them.
    """
    valid_keys = {f.name for f in fields(dclass_type)}
    filtered_data = {}

    for k, v in data.items():
        if k in valid_keys:
            filtered_data[k] = v
        else:
            logger.warning("Config warning: Unknown key '%s' in section '%s' ignored.", k, section_name)

    return dclass_type(**filtered_data)

@dataclass
class AppConfig:
    """Main application configuration container."""
    project: ProjectConfig
    model: ModelConfig
    processing: ProcessingConfig
    logging: LoggingConfig

    @classmethod
    def load(cls, config_path: Path | str) -> 'AppConfig':
        """Load application configuration from a YAML file."""
        path = Path(config_path)
        if not path.exists():
            raise ConfigError(f"Configuration file '{path}' not found.")

        try:
            with path.open('r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigError(f"Error parsing YAML: {e}") from e

        return cls(
            project=ProjectConfig(
                zip_path=Path(data['project']['zip_path']),
                prompt_file=Path(data['project']['prompt_file']),
                system_prompt_file=Path(data['project']['system_prompt_file']),
                output_dir=Path(data['project']['output_dir']),
                report_file=data['project']['report_file']
            ),
            model=safe_load_dataclass(ModelConfig, data['model'], 'model'),
            processing=safe_load_dataclass(ProcessingConfig, data['processing'], 'processing'),
            logging=LoggingConfig(**data.get('logging', {'level': 'INFO'}))
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

def print_error(msg: str, indent: int = 0):
    """Print error message withicon"""
    prefix = " " * indent
    print(f"{prefix}❗️{msg}")

def print_warning(msg: str, indent: int = 0):
    """Print warning message with icon"""
    prefix = " " * indent
    print(f"{prefix}⚠️{msg}")

def print_info(msg: str, indent: int = 2):
    """Print info message with default 2-space indent"""
    prefix = " " * indent
    print(f"{prefix}{msg}")

def build_file_tree(all_files: list[str], included_files: set[str]) -> str:
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
            is_last_item = idx == len(items) - 1

            if isinstance(value, dict):
                # It's a directory
                lines.extend(render(value, prefix, key, is_last_item))
            else:
                # It's a file
                connector = "└── " if is_last_item else "├── "
                lines.append(f"{prefix}{connector}{key}{value}")

        return lines

    return "\n".join(render(tree))

# --- 3. CORE LOGIC ---

def check_model_availability(model_name: str):
    """Check if the specified Gemini model is available via the API."""
    spinner = Halo(text=f'Verifying model access: {model_name}...', spinner='dots')
    spinner.start()
    try:
        found = False
        for m in genai.list_models():
            if m.name == model_name or m.name.endswith(f"/{model_name}"):
                found = True
                break
        if found:
            spinner.succeed(f"Model '{model_name}' is valid.")
        else:
            spinner.fail(f"Model '{model_name}' not found.")
            raise ValueError(f"Model '{model_name}' not found.")
    except Exception as e:
        spinner.fail(f"API Connection Error: {e}")
        raise ConnectionError(f"API Connection Error: {e}") from e

def scan_zip(
    zip_path: Path,
    cfg: ProcessingConfig
) -> tuple[list[str], list[str], set[str], list[str], dict[str, list[str]]]:
    """Scans zip file and returns file lists and stats."""
    valid_exts = tuple(cfg.valid_extensions)
    include_names = set(n.lower() for n in cfg.include_filenames)
    ignore_dirs = cfg.ignore_dirs

    skipped_no_ext = []
    skipped_by_ext = {}
    encountered_ignore_dirs = set()
    included_files = []
    all_files = []

    with zipfile.ZipFile(zip_path, 'r') as z:
        for filename in z.namelist():
            if filename.endswith('/'):
                continue

            matched_ignore = None
            for d in ignore_dirs:
                if d in filename:
                    matched_ignore = d
                    break

            if matched_ignore:
                encountered_ignore_dirs.add(matched_ignore)
                continue

            all_files.append(filename)

            is_valid = filename.lower().endswith(valid_exts) or \
                       Path(filename).name.lower() in include_names

            if is_valid:
                try:
                    z.read(filename).decode('utf-8')
                    included_files.append(filename)
                except Exception:
                    ext = Path(filename).suffix.lower()
                    key = filename + " (Decode Error)"
                    if ext:
                        skipped_by_ext.setdefault(ext, []).append(key)
                    else:
                        skipped_no_ext.append(key)
            else:
                ext = Path(filename).suffix.lower()
                if not ext:
                    skipped_no_ext.append(filename)
                else:
                    skipped_by_ext.setdefault(ext, []).append(filename)

    return all_files, included_files, encountered_ignore_dirs, skipped_no_ext, skipped_by_ext

def write_context(
    output_path: Path,
    zip_path: Path,
    all_files: list[str],
    included_files: list[str]
) -> None:
    """Writes the context file with tree and file contents."""
    with output_path.open('w', encoding='utf-8') as out_f:
        out_f.write("=== PROJECT FILE TREE ===\n\n")
        tree_str = build_file_tree(all_files, set(included_files))
        out_f.write(tree_str)
        out_f.write(f"\n\nTotal files in archive: {len(all_files)}")
        out_f.write(f"\nAttached files: {len(included_files)}")
        out_f.write(f"\nNot attached: {len(all_files) - len(included_files)}\n")
        out_f.write("\n" + "="*50 + "\n\n")

        with zipfile.ZipFile(zip_path, 'r') as z:
            for filename in included_files:
                content = z.read(filename).decode('utf-8')
                out_f.write(
                    f"--- START FILE: {filename} ---\n{content}\n"
                    f"--- END FILE: {filename} ---\n\n"
                )

def write_report(
    report_path: Path,
    zip_name: str,
    ignore_dirs: set[str],
    skipped_no_ext: list[str],
    skipped_by_ext: dict[str, list[str]]
) -> None:
    """Writes the execution report."""
    with report_path.open('w', encoding='utf-8') as rep:
        rep.write("--- EXECUTION REPORT ---\n")
        rep.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        rep.write(f"Source: {zip_name}\n\n")

        rep.write("--- SKIPPED FILES (Noise Reduction) ---\n")

        if ignore_dirs:
            sorted_dirs = sorted([f"{d}/" for d in ignore_dirs])
            rep.write(f"Ignored Folders: {', '.join(sorted_dirs)}\n")
        else:
            rep.write("Ignored Folders: None encountered\n")

        if skipped_no_ext:
            rep.write(f"Misc / No Extension ({len(skipped_no_ext)}):\n")
            for f in sorted(skipped_no_ext):
                rep.write(f" - {f}\n")

        for ext, files in sorted(skipped_by_ext.items()):
            count = len(files)
            if count > COLLAPSE_THRESHOLD:
                rep.write(f" - {ext}: {count} files omitted\n")
            else:
                rep.write(f" - {ext} ({count}):\n")
                for f in sorted(files):
                    rep.write(f"    * {f}\n")
        rep.write("\n")

def extract_and_report(
    zip_path: Path,
    output_path: Path,
    report_path: Path,
    cfg: ProcessingConfig
) -> None:
    """Extract context and generate report from zip archive.

    Combines scan_zip, write_context, and write_report to process a zip archive.
    """
    all_files, included_files, ignore_dirs, skipped_no, skipped_ext = scan_zip(zip_path, cfg)

    write_context(output_path, zip_path, all_files, included_files)
    write_report(report_path, zip_path.name, ignore_dirs, skipped_no, skipped_ext)

def append_inference_stats(report_path: Path, stats: InferenceStats) -> None:
    """
    Appends clean, human-readable stats to the report.
    """
    with report_path.open('a', encoding='utf-8') as rep:
        rep.write("--- INFERENCE STATS ---\n")
        rep.write(f"Model: {stats.model_name}\n")
        rep.write(f"Chunks Received: {stats.chunk_count}\n")
        rep.write(f"Time to First Token: {format_duration(stats.time_to_first_token)}\n")
        rep.write(f"Total time: {format_duration(stats.duration_seconds)}\n")

        # 1. Token Usage
        # Visual alignment using fixed width (:>6)
        rep.write("Token Usage:\n")
        rep.write(
            f"  - Input (context):  {format_token_count(stats.input_tokens):>6} "
            f"({stats.input_tokens})\n"
        )
        rep.write(
            f"  - Output (gen):     {format_token_count(stats.output_tokens):>6} "
            f"({stats.output_tokens})\n"
        )
        rep.write(
            f"  - Total:            {format_token_count(stats.total_tokens):>6} "
            f"({stats.total_tokens})\n"
        )

        if stats.duration_seconds > 0:
            rep.write(f"Token Speed: {int(stats.token_speed)} tokens/sec (output)\n")
        else:
            rep.write("Token Speed: N/A\n")

        # Finish Reason
        if stats.finish_reason != "STOP":
            rep.write(f"Finish Reason: {stats.finish_reason}\n")

def get_or_upload_file(local_path: Path) -> tuple[Any, bool]:
    """Get file from Gemini cache or upload if not present."""
    def get_hash(fp):
        h = hashlib.md5()
        with open(fp, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                h.update(chunk)
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
        uploaded = genai.upload_file(
            path=local_path,
            mime_type="text/plain",
            display_name=target_display_name
        )
        while uploaded.state.name == "PROCESSING":
            time.sleep(2)
            uploaded = genai.get_file(uploaded.name)
        if uploaded.state.name == "FAILED":
            raise ValueError("Upload failed")
        spinner.succeed("Upload complete.")
        return uploaded, False
    except Exception as e:
        spinner.fail(f"Upload error: {e}")
        raise e

@dataclass
class GeneratedFile:
    """Represents a file generated by the LLM."""
    filename: str
    content: str

def parse_generated_files(response_text: str) -> list[GeneratedFile]:
    """
    Parses LLM response text and extracts generated files.

    Args:
        response_text: Raw text response from the LLM.

    Returns:
        List of GeneratedFile objects with filename and content.
    """
    pattern = r"--- START OUTPUT: (.*?) ---\n(.*?)--- END OUTPUT: \1 ---"
    matches = re.findall(pattern, response_text, re.DOTALL)

    return [
        GeneratedFile(filename=filename.strip(), content=content.strip())
        for filename, content in matches
    ]

def resolve_file_conflicts(files: list[GeneratedFile]) -> list[GeneratedFile]:
    """
    Resolves filename conflicts in-memory.

    - Removes exact duplicates (same filename + content)
    - Renames files with same name but different content (file.1.md, file.2.md)
    - Preserves directory structure

    Args:
        files: List of GeneratedFile objects from parsing.

    Returns:
        List of unique GeneratedFile objects ready to save.
    """
    # Track: filename -> list of (content, index in result)
    seen: dict[str, list[tuple[str, int]]] = {}
    result: list[GeneratedFile] = []

    # Statistics
    renamed_count = 0
    skipped_count = 0

    for file in files:
        filename = file.filename
        content = file.content

        if filename not in seen:
            # First occurrence of this filename
            seen[filename] = [(content, len(result))]
            result.append(file)
            logger.debug("Added file: %s", filename)
        else:
            # Filename already seen, check content
            found_duplicate = False

            for seen_content, _ in seen[filename]:
                if seen_content == content:
                    # Exact duplicate (same name + content), skip
                    logger.info("Skipped exact duplicate: %s", filename)
                    skipped_count += 1
                    found_duplicate = True
                    break

            if not found_duplicate:
                # Same name, different content - need to rename
                relative_path = Path(filename)
                parent_dir = relative_path.parent
                base = relative_path.stem
                ext = relative_path.suffix

                # Find next available number
                counter = 1
                while True:
                    new_name = f"{base}.{counter}{ext}"
                    new_filename = (
                        str(parent_dir / new_name)
                        if parent_dir != Path('.')
                        else new_name
                    )

                    # Check if this numbered name already exists in our seen list
                    if new_filename not in seen:
                        seen[new_filename] = [(content, len(result))]
                        result.append(GeneratedFile(filename=new_filename, content=content))
                        logger.info(
                            "Renamed conflicting file: %s -> %s",
                            filename,
                            new_filename
                        )
                        renamed_count += 1
                        break

                    # Check if numbered file has same content
                    is_dup = any(sc == content for sc, _ in seen[new_filename])
                    if is_dup:
                        logger.info(
                            "Skipped duplicate: %s (same as %s)",
                            filename,
                            new_filename
                        )
                        skipped_count += 1
                        found_duplicate = True
                        break

                    counter += 1

    logger.info(
        "Processed %d files → %d unique (%d renamed, %d duplicates skipped)",
        len(files),
        len(result),
        renamed_count,
        skipped_count
    )
    return result

def save_files_to_disk(files: list[GeneratedFile], target_dir: Path) -> None:
    """
    Saves a list of GeneratedFile objects to disk.

    Assumes files have already been deduplicated and conflict-resolved.

    Args:
        files: List of GeneratedFile objects to save.
        target_dir: Target directory to save files to.
    """
    if not files:
        logger.warning("No files to save")
        return

    logger.info("Saving %d files to '%s'", len(files), target_dir)
    safe_dir = target_dir.resolve()

    saved_count = 0
    error_count = 0

    for file in files:
        full_path = (safe_dir / file.filename).resolve()

        # Security check: prevent path traversal
        if not str(full_path).startswith(str(safe_dir)):
            logger.warning(
                "Path traversal attempt blocked: %s", file.filename
            )
            continue

        full_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with full_path.open('w', encoding='utf-8') as f:
                f.write(file.content)
            logger.debug("Saved: %s", file.filename)
            saved_count += 1
        except Exception as e:
            logger.error("Failed to save %s: %s", file.filename, e)
            error_count += 1

    if error_count > 0:
        logger.warning(
            "Saved %d/%d files (%d errors)", saved_count, len(files), error_count
        )
    else:
        logger.info("Successfully saved all %d files", saved_count)

# --- 4. MAIN EXECUTION ---

def setup_logging(run_dir: Path, config_log_level: str) -> None:
    """Configure logging with file and console handlers."""
    log_level = getattr(logging, config_log_level.upper(), logging.INFO)

    # Create formatters
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(funcName)s - %(message)s')
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(funcName)s - %(message)s',
        datefmt='%H:%M:%S'
    )

    # Create handlers
    file_handler = logging.FileHandler(run_dir / DEBUG_LOG_FILENAME, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_formatter)

    # Configure basic config with both handlers
    logging.basicConfig(
        level=logging.DEBUG,
        handlers=[file_handler, console_handler],
        force=True
    )

def initialize_app() -> tuple[AppConfig, Path]:
    """Initialize app configuration and create output directory."""
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print_error("GOOGLE_API_KEY not found in environment variables.")
        sys.exit(1)
    genai.configure(api_key=api_key)

    config = AppConfig.load("config.yaml")

    # Setup output directory
    zip_name = config.project.zip_path.stem
    timestamp = time.strftime("%Y%m%d-%H%M")
    run_dir = config.project.output_dir / f"{zip_name}-{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    return config, run_dir

def validate_files(config: AppConfig) -> None:
    """Validate that required files exist and optionally check model availability."""
    for p in [
        config.project.zip_path,
        config.project.prompt_file,
        config.project.system_prompt_file
    ]:
        if not p.exists():
            print_error(f"Error: {p} not found")
            sys.exit(1)

    if config.model.validate_model:
        check_model_availability(config.model.name)

def prepare_context(config: AppConfig, run_dir: Path) -> tuple[Path, Path]:
    """Prepare context file and return paths."""
    report_path = run_dir / config.project.report_file
    context_path = run_dir / CONTEXT_FILENAME

    # Create context file
    with Halo(text='Processing ZIP...', spinner='dots'):
        extract_and_report(config.project.zip_path, context_path, report_path, config.processing)

    print_info(f"Context saved: {context_path}")
    return context_path, report_path

def run_inference(
    model_config: ModelConfig,
    gemini_file: Any,
    sys_prompt: str,
    user_prompt: str
) -> tuple[str, InferenceStats]:
    """
    Run model inference with streaming enabled.

    Returns:
        tuple: (text, stats)
    """
    spinner = Halo(text=f'Generating with {model_config.name}...', spinner='dots')
    spinner.start()
    start_time = time.time()
    time_to_first_token = None
    chunk_count = 0
    full_response = None

    # Queues for passing data between threads
    chunk_queue = queue.Queue()
    error_queue = queue.Queue()

    def stream_consumer():
        """Background thread to consume the stream."""
        try:
            model = genai.GenerativeModel(model_config.name)
            response_stream = model.generate_content(
                [gemini_file, f"{sys_prompt}\n\nUSER REQUEST:\n{user_prompt}"],
                request_options={"timeout": model_config.timeout},
                stream=True
            )
            for chunk in response_stream:
                chunk_queue.put(chunk)
            chunk_queue.put(None)  # Sentinel for end of stream
        except Exception as e:
            error_queue.put(e)

    # Start background thread
    t = threading.Thread(target=stream_consumer, daemon=True)
    t.start()

    # Accumulate all text parts
    accumulated_text = []

    try:
        while True:
            # Check for errors from the thread
            if not error_queue.empty():
                raise error_queue.get()

            try:
                # Wait for next chunk with precise timeout
                chunk = chunk_queue.get(timeout=model_config.chunk_timeout)
            except queue.Empty:
                # Timeout hit!
                elapsed_total = int(time.time() - start_time)
                spinner.fail(
                    f"No chunks received for {model_config.chunk_timeout}s "
                    f"(Total elapsed: {elapsed_total}s)"
                )
                raise TimeoutError(
                    f"No activity from model for {model_config.chunk_timeout}s"
                )

            if chunk is None:  # End of stream
                break

            chunk_count += 1
            current_time = time.time()

            # Record time to first token
            if time_to_first_token is None:
                time_to_first_token = current_time - start_time
                logger.debug("Time to first token: %.2fs", time_to_first_token)

            # Update spinner with progress
            elapsed = int(current_time - start_time)
            spinner.text = (
                f'Generating with {model_config.name}... '
                f'chunk {chunk_count}, {elapsed}s'
            )

            # Accumulate text and keep last chunk for metadata
            if chunk.text:
                accumulated_text.append(chunk.text)
            full_response = chunk

        end_time = time.time()
        duration = end_time - start_time

        if full_response is None:
            spinner.fail("No response received from streaming API")
            raise ValueError("Empty streaming response")

        final_text = "".join(accumulated_text)

        # Extract statistics from the last chunk (full_response)
        input_tokens = 0
        output_tokens = 0
        total_tokens = 0
        finish_reason = "UNKNOWN"

        if hasattr(full_response, 'usage_metadata'):
            usage = full_response.usage_metadata
            input_tokens = usage.prompt_token_count
            output_tokens = usage.candidates_token_count
            total_tokens = usage.total_token_count

        if full_response.candidates:
            finish_reason = full_response.candidates[0].finish_reason.name

        token_speed = output_tokens / duration if duration > 0 else 0.0

        stats = InferenceStats(
            model_name=model_config.name,
            duration_seconds=duration,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            finish_reason=finish_reason,
            token_speed=token_speed,
            time_to_first_token=time_to_first_token or 0.0,
            chunk_count=chunk_count
        )

        spinner.succeed(f"Generation complete! ({chunk_count} chunks, {format_duration(duration)})")
        return final_text, stats

    except google_exceptions.DeadlineExceeded as e:
        end_time = time.time()
        elapsed = end_time - start_time
        spinner.fail(f"Generation timed out after {format_duration(elapsed)}")
        logger.debug("API Timeout details: %s", e)
        print_error(
            "The model took too long to respond (Timeout). "
            "Try increasing the timeout in config.yaml."
        )
        sys.exit(1)
    except TimeoutError:
        # Already logged and displayed by queue timeout handler
        sys.exit(1)
    except Exception as e:
        end_time = time.time()
        elapsed = end_time - start_time
        spinner.fail(f"Generation failed after {format_duration(elapsed)}: {e}")
        raise e

def process_response(
    response: str,
    run_dir: Path,
    report_path: Path,
    stats: InferenceStats
) -> None:
    """Process model response: log stats, save response, and extract files."""
    # Process stats
    logger.debug(stats)
    append_inference_stats(report_path, stats)
    print_info(f"Report updated: {report_path}")

    # Save raw model response
    response_path = run_dir / RESPONSE_FILENAME
    with response_path.open('w', encoding='utf-8') as f:
        f.write(response)
    print_info(f"Raw response saved: {response_path}")

    # Check if response has valid content
    if stats.finish_reason != "STOP":
        print_warning(f"Response finished with reason: {stats.finish_reason}")
    else:
        files = parse_generated_files(response)
        unique_files = resolve_file_conflicts(files)
        save_files_to_disk(unique_files, run_dir)

def main() -> None:
    """Main entry point for the repository analyzer."""
    try:
        config, run_dir = initialize_app()
        setup_logging(run_dir, config.logging.level)
        print_info(f"Output directory created: {run_dir}")

        validate_files(config)
        context_path, report_path = prepare_context(config, run_dir)

        gemini_file, is_cached = get_or_upload_file(context_path)
        if is_cached:
            print_info(f"Using cloud-cached context: {gemini_file.display_name}")

        sys_prompt = config.project.system_prompt_file.read_text(encoding='utf-8')
        user_prompt = config.project.prompt_file.read_text(encoding='utf-8')

        response, stats = run_inference(config.model, gemini_file, sys_prompt, user_prompt)

        process_response(response, run_dir, report_path, stats)

    except RepoAnalyzerError as e:
        print_error(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected Error: {e}")
        logger.debug("Unexpected error occurred", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
