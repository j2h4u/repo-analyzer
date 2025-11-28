"""ZIP archive scanning and analysis."""

import zipfile
from pathlib import Path

from repo_analyzer.config import ProcessingConfig


def scan_zip(
    zip_path: Path,
    cfg: ProcessingConfig
) -> tuple[list[str], list[str], set[str], list[str], dict[str, list[str]]]:
    """Scan ZIP file and return file lists and statistics.
    
    Args:
        zip_path: Path to ZIP archive
        cfg: Processing configuration with filters
        
    Returns:
        Tuple of:
        - all_files: All files found in archive
        - included_files: Files that passed filtering
        - encountered_ignore_dirs: Ignored directories that were found
        - skipped_no_ext: Files without extension that were skipped
        - skipped_by_ext: Files skipped, grouped by extension
    """
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
