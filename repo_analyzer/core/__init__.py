"""Core business logic for repository analysis."""

from .scanner import scan_zip
from .file_parser import GeneratedFile, parse_generated_files, resolve_file_conflicts

__all__ = [
    'scan_zip',
    'GeneratedFile',
    'parse_generated_files',
    'resolve_file_conflicts',
]
