"""Core business logic for repository analysis."""

from .scanner import scan_zip
from .file_parser import GeneratedFile, parse_generated_files, resolve_file_conflicts
from .result_types import (
    ScanResult, ContextResult, GeminiUploadResult,
    InferenceResult, ParsedFilesResult, PipelineResult
)
from .pipeline import RepositoryAnalysisPipeline

__all__ = [
    'scan_zip',
    'GeneratedFile',
    'parse_generated_files',
    'resolve_file_conflicts',
    'ScanResult',
    'ContextResult',
    'GeminiUploadResult',
    'InferenceResult',
    'ParsedFilesResult',
    'PipelineResult',
    'RepositoryAnalysisPipeline',
]
