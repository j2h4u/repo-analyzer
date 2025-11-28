"""Result types for pipeline data transfer between stages."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import InferenceStats


@dataclass
class ScanResult:
    """Result of scanning a ZIP archive."""
    all_files: list[str]
    included_files: list[str]
    encountered_ignore_dirs: set[str]
    skipped_no_ext: list[str]
    skipped_by_ext: dict[str, list[str]]


@dataclass
class ContextResult:
    """Result of building context from scanned files."""
    tree_content: str
    file_contents: list[tuple[str, str]]  # [(filename, content), ...]
    total_files: int
    included_count: int
    
    def to_text(self) -> str:
        """Combine tree and file contents into single text for context file."""
        lines = ["=== PROJECT FILE TREE ===\n"]
        lines.append(self.tree_content)
        lines.append(f"\n\nTotal files in archive: {self.total_files}")
        lines.append(f"\nAttached files: {self.included_count}")
        lines.append(f"\nNot attached: {self.total_files - self.included_count}\n")
        lines.append("\n" + "="*50 + "\n\n")
        
        for filename, content in self.file_contents:
            lines.append(f"--- START FILE: {filename} ---\n")
            lines.append(content)
            lines.append(f"\n--- END FILE: {filename} ---\n\n")
        
        return "".join(lines)


@dataclass
class GeminiUploadResult:
    """Result of uploading file to Gemini."""
    file_handle: Any  # genai.File
    was_cached: bool
    file_hash: str


@dataclass  
class InferenceResult:
    """Result of model inference."""
    response_text: str
    stats: InferenceStats


@dataclass
class ParsedFilesResult:
    """Result of parsing and resolving generated files."""
    files: list[Any]  # list[GeneratedFile] - imported from file_parser
    renamed_count: int
    skipped_count: int


@dataclass
class PipelineResult:
    """Complete result of repository analysis pipeline."""
    scan: ScanResult
    context: ContextResult
    upload: GeminiUploadResult
    inference: InferenceResult
    parsed: ParsedFilesResult
