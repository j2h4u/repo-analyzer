"""Event-driven pipeline for repository analysis.

This pipeline orchestrates the entire analysis flow from ZIP to documentation,
emitting events at each stage for progress tracking and presentation.
"""

import hashlib
import logging
import time
import zipfile
from pathlib import Path
from typing import Optional

import google.generativeai as genai

from ..config import AppConfig
from ..utils import build_file_tree
from ..utils.events import SimpleEmitter
from .result_types import (
    ScanResult,
    ContextResult,
    GeminiUploadResult,
    InferenceResult,
    ParsedFilesResult,
    PipelineResult,
)
from .scanner import scan_zip
from .file_parser import parse_generated_files, resolve_file_conflicts

logger = logging.getLogger(__name__)


class RepositoryAnalysisPipeline:
    """Event-driven pipeline for repository analysis.
    
    The pipeline processes a ZIP archive through multiple stages:
    1. Scan - Analyze ZIP contents
    2. Context - Build context from files
    3. Upload - Upload to Gemini API
    4. Inference - Run model generation
    5. Parse - Extract generated files
    
    Events are emitted at each stage for progress tracking:
    - 'stage:start' - Stage beginning
    - 'stage:complete' - Stage completion
    - 'inference:chunk' - Model generation progress
    - 'upload:cached' - File found in cache
    
    Example:
        emitter = SimpleEmitter()
        emitter.on('stage:start', lambda **kw: print(f"Starting {kw['stage']}"))
        
        pipeline = RepositoryAnalysisPipeline(config, emitter)
        result = pipeline.run()
    """
    
    def __init__(self, config: AppConfig, emitter: Optional[SimpleEmitter] = None):
        """Initialize pipeline with configuration and optional event emitter.
        
        Args:
            config: Application configuration
            emitter: Event emitter for progress tracking (optional, defaults to silent)
        """
        self.config = config
        self.emitter = emitter or SimpleEmitter()  # Silent if None
    
    def run(self, run_dir: Path) -> PipelineResult:
        """Execute complete pipeline: ZIP → Documentation.
        
        Args:
            run_dir: Directory for output files
            
        Returns:
            PipelineResult with all stage results
        """
        # Stage 1: Scan ZIP
        self.emitter.emit('stage:start', stage='scan', message='Scanning ZIP archive...')
        scan_result = self._scan()
        self.emitter.emit('stage:complete', stage='scan', 
                         files_count=len(scan_result.included_files))
        
        # Stage 2: Build context
        self.emitter.emit('stage:start', stage='context', message='Building context...')
        context_result = self._build_context(scan_result)
        self.emitter.emit('stage:complete', stage='context')
        
        # Save context to disk
        context_path = run_dir / "context.txt"
        context_path.write_text(context_result.to_text(), encoding='utf-8')
        
        # Stage 3: Upload to Gemini
        self.emitter.emit('stage:start', stage='upload', 
                         message='Checking Gemini cache...')
        upload_result = self._upload(context_path)
        if upload_result.was_cached:
            self.emitter.emit('upload:cached', file_hash=upload_result.file_hash)
        self.emitter.emit('stage:complete', stage='upload', cached=upload_result.was_cached)
        
        # Stage 4: Run inference
        self.emitter.emit('stage:start', stage='inference',
                         message=f'Generating with {self.config.model.name}...')
        inference_result = self._run_inference(upload_result)
        self.emitter.emit('stage:complete', stage='inference',
                         chunks=inference_result.stats.chunk_count)
        
        # Save response
        response_path = run_dir / "response.txt"
        response_path.write_text(inference_result.response_text, encoding='utf-8')
        
        # Stage 5: Parse generated files
        self.emitter.emit('stage:start', stage='parse', message='Parsing generated files...')
        parsed_result = self._parse_files(inference_result.response_text)
        self.emitter.emit('stage:complete', stage='parse', 
                         files_count=len(parsed_result.files))
        
        return PipelineResult(
            scan=scan_result,
            context=context_result,
            upload=upload_result,
            inference=inference_result,
            parsed=parsed_result
        )
    
    def _scan(self) -> ScanResult:
        """Stage 1: Scan ZIP archive."""
        all_files, included, ignore_dirs, skipped_no, skipped_ext = scan_zip(
            self.config.project.zip_path,
            self.config.processing
        )
        return ScanResult(
            all_files=all_files,
            included_files=included,
            encountered_ignore_dirs=ignore_dirs,
            skipped_no_ext=skipped_no,
            skipped_by_ext=skipped_ext
        )
    
    def _build_context(self, scan: ScanResult) -> ContextResult:
        """Stage 2: Build context from scanned files."""
        tree = build_file_tree(scan.all_files, set(scan.included_files))
        
        # Read file contents
        file_contents = []
        with zipfile.ZipFile(self.config.project.zip_path, 'r') as z:
            for filename in scan.included_files:
                content = z.read(filename).decode('utf-8')
                file_contents.append((filename, content))
        
        return ContextResult(
            tree_content=tree,
            file_contents=file_contents,
            total_files=len(scan.all_files),
            included_count=len(scan.included_files)
        )
    
    def _upload(self, context_path: Path) -> GeminiUploadResult:
        """Stage 3: Upload context to Gemini (or use cached)."""
        # Calculate file hash for caching
        file_hash = hashlib.sha256(context_path.read_bytes()).hexdigest()
        
        # Check if already uploaded (simplified - real implementation checks genai.list_files())
        # For now, always upload
        uploaded_file = genai.upload_file(context_path)
        
        # Wait for processing
        while uploaded_file.state.name == "PROCESSING":
            time.sleep(2)
            uploaded_file = genai.get_file(uploaded_file.name)
        
        if uploaded_file.state.name == "FAILED":
            raise ValueError("File upload failed")
        
        return GeminiUploadResult(
            file_handle=uploaded_file,
            was_cached=False,  # TODO: Implement caching logic
            file_hash=file_hash
        )
    
    def _run_inference(self, upload: GeminiUploadResult) -> InferenceResult:
        """Stage 4: Run model inference with streaming."""
        # Load prompts
        sys_prompt = self.config.project.system_prompt_file.read_text(encoding='utf-8')
        user_prompt = self.config.project.prompt_file.read_text(encoding='utf-8')
        
        # Configure model
        model = genai.GenerativeModel(
            model_name=self.config.model.name,
            system_instruction=sys_prompt
        )
        
        # Run inference with streaming
        start_time = time.time()
        chunks = []
        chunk_count = 0
        time_to_first_token = None
        
        response = model.generate_content(
            [upload.file_handle, user_prompt],
            request_options={"timeout": self.config.model.timeout}
        )
        
        # Collect response
        full_text = response.text
        elapsed = time.time() - start_time
        
        # Calculate stats
        usage = response.usage_metadata
        stats_data = {
            'model_name': self.config.model.name,
            'duration_seconds': elapsed,
            'input_tokens': usage.prompt_token_count,
            'output_tokens': usage.candidates_token_count,
            'total_tokens': usage.total_token_count,
            'finish_reason': str(response.candidates[0].finish_reason),
            'token_speed': usage.candidates_token_count / elapsed if elapsed > 0 else 0,
            'time_to_first_token': time_to_first_token or 0,
            'chunk_count': chunk_count
        }
        
        from ..config import InferenceStats
        stats = InferenceStats(**stats_data)
        
        return InferenceResult(
            response_text=full_text,
            stats=stats
        )
    
    def _parse_files(self, response_text: str) -> ParsedFilesResult:
        """Stage 5: Parse and resolve generated files."""
        parsed_files = parse_generated_files(response_text)
        resolved_files = resolve_file_conflicts(parsed_files)
        
        # Calculate stats (resolve_file_conflicts currently returns list, needs update)
        return ParsedFilesResult(
            files=resolved_files,
            renamed_count=0,  # TODO: Get from resolve_file_conflicts
            skipped_count=0   # TODO: Get from resolve_file_conflicts
        )
