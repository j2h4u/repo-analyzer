"""CLI presentation layer for repository analyzer.

This module handles all visual feedback in the CLI using Halo spinners
and subscribes to pipeline events for progress tracking.
"""

from halo import Halo
from typing import Optional

from ...utils.events import SimpleEmitter
from ...utils import format_token_count, format_duration


class CLIPresenter:
    """Displays pipeline progress in CLI with beautiful spinners.
    
    Subscribes to pipeline events and provides visual feedback:
    - Halo spinners for stages
    - Progress updates during inference
    - Success/failure messages
    - Info/error/warning messages
    
    Example:
        emitter = SimpleEmitter()
        presenter = CLIPresenter()
        presenter.attach_to_pipeline(emitter)
        
        # Pipeline will emit events, presenter will show spinners
        pipeline = RepositoryAnalysisPipeline(config, emitter)
        result = pipeline.run(run_dir)
    """
    
    def __init__(self):
        """Initialize CLI presenter."""
        self.current_spinner: Optional[Halo] = None
    
    def attach_to_pipeline(self, emitter: SimpleEmitter):
        """Subscribe to pipeline events.
        
        Args:
            emitter: Event emitter from pipeline
        """
        emitter.on('stage:start', self._on_stage_start)
        emitter.on('stage:complete', self._on_stage_complete)
        emitter.on('upload:cached', self._on_upload_cached)
        emitter.on('inference:chunk', self._on_inference_chunk)
    
    def _on_stage_start(self, stage: str, message: str, **_):
        """Handle stage start event - start spinner.
        
        Args:
            stage: Stage name (scan, context, upload, inference, parse)
            message: Message to display
        """
        if self.current_spinner:
            self.current_spinner.stop()
        
        self.current_spinner = Halo(text=message, spinner='dots')
        self.current_spinner.start()
    
    def _on_stage_complete(self, stage: str, **data):
        """Handle stage completion - show success message.
        
        Args:
            stage: Stage name
            **data: Additional data (files_count, chunks, cached, etc.)
        """
        if not self.current_spinner:
            return
        
        success_msg = self._format_success_message(stage, data)
        self.current_spinner.succeed(success_msg)
        self.current_spinner = None
    
    def _on_upload_cached(self, file_hash: str, **_):
        """Handle cached upload - update spinner text.
        
        Args:
            file_hash: Hash of cached file
        """
        if self.current_spinner:
            self.current_spinner.text = 'Using cached file...'
    
    def _on_inference_chunk(self, chunk: int, elapsed: float, **_):
        """Handle inference progress - update spinner.
        
        Args:
            chunk: Current chunk number
            elapsed: Elapsed time in seconds
        """
        if self.current_spinner:
            self.current_spinner.text = f'Generating... chunk {chunk}, {int(elapsed)}s elapsed'
    
    def _format_success_message(self, stage: str, data: dict) -> str:
        """Format success message based on stage and data.
        
        Args:
            stage: Stage name
            data: Stage-specific data
            
        Returns:
            Formatted success message
        """
        if stage == 'scan':
            count = data.get('files_count', 0)
            return f'Scan complete ({count} files included)'
        
        if stage == 'context':
            return 'Context built successfully'
        
        if stage == 'upload':
            if data.get('cached'):
                return 'Using cached file'
            return 'Upload complete'
        
        if stage == 'inference':
            chunks = data.get('chunks', 0)
            return f'Generation complete ({chunks} chunks)'
        
        if stage == 'parse':
            count = data.get('files_count', 0)
            return f'Parsed {count} generated files'
        
        return f'{stage.capitalize()} complete'
    
    @staticmethod
    def print_info(msg: str, indent: int = 2):
        """Print info message with indentation.
        
        Args:
            msg: Message to print
            indent: Number of spaces to indent
        """
        prefix = " " * indent
        print(f"{prefix}{msg}")
    
    @staticmethod
    def print_error(msg: str, indent: int = 0):
        """Print error message with icon.
        
        Args:
            msg: Error message
            indent: Number of spaces to indent
        """
        prefix = " " * indent
        print(f"{prefix}❗️{msg}")
    
    @staticmethod
    def print_warning(msg: str, indent: int = 0):
        """Print warning message with icon.
        
        Args:
            msg: Warning message
            indent: Number of spaces to indent
        """
        prefix = " " * indent
        print(f"{prefix}⚠️{msg}")
