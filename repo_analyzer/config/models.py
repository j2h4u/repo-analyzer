"""Configuration models for the repository analyzer."""

import logging
from dataclasses import dataclass, fields
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


# --- CUSTOM EXCEPTIONS ---

class RepoAnalyzerError(Exception):
    """Base exception for repo analyzer errors."""


class ConfigError(RepoAnalyzerError):
    """Configuration loading error."""


# --- CONFIGURATION DATACLASSES ---

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


# --- HELPER FUNCTIONS ---

def safe_load_dataclass(dclass_type, data: dict, section_name: str):
    """Safely load a dataclass from a dictionary.
    
    Ignores unknown keys and logs warnings for them.
    
    Args:
        dclass_type: Dataclass type to instantiate
        data: Dictionary with configuration data
        section_name: Name of config section (for logging)
        
    Returns:
        Instance of dclass_type with filtered data
    """
    valid_keys = {f.name for f in fields(dclass_type)}
    filtered_data = {}

    for k, v in data.items():
        if k in valid_keys:
            filtered_data[k] = v
        else:
            logger.warning(
                "Config warning: Unknown key '%s' in section '%s' ignored.",
                k, section_name
            )

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
        """Load application configuration from a YAML file.
        
        Args:
            config_path: Path to config.yaml file
            
        Returns:
            AppConfig instance with loaded configuration
            
        Raises:
            ConfigError: If file not found or YAML parsing fails
        """
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
            processing=safe_load_dataclass(
                ProcessingConfig, data['processing'], 'processing'
            ),
            logging=LoggingConfig(**data.get('logging', {'level': 'INFO'}))
        )
