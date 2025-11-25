import unittest
from pathlib import Path
from unittest.mock import patch, mock_open
import sys
import os

# Add parent directory to path to import analyze_repo_zip
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyze_repo_zip import (
    format_token_count,
    format_duration,
    build_file_tree,
    AppConfig,
    ConfigError
)

class TestHelpers(unittest.TestCase):
    def test_format_token_count(self):
        self.assertEqual(format_token_count(500), "500")
        self.assertEqual(format_token_count(1000), "1k")
        self.assertEqual(format_token_count(1500), "2k") # 1.5k -> 2k due to .0f rounding? Let's check impl.
        # Impl: f"{count/1000:.0f}k" -> 1.5 -> 2k (round half to even in Python 3? or standard round?)
        # 1500/1000 = 1.5. format(1.5, ".0f") -> "2".
        self.assertEqual(format_token_count(1000000), "1.0M")
        self.assertEqual(format_token_count(1200000), "1.2M")

    def test_format_duration(self):
        self.assertEqual(format_duration(30), "30s")
        self.assertEqual(format_duration(60), "1m 0s")
        self.assertEqual(format_duration(65), "1m 5s")
        self.assertEqual(format_duration(3600), "60m 0s")

    def test_build_file_tree(self):
        files = ["src/main.py", "src/utils/helper.py", "README.md"]
        included = {"src/main.py", "README.md"}
        
        tree = build_file_tree(files, included)
        
        # Check for key structural elements
        self.assertIn("src/", tree)
        self.assertIn("main.py", tree)
        self.assertIn("utils/", tree)
        self.assertIn("helper.py # not attached", tree)
        self.assertIn("README.md", tree)

class TestConfig(unittest.TestCase):
    def test_load_config_success(self):
        yaml_content = """
project:
  zip_path: "test.zip"
  prompt_file: "prompt.txt"
  system_prompt_file: "sys.txt"
  output_dir: "out"
  report_file: "rep.txt"
model:
  name: "gemini-pro"
  timeout: 60
  validate_model: true
processing:
  valid_extensions: [".py"]
  include_filenames: ["Dockerfile"]
  ignore_dirs: ["__pycache__"]
"""
        with patch("pathlib.Path.open", mock_open(read_data=yaml_content)):
            with patch("pathlib.Path.exists", return_value=True):
                config = AppConfig.load("config.yaml")
                self.assertEqual(config.project.zip_path, Path("test.zip"))
                self.assertEqual(config.model.name, "gemini-pro")
                self.assertEqual(config.processing.valid_extensions, [".py"])

    def test_load_config_not_found(self):
        with patch("pathlib.Path.exists", return_value=False):
            with self.assertRaises(ConfigError):
                AppConfig.load("missing.yaml")

if __name__ == '__main__':
    unittest.main()
