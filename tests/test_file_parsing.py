import unittest
import tempfile
import re
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyze_repo_zip import parse_generated_files, save_files_to_disk, GeneratedFile, resolve_file_conflicts


class TestFileParsing(unittest.TestCase):
    def setUp(self):
        """Create a temporary directory for test outputs."""
        self.test_dir = tempfile.mkdtemp()
        self.test_path = Path(self.test_dir)

    def tearDown(self):
        """Clean up temporary directory."""
        import shutil
        shutil.rmtree(self.test_dir)

    def test_parse_single_file(self):
        """Test that parse_generated_files correctly parses a single file."""
        response_text = """--- START OUTPUT: README.md ---
# Test File
This is a test.
--- END OUTPUT: README.md ---"""
        
        files = parse_generated_files(response_text)
        
        self.assertEqual(len(files), 1)
        self.assertIsInstance(files[0], GeneratedFile)
        self.assertEqual(files[0].filename, "README.md")
        self.assertIn("# Test File", files[0].content)

    def test_parse_multiple_files(self):
        """Test that parse_generated_files correctly parses multiple files."""
        response_text = """--- START OUTPUT: file1.py ---
print("file1")
--- END OUTPUT: file1.py ---

--- START OUTPUT: file2.py ---
print("file2")
--- END OUTPUT: file2.py ---

--- START OUTPUT: file3.py ---
print("file3")
--- END OUTPUT: file3.py ---"""
        
        files = parse_generated_files(response_text)
        
        self.assertEqual(len(files), 3, f"Expected 3 files, got {len(files)}")
        self.assertEqual(files[0].filename, "file1.py")
        self.assertEqual(files[1].filename, "file2.py")
        self.assertEqual(files[2].filename, "file3.py")
        self.assertIn('print("file1")', files[0].content)

    def test_parse_multiline_content(self):
        """Test parsing files with multiple lines."""
        response_text = """--- START OUTPUT: config.yaml ---
project:
  name: test
  version: 1.0
model:
  name: gemini
--- END OUTPUT: config.yaml ---"""
        
        files = parse_generated_files(response_text)
        
        self.assertEqual(len(files), 1)
        self.assertIn("project:", files[0].content)
        self.assertIn("model:", files[0].content)

    def test_save_single_file(self):
        """Test that save_files_to_disk correctly saves a single file."""
        files = [GeneratedFile(filename="test.txt", content="Hello World")]
        
        save_files_to_disk(files, self.test_path)
        
        saved_file = self.test_path / "test.txt"
        self.assertTrue(saved_file.exists())
        self.assertEqual(saved_file.read_text(), "Hello World")

    def test_save_multiple_files(self):
        """Test that save_files_to_disk correctly saves multiple files."""
        files = [
            GeneratedFile(filename="file1.txt", content="Content 1"),
            GeneratedFile(filename="file2.txt", content="Content 2"),
            GeneratedFile(filename="dir/file3.txt", content="Content 3"),
        ]
        
        save_files_to_disk(files, self.test_path)
        
        file1 = self.test_path / "file1.txt"
        file2 = self.test_path / "file2.txt"
        file3 = self.test_path / "dir" / "file3.txt"
        
        self.assertTrue(file1.exists())
        self.assertTrue(file2.exists())
        self.assertTrue(file3.exists())
        self.assertEqual(file1.read_text(), "Content 1")
        self.assertEqual(file2.read_text(), "Content 2")
        self.assertEqual(file3.read_text(), "Content 3")

    def test_no_files_detected(self):
        """Test behavior when no files are in the response."""
        files = []
        
        # This should print "NO FILES DETECTED" and not raise an error
        save_files_to_disk(files, self.test_path)
        
        # Directory should be empty (or only have directories, no files)
        saved_files = list(self.test_path.glob("*"))
        self.assertEqual(len(saved_files), 0)

    def test_duplicate_file_in_subdirectory(self):
        """Test that duplicate files in subdirectories are resolved correctly."""
        files = [
            GeneratedFile(filename="20-modules/school-README.md", content="Version 1"),
            GeneratedFile(filename="20-modules/school-README.md", content="Version 2"),
        ]
        
        # Resolve conflicts in-memory first
        unique_files = resolve_file_conflicts(files)
        save_files_to_disk(unique_files, self.test_path)
        
        file1 = self.test_path / "20-modules" / "school-README.md"
        file2 = self.test_path / "20-modules" / "school-README.1.md"
        
        # Both files should exist in the subdirectory
        self.assertTrue(file1.exists(), "Original file should exist")
        self.assertTrue(file2.exists(), "Numbered file should exist in same directory")
        
        # Check content
        self.assertEqual(file1.read_text(), "Version 1")
        self.assertEqual(file2.read_text(), "Version 2")
        
        # Make sure numbered file is NOT in root directory
        wrong_location = self.test_path / "school-README.1.md"
        self.assertFalse(wrong_location.exists(), "Numbered file should not be in root directory")

    def test_resolve_exact_duplicates(self):
        """Test that resolve_file_conflicts removes exact duplicates."""
        files = [
            GeneratedFile(filename="README.md", content="Same content"),
            GeneratedFile(filename="README.md", content="Same content"),
            GeneratedFile(filename="other.md", content="Different"),
        ]
        
        unique_files = resolve_file_conflicts(files)
        
        # Should have 2 files (one duplicate removed)
        self.assertEqual(len(unique_files), 2)
        self.assertEqual(unique_files[0].filename, "README.md")
        self.assertEqual(unique_files[1].filename, "other.md")

    def test_resolve_same_name_different_content(self):
        """Test that resolve_file_conflicts renames files with same name but different content."""
        files = [
            GeneratedFile(filename="config.yaml", content="version 1"),
            GeneratedFile(filename="config.yaml", content="version 2"),
            GeneratedFile(filename="config.yaml", content="version 3"),
        ]
        
        unique_files = resolve_file_conflicts(files)
        
        # Should have 3 files with numbered names
        self.assertEqual(len(unique_files), 3)
        self.assertEqual(unique_files[0].filename, "config.yaml")
        self.assertEqual(unique_files[1].filename, "config.1.yaml")
        self.assertEqual(unique_files[2].filename, "config.2.yaml")


if __name__ == '__main__':
    unittest.main()
