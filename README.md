# Repo-to-Gemini Analyzer

A Python automation tool that processes a codebase archive (ZIP), uploads the context to Google Gemini API with intelligent caching, and applies LLM-generated changes or refactoring back to your local files.

## Features

- **Smart Context Handling**: Extracts relevant code files from a ZIP archive based on configuration
- **Tree-Style Structure**: Generates a visual file tree (like `tree` command) showing all files, with `# not attached` markers for excluded files
- **Cost & Time Efficient**: Hashes the codebase content and checks Gemini's file cache to avoid re-uploading identical context
- **Model Validation**: Verifies API access and model name validity before starting heavy processing
- **Diagnostic Reporting**: Generates detailed reports showing which files were excluded from the AI context
- **Auto-Extraction**: Parses the LLM response and automatically saves generated code files to the output directory
- **Conflict Resolution**: Automatically deduplicates and renames conflicting files to preserve all LLM-generated content
- **Debug Logging**: Creates detailed `debug.log` with inference stats and processing details for troubleshooting
- **Reusable Context**: Saves the merged context file for debugging or reuse in other AI models
- **Test Coverage**: Includes unit tests for core file parsing and conflict resolution logic

## Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment** - create `.env` file:
   ```shell
   GOOGLE_API_KEY=your_api_key_here
   ```

3. **Configure project** - copy and edit `config.yaml`:
   ```bash
   cp config.yaml.example config.yaml
   ```
   Adjust `zip_path`, `valid_extensions`, `ignore_dirs`, and model settings as needed.

4. **Make script executable** (one-time):
   ```bash
   chmod +x analyze_repo_zip.py
   ```

## Usage

1. **Get repository as ZIP**:
   - Download from GitHub: Code → Download ZIP
   - Or create from local repo: `git archive -o repo.zip HEAD`

2. **Configure ZIP path** in `config.yaml`:
   ```yaml
   zip_path: "./downloaded-repo.zip"
   ```

3. **Review prompts** (optional):
   - Default prompts in `prompts/` are ready to use
   - Edit `prompts/user_prompt.txt` to customize analysis goals if needed

4. **Run the script**:
   ```bash
   ./analyze_repo_zip.py
   ```

The script will:
- Extract and analyze files from the ZIP
- Generate `context.txt` with tree structure and file contents
- Upload to Gemini (or use cloud-cached version)
- Generate AI response and save files to `output/` folder

## Utility Tools

- **`tools/get-available-gemini-models.py`** - Lists available Gemini models with context window sizes
- **`tools/delete-remote-files.py`** - Manages cached files in Gemini's cloud storage

## Output

After each run, a timestamped directory is created in `output/` containing:

- **`context.txt`** - Full merged context with:
  - Tree-style directory structure (files marked with `# not attached` if excluded)
  - Complete contents of all included files
  - Perfect for reusing in other AI models or debugging
- **`report.txt`** - Execution report with token usage, processing duration, and skipped files
- **`debug.log`** - Detailed debug log with inference statistics and processing events
- **`response.txt`** - Raw model response for inspection
- **Generated files** - AI-generated code preserving the original directory structure

## Tips

- Check `report.txt` to see which files were excluded
- Review `debug.log` for detailed processing information and troubleshooting
- Use `pip freeze > requirements.txt` to update dependencies
- Run `get-available-gemini-models.py` to find the best model for your needs
- Execute `pytest tests/` to run unit tests

## Development
> [!NOTE]
> Before contributing or refactoring, please read [BUSINESS_CONTEXT.md](doc/BUSINESS_CONTEXT.md) to understand the architectural vision and constraints.

## License

MIT
