# Repo-to-Gemini Analyzer

A Python automation tool that processes a codebase archive (ZIP), uploads the context to Google Gemini API with intelligent caching, and applies LLM-generated changes or refactoring back to your local files.

## Features

- **Smart Context Handling**: Extracts relevant code files from a ZIP archive based on configuration
- **Cost & Time Efficient**: Hashes the codebase content and checks Gemini's file cache to avoid re-uploading identical context
- **Model Validation**: Verifies API access and model name validity before starting heavy processing
- **Diagnostic Reporting**: Generates detailed reports showing which files were excluded from the AI context
- **Auto-Extraction**: Parses the LLM response and automatically saves generated code files to the output directory

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

## Usage

1. Zip your target repository (e.g., `git archive -o repo.zip HEAD`)
2. Write your instructions in `user_prompt.txt`
3. Run the script:
   ```bash
   chmod +x analyze_repo_zip.py
   ./analyze_repo_zip.py
   ```

The script will extract relevant files, upload to Gemini (or use cached version), generate response, and save files to `output/` folder.

## Utility Tools

- **`tools/get-available-gemini-models.py`** - Lists available Gemini models with context window sizes
- **`tools/delete-remote-files.py`** - Manages cached files in Gemini's cloud storage

## Output

- **`output/`** - Generated code files (preserves directory structure)
- **`context.txt`** - Merged codebase context sent to Gemini (useful for debugging or reuse in other models)
- **`report.txt`** - Execution report with token usage, processing duration, and skipped files

## Tips

- Check `report.txt` to see which files were excluded
- Use `pip freeze > requirements.txt` to update dependencies
- Run `get-available-gemini-models.py` to find the best model for your needs

## License

MIT
