# Repo-to-Gemini Generator

A Python automation tool that processes a codebase archive (ZIP), uploads the context to Google Gemini API with caching support, and applies LLM-generated changes or refactoring back to your local files.

## Features

- **Smart Context Handling**: Extracts relevant code files from a ZIP archive based on configuration.
- **Cost & Time Efficient**: Hashes the codebase content and checks Gemini's file cache to avoid re-uploading the same context multiple times.
- **Model Validation**: Verifies API access and model name validity before starting heavy processing.
- **Diagnostic Reporting**: Generates a `skipped_files_report.txt` to show exactly which files were excluded from the context sent to the AI (filtered to reduce noise).
- **Auto-Extraction**: Parses the LLM response and automatically saves the generated code files to the output directory.

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configure Environment:

Create a `.env` file:

```shell
GOOGLE_API_KEY=your_api_key_here
```

## Configure Project

Copy config.yaml.example to config.yaml and adjust:

`zip_path`: Path to your zipped codebase.
`valid_extensions`: List of extensions to send to the AI (e.g., .py, .js).
`ignore_dirs`: Folders to completely exclude (e.g., node_modules).

## Usage

Zip your target repository (e.g., `git archive -o repo.zip HEAD`, or download it from GitHub).
Write your instructions in `user_prompt.txt`.
Run the script:
```shell
chmod +x analyze_repo_zip.py
./analyze_repo_zip.py
```

The script will:
- Extract relevant files.
- Merge them into one helluva big file.
- Upload it (or use cached version oin the cloud).
- Generate response and save files to the output/ folder.

## Configuration Example

```yaml
processing:
  valid_extensions: [".py", ".ts", ".tsx"]
  include_filenames: ["Dockerfile", "package.json"]
  ignore_dirs: ["node_modules", ".git", "build"]
```

## License

MIT
