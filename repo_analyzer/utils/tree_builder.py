"""Tree building utilities for visualizing directory structures."""


def build_file_tree(all_files: list[str], included_files: set[str]) -> str:
    """Build a tree-style directory structure similar to the `tree` command.
    
    Files not in included_files are marked with comment-like "# not attached".
    
    Args:
        all_files: List of all file paths in the archive
        included_files: Set of files that are included in analysis
        
    Returns:
        Formatted tree string representation
    """
    # Build hierarchical structure
    tree = {}

    for filepath in all_files:
        parts = filepath.split('/')
        current = tree

        # Navigate through directories
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        # Add the file with marker
        filename = parts[-1]
        marker = "" if filepath in included_files else " # not attached"
        current[filename] = marker  # String value indicates it's a file

    # Render the tree
    def render(node, prefix="", name=".", is_last=True):
        lines = []

        if isinstance(node, str):
            # It's a file (leaf node)
            return []

        # First line is the directory name
        if name != ".":
            connector = "└── " if is_last else "├── "
            lines.append(f"{prefix}{connector}{name}/")
            prefix += "    " if is_last else "│   "
        else:
            lines.append(".")

        # Get all items (dirs and files)
        items = sorted(node.items())

        for idx, (key, value) in enumerate(items):
            is_last_item = idx == len(items) - 1

            if isinstance(value, dict):
                # It's a directory
                lines.extend(render(value, prefix, key, is_last_item))
            else:
                # It's a file
                connector = "└── " if is_last_item else "├── "
                lines.append(f"{prefix}{connector}{key}{value}")

        return lines

    return "\n".join(render(tree))
