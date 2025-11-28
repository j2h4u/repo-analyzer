"""CLI helper functions for formatted console output."""



def print_message(msg: str, icon: str = "", indent: int = 0):
    """Base function for printing formatted messages.
    
    Args:
        msg: Message to display
        icon: Optional icon prefix
        indent: Number of spaces to indent
    """
    prefix = " " * indent
    print(f"{prefix}{icon}{msg}")


def print_error(msg: str, indent: int = 0):
    """Print error message with icon.
    
    Args:
        msg: Error message to display
        indent: Number of spaces to indent
    """
    print_message(msg, icon="❗️", indent=indent)


def print_warning(msg: str, indent: int = 0):
    """Print warning message with icon.
    
    Args:
        msg: Warning message to display
        indent: Number of spaces to indent
    """
    print_message(msg, icon="⚠️", indent=indent)


def print_info(msg: str, indent: int = 2):
    """Print info message with default 2-space indent.
    
    Args:
        msg: Info message to display
        indent: Number of spaces to indent (default: 2)
    """
    print_message(msg, indent=indent)
