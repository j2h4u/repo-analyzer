"""Formatting utilities for presenting data."""


def format_token_count(count: int) -> str:
    """Format token count for display.
    
    Converts large numbers to human-readable format:
    - 300619 -> '301k'
    - 1500000 -> '1.5M'
    
    Args:
        count: Token count to format
        
    Returns:
        Formatted string representation
    """
    if count >= 1_000_000:
        return f"{count/1_000_000:.1f}M"
    if count >= 1_000:
        return f"{count/1000:.0f}k"
    return str(count)


def format_duration(seconds: float) -> str:
    """Format duration for display.
    
    Converts seconds to human-readable format:
    - 147.5 -> '2m 27s'
    - 45 -> '45s'
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted string representation
    """
    m, s = divmod(int(seconds), 60)
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"
