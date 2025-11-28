"""Event emitter for pipeline progress notifications."""

from typing import Callable, Dict, List


class SimpleEmitter:
    """Lightweight event emitter for pipeline progress.
    
    Allows components to emit events that other components can listen to,
    enabling loose coupling between business logic and presentation layers.
    
    Example:
        >>> emitter = SimpleEmitter()
        >>> emitter.on('progress', lambda **kw: print(f"Progress: {kw['percent']}%"))
        >>> emitter.emit('progress', percent=50)
        Progress: 50%
    """
    
    def __init__(self):
        """Initialize empty event handlers dictionary."""
        self._handlers: Dict[str, List[Callable]] = {}
    
    def on(self, event: str, handler: Callable) -> 'SimpleEmitter':
        """Subscribe to an event.
        
        Args:
            event: Event name to listen for
            handler: Callable to invoke when event is emitted.
                    Will receive event data as keyword arguments.
        
        Returns:
            Self for method chaining
        """
        if event not in self._handlers:
            self._handlers[event] = []
        self._handlers[event].append(handler)
        return self
    
    def emit(self, event: str, **data):
        """Emit an event with data.
        
        All handlers registered for this event will be called with
        the provided data as keyword arguments.
        
        Args:
            event: Event name to emit
            **data: Arbitrary keyword arguments to pass to handlers
        """
        for handler in self._handlers.get(event, []):
            handler(**data)
