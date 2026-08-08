import time
from typing import Any, Callable, Dict

def measure_execution_time(func: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[Any, float]:
    """
    Execute a function and return its result along with execution duration in milliseconds.
    """
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()
    duration_ms = (end_time - start_time) * 1000.0
    return result, round(duration_ms, 2)
