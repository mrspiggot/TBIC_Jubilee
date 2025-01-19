from abc import ABC, abstractmethod
import time
from typing import Optional, Callable, TypeVar, Any
import logging
from functools import wraps

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Generic type for function return
T = TypeVar('T')


class BackoffStrategy(ABC):
    """Abstract base class for backoff strategies.

    This class defines the interface for implementing different backoff strategies
    for handling API rate limits and transient failures.
    """

    @abstractmethod
    def execute(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Execute the function with retry logic.

        Args:
            func: The function to execute with retries
            *args: Positional arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function

        Returns:
            The result of the function execution

        Raises:
            Exception: The last exception encountered after all retries
        """
        pass


class ExponentialBackoff(BackoffStrategy):
    """Implements exponential backoff with maximum retries.

    This implementation increases the delay between retries exponentially,
    with a maximum delay cap.

    Attributes:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay between retries in seconds
        exponential_base: Base for exponential calculation
    """

    def __init__(self, max_retries: int = 3, initial_delay: float = 1.0,
                 max_delay: float = 60.0, exponential_base: float = 2.0):
        """Initialize the exponential backoff strategy.

        Args:
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay in seconds
            max_delay: Maximum delay between retries in seconds
            exponential_base: Base for exponential calculation
        """
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base

    def calculate_delay(self, attempt: int) -> float:
        """Calculate the delay for the current attempt.

        Args:
            attempt: The current attempt number (0-based)

        Returns:
            The calculated delay in seconds
        """
        delay = min(
            self.initial_delay * (self.exponential_base ** attempt),
            self.max_delay
        )
        return delay

    def execute(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Execute the function with exponential backoff retry logic.

        Args:
            func: The function to execute with retries
            *args: Positional arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function

        Returns:
            The result of the function execution

        Raises:
            Exception: The last exception encountered after all retries
        """
        last_exception: Optional[Exception] = None

        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)

            except Exception as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    delay = self.calculate_delay(attempt)
                    logger.warning(
                        f"Attempt {attempt + 1}/{self.max_retries} failed: {str(e)}. "
                        f"Retrying in {delay:.2f} seconds..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        f"All {self.max_retries} attempts failed. "
                        f"Last error: {str(e)}"
                    )

        if last_exception:
            raise last_exception


def with_backoff(strategy: Optional[BackoffStrategy] = None) -> Callable:
    """Decorator to apply backoff strategy to a function.

    Args:
        strategy: The backoff strategy to use. If None, uses default ExponentialBackoff

    Returns:
        Decorated function with backoff strategy applied
    """
    if strategy is None:
        strategy = ExponentialBackoff()

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            return strategy.execute(func, *args, **kwargs)

        return wrapper

    return decorator