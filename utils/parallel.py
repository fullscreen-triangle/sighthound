"""
Utilities for parallel processing
"""

import os
import logging
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future, as_completed
from typing import List, Callable, TypeVar, Generic, Iterable, Dict, Any, Optional, Union, Tuple
import time
from functools import wraps
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm
from utils.profiler import timer, profile

# Configure logger
logger = logging.getLogger(__name__)

# Type variables for generics
T = TypeVar('T')
R = TypeVar('R')

@dataclass
class ParallelConfig:
    """Configuration for parallel processing operations"""
    n_workers: int = None  # Number of workers (None = auto-detect)
    chunk_size: int = 1000  # Number of items to process in each chunk
    use_processes: bool = True  # Use processes instead of threads
    show_progress: bool = True  # Show progress bar
    progress_desc: str = "Processing"  # Description for progress bar
    timeout: Optional[float] = None  # Timeout for processing
    raise_exceptions: bool = False  # Raise exceptions instead of collecting them
    preserve_order: bool = True  # Preserve original order of items
    batch_size: int = 10  # Number of items to process in each batch
    max_queue_size: int = 10000  # Maximum number of items to queue

class ParallelExecutionError(Exception):
    """Exception raised when parallel execution fails"""
    def __init__(self, message: str, errors: Dict[int, Exception] = None):
        self.errors = errors or {}
        self.message = message
        super().__init__(f"{message}. {len(self.errors)} errors occurred.")

@dataclass
class ParallelResult(Generic[T]):
    """Container for results of parallel processing"""
    successful: List[T]  # Successfully processed items
    failed: Dict[int, Exception]  # Map of index to exception for failed items
    execution_time: float  # Total execution time in seconds
    
    @property
    def success_count(self) -> int:
        """Number of successful operations"""
        return len(self.successful)
    
    @property
    def failure_count(self) -> int:
        """Number of failed operations"""
        return len(self.failed)
    
    @property
    def total_count(self) -> int:
        """Total number of operations"""
        return self.success_count + self.failure_count
    
    @property
    def success_rate(self) -> float:
        """Success rate as fraction"""
        if self.total_count == 0:
            return 0.0
        return self.success_count / self.total_count
    
    def get_errors_summary(self) -> Dict[str, int]:
        """
        Get summary of errors by type
        
        Returns:
            Dictionary mapping error types to counts
        """
        if not self.failed:
            return {}
            
        result = {}
        for exc in self.failed.values():
            error_type = type(exc).__name__
            result[error_type] = result.get(error_type, 0) + 1
            
        return result

class ParallelExecutor:
    """
    Utility class for parallel processing using either threads or processes
    """
    
    @staticmethod
    def get_optimal_workers(cpu_bound: bool = False) -> int:
        """
        Get optimal number of workers based on system resources
        
        Args:
            cpu_bound: Whether the tasks are CPU-bound (True) or I/O-bound (False)
            
        Returns:
            Optimal number of workers
        """
        if cpu_bound:
            # For CPU-bound tasks, use available CPUs
            workers = multiprocessing.cpu_count()
            # Leave one CPU for system operations
            return max(1, workers - 1)
        else:
            # For I/O-bound tasks, use more threads than CPUs
            workers = multiprocessing.cpu_count() * 2
            return min(32, workers)  # Limit to avoid excessive overhead
    
    @staticmethod
    def map_parallel(
        function: Callable[[T], R],
        items: List[T],
        workers: Optional[int] = None,
        use_processes: bool = False,
        timeout: Optional[float] = None,
        error_handling: str = 'collect',
        show_progress: bool = False
    ) -> ParallelResult[R]:
        """
        Execute a function over a list of items in parallel
        
        Args:
            function: Function to apply to each item
            items: List of items to process
            workers: Number of workers (threads/processes)
            use_processes: Whether to use processes instead of threads
            timeout: Maximum execution time in seconds (None for no limit)
            error_handling: How to handle errors ('collect', 'raise', or 'ignore')
            show_progress: Whether to show progress information
            
        Returns:
            ParallelResult container with results
            
        Raises:
            ParallelExecutionError: If error_handling is 'raise' and any execution fails
        """
        if not items:
            return ParallelResult(
                successful=[],
                failed={},
                execution_time=0.0
            )
            
        # Determine number of workers if not specified
        if workers is None:
            workers = ParallelExecutor.get_optimal_workers(use_processes)
            
        # Cap workers by the number of items
        workers = min(workers, len(items))
        
        if show_progress:
            logger.info(f"Processing {len(items)} items with {workers} {'processes' if use_processes else 'threads'}")
        
        # Choose executor based on type of operation
        start_time = time.time()
        results: List[Optional[R]] = [None] * len(items)
        errors: Dict[int, Exception] = {}
        
        executor_class = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
        
        with executor_class(max_workers=workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(function, item): i
                for i, item in enumerate(items)
            }
            
            # Process results as they complete
            completed = 0
            total = len(future_to_index)
            
            for future in as_completed(future_to_index, timeout=timeout):
                index = future_to_index[future]
                
                try:
                    results[index] = future.result()
                except Exception as exc:
                    logger.debug(f"Error processing item {index}: {exc}")
                    errors[index] = exc
                
                completed += 1
                if show_progress and completed % max(1, total // 20) == 0:  # Show progress ~20 times
                    percent = (completed / total) * 100
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    logger.info(f"Progress: {percent:.1f}% ({completed}/{total}), "
                               f"Rate: {rate:.1f} items/s")
        
        execution_time = time.time() - start_time
        
        if show_progress:
            logger.info(f"Completed in {execution_time:.2f}s with {len(errors)} errors")
        
        # Filter out None values from results where errors occurred
        successful = [result for i, result in enumerate(results) if i not in errors]
        
        # Handle errors based on strategy
        if errors and error_handling == 'raise':
            raise ParallelExecutionError(
                f"Failed to process {len(errors)} of {len(items)} items",
                errors
            )
            
        if error_handling == 'ignore':
            errors = {}
            
        return ParallelResult(
            successful=successful,
            failed=errors,
            execution_time=execution_time
        )
    
    @staticmethod
    def chunked_parallel(
        function: Callable[[List[T]], List[R]],
        items: List[T],
        chunk_size: int,
        workers: Optional[int] = None,
        use_processes: bool = False,
        show_progress: bool = False
    ) -> ParallelResult[R]:
        """
        Process items in parallel chunks
        
        Args:
            function: Function that takes a chunk of items and returns a list of results
            items: List of items to process
            chunk_size: Size of each chunk
            workers: Number of workers (threads/processes)
            use_processes: Whether to use processes instead of threads
            show_progress: Whether to show progress information
            
        Returns:
            ParallelResult container with flattened results
        """
        if not items:
            return ParallelResult(
                successful=[],
                failed={},
                execution_time=0.0
            )
            
        # Create chunks
        chunks: List[List[T]] = []
        for i in range(0, len(items), chunk_size):
            chunks.append(items[i:i+chunk_size])
            
        if show_progress:
            logger.info(f"Created {len(chunks)} chunks of size {chunk_size} from {len(items)} items")
            
        # Define a function to process a single chunk and return the chunk index with the results
        def process_chunk(chunk_with_index: Tuple[int, List[T]]) -> Tuple[int, List[R]]:
            chunk_index, chunk = chunk_with_index
            if show_progress:
                logger.debug(f"Processing chunk {chunk_index+1}/{len(chunks)} with {len(chunk)} items")
            return chunk_index, function(chunk)
            
        # Determine number of workers if not specified
        if workers is None:
            workers = ParallelExecutor.get_optimal_workers(use_processes)
            
        # Cap workers by the number of chunks
        workers = min(workers, len(chunks))
        
        # Start execution
        start_time = time.time()
        chunk_results: List[Optional[List[R]]] = [None] * len(chunks)
        errors: Dict[int, Exception] = {}
        
        executor_class = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
        
        with executor_class(max_workers=workers) as executor:
            # Submit all chunks
            future_to_index = {
                executor.submit(process_chunk, (i, chunk)): i
                for i, chunk in enumerate(chunks)
            }
            
            # Process results as they complete
            completed = 0
            total = len(future_to_index)
            
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                
                try:
                    chunk_index, result = future.result()
                    chunk_results[chunk_index] = result
                except Exception as exc:
                    logger.debug(f"Error processing chunk {index}: {exc}")
                    errors[index] = exc
                
                completed += 1
                if show_progress and completed % max(1, total // 10) == 0:  # Show progress ~10 times
                    percent = (completed / total) * 100
                    elapsed = time.time() - start_time
                    logger.info(f"Chunk progress: {percent:.1f}% ({completed}/{total}), "
                               f"Elapsed: {elapsed:.2f}s")
        
        execution_time = time.time() - start_time
        
        if show_progress:
            logger.info(f"Chunk processing completed in {execution_time:.2f}s with {len(errors)} errors")
        
        # Flatten successful results, applying the original offsets for error mapping
        successful: List[R] = []
        failed_items: Dict[int, Exception] = {}
        
        for chunk_index, chunk_result in enumerate(chunk_results):
            if chunk_result is not None:
                successful.extend(chunk_result)
            else:
                # Handle error - map to original item indices
                start_idx = chunk_index * chunk_size
                for i in range(min(chunk_size, len(items) - start_idx)):
                    failed_items[start_idx + i] = errors[chunk_index]
        
        return ParallelResult(
            successful=successful,
            failed=failed_items,
            execution_time=execution_time
        )

def parallel(workers: Optional[int] = None, use_processes: bool = False):
    """
    Decorator to parallelize a function that operates on a sequence
    
    The decorated function should take a sequence as its first argument
    and return a sequence of results.
    
    Args:
        workers: Number of workers (threads/processes)
        use_processes: Whether to use processes instead of threads
        
    Returns:
        Decorated function that processes items in parallel
    """
    def decorator(func):
        @wraps(func)
        def wrapper(items, *args, **kwargs):
            if not items:
                return []
                
            # Define a function that applies func to a single item
            def process_item(item):
                return func([item], *args, **kwargs)[0]
                
            result = ParallelExecutor.map_parallel(
                process_item,
                items,
                workers=workers,
                use_processes=use_processes
            )
            
            return result.successful
        return wrapper
    return decorator

class ThreadSafeCounter:
    """Thread-safe counter for tracking progress"""
    
    def __init__(self, initial: int = 0):
        self._value = initial
        self._lock = threading.Lock()
        
    def increment(self, amount: int = 1) -> int:
        """
        Increment counter and return new value
        
        Args:
            amount: Amount to increment by
            
        Returns:
            New counter value
        """
        with self._lock:
            self._value += amount
            return self._value
            
    def get(self) -> int:
        """
        Get current value
        
        Returns:
            Current counter value
        """
        with self._lock:
            return self._value
            
    def reset(self, value: int = 0) -> None:
        """
        Reset counter to specified value
        
        Args:
            value: Value to reset to
        """
        with self._lock:
            self._value = value

class ProgressTracker:
    """
    Utility class for tracking progress of parallel operations
    """
    
    def __init__(self, total: int, update_interval: float = 1.0):
        """
        Initialize progress tracker
        
        Args:
            total: Total number of items to process
            update_interval: Time interval for progress updates in seconds
        """
        self.total = total
        self.completed = ThreadSafeCounter()
        self.start_time = time.time()
        self.update_interval = update_interval
        self.last_update = self.start_time
        self._lock = threading.Lock()
        
    def update(self, completed: int = 1) -> None:
        """
        Update progress
        
        Args:
            completed: Number of newly completed items
        """
        new_count = self.completed.increment(completed)
        
        current_time = time.time()
        with self._lock:
            if current_time - self.last_update >= self.update_interval:
                self._log_progress()
                self.last_update = current_time
                
    def _log_progress(self) -> None:
        """Log progress information"""
        elapsed = time.time() - self.start_time
        completed = self.completed.get()
        
        if elapsed > 0 and completed > 0:
            items_per_sec = completed / elapsed
            remaining = (self.total - completed) / items_per_sec if items_per_sec > 0 else 0
            
            logger.info(
                f"Progress: {completed}/{self.total} ({completed/self.total:.1%}) "
                f"- {items_per_sec:.2f} items/sec, {remaining:.1f}s remaining"
            )
        else:
            logger.info(f"Progress: {completed}/{self.total} ({completed/self.total:.1%})")
            
    def finish(self) -> None:
        """Log final progress information"""
        elapsed = time.time() - self.start_time
        completed = self.completed.get()
        
        if elapsed > 0 and completed > 0:
            items_per_sec = completed / elapsed
            logger.info(
                f"Completed: {completed}/{self.total} ({completed/self.total:.1%}) "
                f"in {elapsed:.2f}s - {items_per_sec:.2f} items/sec"
            )
        else:
            logger.info(f"Completed: {completed}/{self.total} in {elapsed:.2f}s")

class ParallelProcessor:
    """
    Utility for parallel processing of independent operations
    
    This class provides methods for executing operations in parallel
    using either threads or processes, with configurable options for
    batch processing, progress reporting, and error handling.
    """
    
    def __init__(self, config: Optional[ParallelConfig] = None):
        """
        Initialize the parallel processor
        
        Args:
            config: Configuration for parallel processing
        """
        self.config = config or ParallelConfig()
        
    @timer
    def map(self, 
            func: Callable[[T], R], 
            items: List[T], 
            **kwargs) -> List[R]:
        """
        Apply a function to each item in parallel
        
        Args:
            func: Function to apply to each item
            items: List of items to process
            **kwargs: Additional arguments to pass to the function
            
        Returns:
            List of results
        """
        if not items:
            return []
            
        # Determine executor type based on configuration
        executor_class = ProcessPoolExecutor if self.config.use_processes else ThreadPoolExecutor
        
        # Create partial function with additional arguments
        if kwargs:
            from functools import partial
            func = partial(func, **kwargs)
        
        # Process items in parallel with progress reporting
        results = []
        
        with executor_class(max_workers=self.config.n_workers) as executor:
            # Submit all tasks
            futures = {executor.submit(func, item): i for i, item in enumerate(items)}
            
            # Process completed tasks with optional progress reporting
            iterator = as_completed(futures)
            if self.config.show_progress:
                iterator = tqdm(
                    iterator, 
                    total=len(items), 
                    desc=self.config.progress_desc
                )
                
            # Collect results
            results_dict = {}
            for future in iterator:
                idx = futures[future]
                try:
                    result = future.result(timeout=self.config.timeout)
                    results_dict[idx] = result
                except Exception as e:
                    if self.config.raise_exceptions:
                        raise
                    logger.error(f"Error processing item {idx}: {str(e)}")
                    results_dict[idx] = None
            
            # Preserve original order if requested
            if self.config.preserve_order:
                results = [results_dict.get(i) for i in range(len(items))]
            else:
                results = list(results_dict.values())
                
        return results
    
    @timer
    def batch_map(self, 
                 func: Callable[[List[T]], List[R]], 
                 items: List[T], 
                 **kwargs) -> List[R]:
        """
        Apply a function to batches of items in parallel
        
        Args:
            func: Function that processes a batch of items
            items: List of items to process
            **kwargs: Additional arguments to pass to the function
            
        Returns:
            List of results (flattened)
        """
        if not items:
            return []
            
        # Create batches
        batches = self._create_batches(items, self.config.batch_size)
        
        # Process batches in parallel
        batch_results = self.map(func, batches, **kwargs)
        
        # Flatten results
        results = []
        for batch_result in batch_results:
            if batch_result is not None:
                results.extend(batch_result)
        
        return results
    
    @timer
    def process_chunks(self,
                      func: Callable[[List[T]], List[R]],
                      items: List[T],
                      chunk_size: Optional[int] = None,
                      **kwargs) -> List[R]:
        """
        Process data in chunks to handle large datasets efficiently
        
        Args:
            func: Function that processes a chunk of items
            items: List of items to process
            chunk_size: Size of each chunk (defaults to config.chunk_size)
            **kwargs: Additional arguments to pass to the function
            
        Returns:
            List of results (flattened)
        """
        if not items:
            return []
            
        # Determine chunk size
        chunk_size = chunk_size or self.config.chunk_size
        
        # Create chunks
        chunks = self._create_batches(items, chunk_size)
        
        logger.info(f"Processing {len(items)} items in {len(chunks)} chunks")
        
        # Process chunks in parallel
        chunk_results = self.map(func, chunks, **kwargs)
        
        # Flatten results
        results = []
        for chunk_result in chunk_results:
            if chunk_result is not None:
                results.extend(chunk_result)
        
        return results
    
    def _create_batches(self, items: List[T], batch_size: int) -> List[List[T]]:
        """
        Split items into batches
        
        Args:
            items: List of items to split
            batch_size: Size of each batch
            
        Returns:
            List of batches
        """
        return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]

def parallel_map(func: Callable[[T], R], 
                items: List[T], 
                n_workers: int = None,
                use_processes: bool = True,
                show_progress: bool = True,
                **kwargs) -> List[R]:
    """
    Convenience function for parallel mapping
    
    Args:
        func: Function to apply to each item
        items: List of items to process
        n_workers: Number of workers (default: CPU count - 1)
        use_processes: Use processes instead of threads
        show_progress: Show progress bar
        **kwargs: Additional arguments to pass to the function
        
    Returns:
        List of results
    """
    config = ParallelConfig(
        n_workers=n_workers or max(1, multiprocessing.cpu_count() - 1),
        use_processes=use_processes,
        show_progress=show_progress
    )
    processor = ParallelProcessor(config)
    return processor.map(func, items, **kwargs)

def parallel_batch_process(func: Callable[[List[T]], List[R]],
                          items: List[T],
                          batch_size: int = 10,
                          n_workers: int = None,
                          use_processes: bool = True,
                          show_progress: bool = True,
                          **kwargs) -> List[R]:
    """
    Convenience function for parallel batch processing
    
    Args:
        func: Function that processes a batch of items
        items: List of items to process
        batch_size: Number of items in each batch
        n_workers: Number of workers (default: CPU count - 1)
        use_processes: Use processes instead of threads
        show_progress: Show progress bar
        **kwargs: Additional arguments to pass to the function
        
    Returns:
        List of results (flattened)
    """
    config = ParallelConfig(
        n_workers=n_workers or max(1, multiprocessing.cpu_count() - 1),
        batch_size=batch_size,
        use_processes=use_processes,
        show_progress=show_progress
    )
    processor = ParallelProcessor(config)
    return processor.batch_map(func, items, **kwargs)

@timer
def process_large_file(file_path: str,
                      process_func: Callable[[List[str]], List[R]],
                      chunk_size: int = 10000,
                      n_workers: int = None,
                      **kwargs) -> List[R]:
    """
    Process a large file in chunks using parallel processing
    
    Args:
        file_path: Path to the file
        process_func: Function to process each chunk of lines
        chunk_size: Number of lines to process in each chunk
        n_workers: Number of workers (default: CPU count - 1)
        **kwargs: Additional arguments to pass to the process function
        
    Returns:
        List of results (flattened)
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return []
        
    # Count total lines for progress reporting
    total_lines = sum(1 for _ in open(file_path, 'r'))
    
    # Process file in chunks
    results = []
    current_chunk = []
    
    config = ParallelConfig(
        n_workers=n_workers or max(1, multiprocessing.cpu_count() - 1),
        show_progress=True,
        progress_desc=f"Processing {os.path.basename(file_path)}"
    )
    processor = ParallelProcessor(config)
    
    with open(file_path, 'r') as f:
        for i, line in enumerate(tqdm(f, total=total_lines, desc=f"Reading {os.path.basename(file_path)}")):
            current_chunk.append(line.strip())
            
            if len(current_chunk) >= chunk_size:
                # Process chunk
                chunk_results = processor.map(process_func, [current_chunk], **kwargs)
                if chunk_results and chunk_results[0]:
                    results.extend(chunk_results[0])
                
                # Reset chunk
                current_chunk = []
    
    # Process remaining lines
    if current_chunk:
        chunk_results = processor.map(process_func, [current_chunk], **kwargs)
        if chunk_results and chunk_results[0]:
            results.extend(chunk_results[0])
    
    return results
