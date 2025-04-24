"""
Memory optimization utilities for handling large trajectory datasets
"""

import os
import gc
import sys
import logging
import tempfile
import pickle
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, TypeVar, Generic, Iterator
from dataclasses import dataclass, field
from collections import deque
from functools import wraps
import json
import mmap
from datetime import datetime

# Configure logger
logger = logging.getLogger(__name__)

# Type variable for generics
T = TypeVar('T')

class MemoryManager:
    """
    Utility for tracking and managing memory usage
    """
    
    @staticmethod
    def get_size(obj: Any, seen: Optional[set] = None) -> int:
        """
        Recursively calculate approximate size of an object in bytes
        
        Args:
            obj: Object to measure
            seen: Set of already seen objects (for recursion)
            
        Returns:
            Size in bytes
        """
        if seen is None:
            seen = set()
            
        # Skip already seen objects
        obj_id = id(obj)
        if obj_id in seen:
            return 0
        seen.add(obj_id)
        
        size = sys.getsizeof(obj)
        
        if isinstance(obj, dict):
            size += sum(MemoryManager.get_size(k, seen) + MemoryManager.get_size(v, seen) for k, v in obj.items())
        elif isinstance(obj, (list, tuple, set, deque)):
            size += sum(MemoryManager.get_size(item, seen) for item in obj)
        elif isinstance(obj, pd.DataFrame):
            # Pandas DataFrame - estimate using memory_usage
            size = obj.memory_usage(deep=True).sum()
        elif isinstance(obj, np.ndarray):
            # NumPy array
            size = obj.nbytes
            
        return size
        
    @staticmethod
    def bytes_to_mb(bytes_value: int) -> float:
        """Convert bytes to megabytes"""
        return bytes_value / (1024 * 1024)
        
    @staticmethod
    def format_size(bytes_value: int) -> str:
        """Format bytes to human-readable string"""
        if bytes_value < 1024:
            return f"{bytes_value} bytes"
        elif bytes_value < 1024 * 1024:
            return f"{bytes_value / 1024:.2f} KB"
        elif bytes_value < 1024 * 1024 * 1024:
            return f"{bytes_value / (1024 * 1024):.2f} MB"
        else:
            return f"{bytes_value / (1024 * 1024 * 1024):.2f} GB"
    
    @staticmethod
    def force_garbage_collection() -> Dict[str, Union[int, float]]:
        """
        Force garbage collection and return memory usage statistics
        
        Returns:
            Dictionary with memory usage statistics
        """
        before = gc.get_count()
        before_size = MemoryManager.get_process_memory()
        
        # Collect all generations
        collected = gc.collect()
        
        after = gc.get_count()
        after_size = MemoryManager.get_process_memory()
        
        return {
            "collected_objects": collected,
            "before_count": before,
            "after_count": after,
            "before_memory_mb": before_size,
            "after_memory_mb": after_size,
            "freed_memory_mb": before_size - after_size,
        }
    
    @staticmethod
    def get_process_memory() -> float:
        """
        Get current memory usage of the process in MB
        
        Returns:
            Memory usage in MB
        """
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            return memory_info.rss / (1024 * 1024)  # Return in MB
        except ImportError:
            # Fallback if psutil is not available
            logger.warning("psutil not available, using less accurate memory estimation")
            return 0.0  # Can't estimate accurately without psutil

    @staticmethod
    def log_memory_usage(label: str = "Current memory usage") -> None:
        """
        Log current memory usage
        
        Args:
            label: Label for the log message
        """
        memory_mb = MemoryManager.get_process_memory()
        logger.info(f"{label}: {memory_mb:.2f} MB")
    
    @staticmethod
    def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize memory usage of a pandas DataFrame by downcasting dtypes
        
        Args:
            df: DataFrame to optimize
            
        Returns:
            Optimized DataFrame
        """
        # Store original types to avoid issues with categorical conversions
        original_dtypes = df.dtypes
        
        # Create a copy to avoid modifying the original
        result = df.copy()
        
        # Process each column
        for col in result.columns:
            col_type = original_dtypes[col].name
            
            # Convert integer columns to smallest possible integer type
            if 'int' in col_type:
                max_val = result[col].max()
                min_val = result[col].min()
                
                if min_val >= 0:  # Unsigned integers
                    if max_val <= 255:
                        result[col] = result[col].astype(np.uint8)
                    elif max_val <= 65535:
                        result[col] = result[col].astype(np.uint16)
                    elif max_val <= 4294967295:
                        result[col] = result[col].astype(np.uint32)
                else:  # Signed integers
                    if min_val >= -128 and max_val <= 127:
                        result[col] = result[col].astype(np.int8)
                    elif min_val >= -32768 and max_val <= 32767:
                        result[col] = result[col].astype(np.int16)
                    elif min_val >= -2147483648 and max_val <= 2147483647:
                        result[col] = result[col].astype(np.int32)
            
            # Convert float columns to float32 if possible
            elif 'float' in col_type:
                result[col] = result[col].astype(np.float32)
            
            # Convert string columns to categorical if they have few unique values
            elif col_type in ['object', 'string']:
                num_unique = result[col].nunique()
                # Only convert to categorical if there are many repeated values
                if num_unique < len(result) * 0.5:  # 50% threshold for unique values
                    result[col] = result[col].astype('category')
        
        # Report memory savings
        original_size = MemoryManager.bytes_to_mb(MemoryManager.get_size(df))
        optimized_size = MemoryManager.bytes_to_mb(MemoryManager.get_size(result))
        savings = original_size - optimized_size
        savings_pct = (savings / original_size) * 100 if original_size > 0 else 0
        
        logger.info(f"DataFrame optimization: {original_size:.2f} MB -> {optimized_size:.2f} MB "
                   f"(saved {savings:.2f} MB, {savings_pct:.1f}%)")
        
        return result


class ChunkedProcessor(Generic[T]):
    """
    Process large datasets in chunks to minimize memory usage
    """
    
    def __init__(self, chunk_size: int = 1000):
        """
        Initialize chunked processor
        
        Args:
            chunk_size: Number of items to process in each chunk
        """
        self.chunk_size = chunk_size
    
    def process_dataset(
        self, 
        data: Union[List[T], pd.DataFrame, np.ndarray],
        process_function: callable,
        *args,
        **kwargs
    ) -> List[Any]:
        """
        Process a dataset in chunks to minimize memory usage
        
        Args:
            data: Dataset to process
            process_function: Function to apply to each chunk
            *args, **kwargs: Additional arguments to pass to process_function
            
        Returns:
            List of processed results
        """
        # Determine total length
        total_length = len(data)
        
        # Calculate number of chunks
        num_chunks = (total_length + self.chunk_size - 1) // self.chunk_size
        
        results = []
        
        for i in range(num_chunks):
            # Get the current chunk
            start_idx = i * self.chunk_size
            end_idx = min(start_idx + self.chunk_size, total_length)
            
            # Extract chunk (handle different data types)
            if isinstance(data, pd.DataFrame):
                chunk = data.iloc[start_idx:end_idx].copy()
            elif isinstance(data, np.ndarray):
                chunk = data[start_idx:end_idx].copy()
            else:  # Assume list-like
                chunk = data[start_idx:end_idx]
            
            # Process the chunk
            logger.debug(f"Processing chunk {i+1}/{num_chunks} (items {start_idx+1}-{end_idx})")
            chunk_result = process_function(chunk, *args, **kwargs)
            
            # Store result
            results.append(chunk_result)
            
            # Force garbage collection on large datasets
            if total_length > 10000 and i % 10 == 0:
                MemoryManager.force_garbage_collection()
        
        # Combine results (assume process_function returns list-like)
        if results and isinstance(results[0], (list, tuple)):
            # Flatten list of lists
            flattened = []
            for chunk_result in results:
                flattened.extend(chunk_result)
            return flattened
        elif results and isinstance(results[0], (pd.DataFrame)):
            # Combine DataFrames
            return pd.concat(results)
        else:
            # Return as-is
            return results


class DiskBackedArray(Generic[T]):
    """
    Array-like object that stores data on disk to reduce memory usage
    """
    
    def __init__(self, initial_data: Optional[List[T]] = None, temp_dir: Optional[str] = None):
        """
        Initialize a disk-backed array
        
        Args:
            initial_data: Optional initial data
            temp_dir: Directory for temporary files
        """
        self.temp_dir = temp_dir
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, dir=temp_dir)
        self.length = 0
        self.offsets = []  # Store offsets for each item
        
        if initial_data:
            for item in initial_data:
                self.append(item)
    
    def __del__(self):
        """Cleanup when object is deleted"""
        self.close()
        
    def close(self):
        """Close and remove temporary file"""
        if hasattr(self, 'temp_file') and self.temp_file:
            self.temp_file.close()
            if os.path.exists(self.temp_file.name):
                os.unlink(self.temp_file.name)
    
    def append(self, item: T) -> None:
        """
        Append an item to the array
        
        Args:
            item: Item to append
        """
        # Save current position
        current_offset = self.temp_file.tell()
        self.offsets.append(current_offset)
        
        # Serialize and write the item
        serialized = pickle.dumps(item)
        size = len(serialized)
        
        # Write size and data
        self.temp_file.write(size.to_bytes(4, byteorder='little'))
        self.temp_file.write(serialized)
        self.temp_file.flush()
        
        self.length += 1
    
    def __getitem__(self, idx: int) -> T:
        """
        Get item at index
        
        Args:
            idx: Index of item to retrieve
            
        Returns:
            Item at index
        """
        if idx < 0:
            idx += self.length
            
        if idx < 0 or idx >= self.length:
            raise IndexError(f"Index {idx} out of range (0-{self.length-1})")
        
        # Seek to the offset
        self.temp_file.seek(self.offsets[idx])
        
        # Read size
        size_bytes = self.temp_file.read(4)
        size = int.from_bytes(size_bytes, byteorder='little')
        
        # Read data
        data = self.temp_file.read(size)
        
        # Deserialize
        return pickle.loads(data)
    
    def __len__(self) -> int:
        """
        Get length of the array
        
        Returns:
            Number of items in the array
        """
        return self.length
    
    def __iter__(self) -> Iterator[T]:
        """
        Iterate over items
        
        Returns:
            Iterator of items
        """
        for i in range(self.length):
            yield self[i]
"""
Memory optimization utilities for handling large trajectory datasets
"""

import os
import gc
import sys
import logging
import tempfile
import pickle
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, TypeVar, Generic, Iterator
from dataclasses import dataclass
from collections import deque
from functools import wraps

# Configure logger
logger = logging.getLogger(__name__)

# Type variable for generics
T = TypeVar('T')

class MemoryManager:
    """
    Utility for tracking and managing memory usage
    """
    
    @staticmethod
    def get_size(obj: Any, seen: Optional[set] = None) -> int:
        """
        Recursively calculate approximate size of an object in bytes
        
        Args:
            obj: Object to measure
            seen: Set of already seen objects (for recursion)
            
        Returns:
            Size in bytes
        """
        if seen is None:
            seen = set()
            
        # Skip already seen objects
        obj_id = id(obj)
        if obj_id in seen:
            return 0
        seen.add(obj_id)
        
        size = sys.getsizeof(obj)
        
        if isinstance(obj, dict):
            size += sum(MemoryManager.get_size(k, seen) + MemoryManager.get_size(v, seen) for k, v in obj.items())
        elif isinstance(obj, (list, tuple, set, deque)):
            size += sum(MemoryManager.get_size(item, seen) for item in obj)
        elif isinstance(obj, pd.DataFrame):
            # Pandas DataFrame - estimate using memory_usage
            size = obj.memory_usage(deep=True).sum()
        elif isinstance(obj, np.ndarray):
            # NumPy array
            size = obj.nbytes
            
        return size
        
    @staticmethod
    def bytes_to_mb(bytes_value: int) -> float:
        """Convert bytes to megabytes"""
        return bytes_value / (1024 * 1024)
        
    @staticmethod
    def format_size(bytes_value: int) -> str:
        """Format bytes to human-readable string"""
        if bytes_value < 1024:
            return f"{bytes_value} bytes"
        elif bytes_value < 1024 * 1024:
            return f"{bytes_value / 1024:.2f} KB"
        elif bytes_value < 1024 * 1024 * 1024:
            return f"{bytes_value / (1024 * 1024):.2f} MB"
        else:
            return f"{bytes_value / (1024 * 1024 * 1024):.2f} GB"
    
    @staticmethod
    def force_garbage_collection() -> Dict[str, Union[int, float]]:
        """
        Force garbage collection and return memory usage statistics
        
        Returns:
            Dictionary with memory usage statistics
        """
        before = gc.get_count()
        before_size = MemoryManager.get_process_memory()
        
        # Collect all generations
        collected = gc.collect()
        
        after = gc.get_count()
        after_size = MemoryManager.get_process_memory()
        
        return {
            "collected_objects": collected,
            "before_count": before,
            "after_count": after,
            "before_memory_mb": before_size,
            "after_memory_mb": after_size,
            "freed_memory_mb": before_size - after_size,
        }
    
    @staticmethod
    def get_process_memory() -> float:
        """
        Get current memory usage of the process in MB
        
        Returns:
            Memory usage in MB
        """
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            return memory_info.rss / (1024 * 1024)  # Return in MB
        except ImportError:
            # Fallback if psutil is not available
            logger.warning("psutil not available, using less accurate memory estimation")
            return 0.0  # Can't estimate accurately without psutil

    @staticmethod
    def log_memory_usage(label: str = "Current memory usage") -> None:
        """
        Log current memory usage
        
        Args:
            label: Label for the log message
        """
        memory_mb = MemoryManager.get_process_memory()
        logger.info(f"{label}: {memory_mb:.2f} MB")
    
    @staticmethod
    def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize memory usage of a pandas DataFrame by downcasting dtypes
        
        Args:
            df: DataFrame to optimize
            
        Returns:
            Optimized DataFrame
        """
        # Store original types to avoid issues with categorical conversions
        original_dtypes = df.dtypes
        
        # Create a copy to avoid modifying the original
        result = df.copy()
        
        # Process each column
        for col in result.columns:
            col_type = original_dtypes[col].name
            
            # Convert integer columns to smallest possible integer type
            if 'int' in col_type:
                max_val = result[col].max()
                min_val = result[col].min()
                
                if min_val >= 0:  # Unsigned integers
                    if max_val <= 255:
                        result[col] = result[col].astype(np.uint8)
                    elif max_val <= 65535:
                        result[col] = result[col].astype(np.uint16)
                    elif max_val <= 4294967295:
                        result[col] = result[col].astype(np.uint32)
                else:  # Signed integers
                    if min_val >= -128 and max_val <= 127:
                        result[col] = result[col].astype(np.int8)
                    elif min_val >= -32768 and max_val <= 32767:
                        result[col] = result[col].astype(np.int16)
                    elif min_val >= -2147483648 and max_val <= 2147483647:
                        result[col] = result[col].astype(np.int32)
            
            # Convert float columns to float32 if possible
            elif 'float' in col_type:
                result[col] = result[col].astype(np.float32)
            
            # Convert string columns to categorical if they have few unique values
            elif col_type in ['object', 'string']:
                num_unique = result[col].nunique()
                # Only convert to categorical if there are many repeated values
                if num_unique < len(result) * 0.5:  # 50% threshold for unique values
                    result[col] = result[col].astype('category')
        
        # Report memory savings
        original_size = MemoryManager.bytes_to_mb(MemoryManager.get_size(df))
        optimized_size = MemoryManager.bytes_to_mb(MemoryManager.get_size(result))
        savings = original_size - optimized_size
        savings_pct = (savings / original_size) * 100 if original_size > 0 else 0
        
        logger.info(f"DataFrame optimization: {original_size:.2f} MB -> {optimized_size:.2f} MB "
                   f"(saved {savings:.2f} MB, {savings_pct:.1f}%)")
        
        return result

class DiskBackedArray(Generic[T]):
    """
    Array-like object that stores data on disk to reduce memory usage
    """
    
    def __init__(self, initial_data: Optional[List[T]] = None, temp_dir: Optional[str] = None):
        """
        Initialize a disk-backed array
        
        Args:
            initial_data: Optional initial data
            temp_dir: Directory for temporary files
        """
        self.temp_dir = temp_dir
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, dir=temp_dir)
        self.length = 0
        self.offsets = []  # Store offsets for each item
        
        if initial_data:
            for item in initial_data:
                self.append(item)
    
    def __del__(self):
        """Cleanup when object is deleted"""
        self.close()
        
    def close(self):
        """Close and remove temporary file"""
        if hasattr(self, 'temp_file') and self.temp_file:
            self.temp_file.close()
            if os.path.exists(self.temp_file.name):
                os.unlink(self.temp_file.name)
    
    def append(self, item: T) -> None:
        """
        Append an item to the array
        
        Args:
            item: Item to append
        """
        # Save current position
        current_offset = self.temp_file.tell()
        self.offsets.append(current_offset)
        
        # Serialize and write the item
        serialized = pickle.dumps(item)
        size = len(serialized)
        
        # Write size and data
        self.temp_file.write(size.to_bytes(4, byteorder='little'))
        self.temp_file.write(serialized)
        self.temp_file.flush()
        
        self.length += 1
    
    def __getitem__(self, idx: int) -> T:
        """
        Get item at index
        
        Args:
            idx: Index of item to retrieve
            
        Returns:
            Item at index
        """
        if idx < 0:
            idx += self.length
            
        if idx < 0 or idx >= self.length:
            raise IndexError(f"Index {idx} out of range (0-{self.length-1})")
        
        # Seek to the offset
        self.temp_file.seek(self.offsets[idx])
        
        # Read size
        size_bytes = self.temp_file.read(4)
        size = int.from_bytes(size_bytes, byteorder='little')
        
        # Read data
        data = self.temp_file.read(size)
        
        # Deserialize
        return pickle.loads(data)
    
    def __len__(self) -> int:
        """
        Get length of the array
        
        Returns:
            Number of items in the array
        """
        return self.length
    
    def __iter__(self) -> Iterator[T]:
        """
        Iterate over items
        
        Returns:
            Iterator of items
        """
        for i in range(self.length):
            yield self[i]

class ChunkedProcessor:
    """
    Process large datasets in chunks to minimize memory usage
    """
    
    def __init__(self, chunk_size: int = 1000):
        """
        Initialize chunked processor
        
        Args:
            chunk_size: Number of items to process in each chunk
        """
        self.chunk_size = chunk_size
    
    def process_dataset(
        self, 
        data: Union[List, pd.DataFrame, np.ndarray],
        process_function: callable,
        *args,
        **kwargs
    ) -> List[Any]:
        """
        Process a dataset in chunks to minimize memory usage
        
        Args:
            data: Dataset to process
            process_function: Function to apply to each chunk
            *args, **kwargs: Additional arguments to pass to process_function
            
        Returns:
            List of processed results
        """
        # Determine total length
        total_length = len(data)
        
        # Calculate number of chunks
        num_chunks = (total_length + self.chunk_size - 1) // self.chunk_size
        
        results = []
        
        for i in range(num_chunks):
            # Get the current chunk
            start_idx = i * self.chunk_size
            end_idx = min(start_idx + self.chunk_size, total_length)
            
            # Extract chunk (handle different data types)
            if isinstance(data, pd.DataFrame):
                chunk = data.iloc[start_idx:end_idx].copy()
            elif isinstance(data, np.ndarray):
                chunk = data[start_idx:end_idx].copy()
            else:  # Assume list-like
                chunk = data[start_idx:end_idx]
            
            # Process the chunk
            logger.debug(f"Processing chunk {i+1}/{num_chunks} (items {start_idx+1}-{end_idx})")
            chunk_result = process_function(chunk, *args, **kwargs)
            
            # Store result
            results.append(chunk_result)
            
            # Force garbage collection on large datasets
            if total_length > 10000 and i % 10 == 0:
                MemoryManager.force_garbage_collection()
        
        # Combine results
        if results and isinstance(results[0], (list, tuple)):
            # Flatten list of lists
            flattened = []
            for chunk_result in results:
                flattened.extend(chunk_result)
            return flattened
        elif results and isinstance(results[0], (pd.DataFrame)):
            # Combine DataFrames
            return pd.concat(results)
        else:
            # Return as-is
            return results

def optimize_memory_usage(func):
    """
    Decorator to optimize memory usage of a function
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function with memory optimization
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Log initial memory usage
        MemoryManager.log_memory_usage(f"Before {func.__name__}")
        
        # Call the function
        result = func(*args, **kwargs)
        
        # Force garbage collection
        gc.collect()
        
        # Log final memory usage
        MemoryManager.log_memory_usage(f"After {func.__name__}")
        
        return result
    
    return wrapper
def disk_backed(func):
    """
    Decorator to make a function use disk-backed arrays for large outputs
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Check if disk_backed parameter was passed
        use_disk = kwargs.pop('disk_backed', False)
        threshold = kwargs.pop('disk_threshold', 100000)  # Default 100K items
        
        # Call the original function
        result = func(*args, **kwargs)
        
        # Convert the result to disk-backed array if needed
        if use_disk and isinstance(result, list) and len(result) > threshold:
            logger.info(f"Converting large result ({len(result)} items) to disk-backed array")
            return DiskBackedArray(result)
        
        return result
    
    return wrapper
