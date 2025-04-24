"""
Profiling utilities for performance optimization
"""

import cProfile
import pstats
import io
import time
import functools
import logging
import os
from typing import Optional, Callable, Dict, Any, Union, List
import numpy as np
import matplotlib.pyplot as plt
from contextlib import contextmanager
import psutil
import json
from datetime import datetime
from pathlib import Path

# Configure logger
logger = logging.getLogger(__name__)

class ExecutionStats:
    """Class to store and analyze function execution statistics"""
    
    def __init__(self, name: str):
        self.name = name
        self.execution_times: List[float] = []
        self.call_counts: int = 0
        self.total_time: float = 0.0
        self.avg_time: float = 0.0
        self.min_time: float = float('inf')
        self.max_time: float = 0.0
        self.memory_usage: List[float] = []
        
    def add_execution(self, execution_time: float, memory_used: Optional[float] = None) -> None:
        """
        Add execution statistics
        
        Args:
            execution_time: Time taken for execution in seconds
            memory_used: Memory used in MB (optional)
        """
        self.execution_times.append(execution_time)
        self.call_counts += 1
        self.total_time += execution_time
        self.avg_time = self.total_time / self.call_counts
        self.min_time = min(self.min_time, execution_time)
        self.max_time = max(self.max_time, execution_time)
        
        if memory_used is not None:
            self.memory_usage.append(memory_used)
            
    def get_summary(self) -> Dict[str, Union[str, float, int]]:
        """
        Get summary of execution statistics
        
        Returns:
            Dictionary with execution statistics
        """
        return {
            "name": self.name,
            "calls": self.call_counts,
            "total_time": self.total_time,
            "avg_time": self.avg_time,
            "min_time": self.min_time,
            "max_time": self.max_time,
            "avg_memory_mb": np.mean(self.memory_usage) if self.memory_usage else None
        }
        
    def plot_performance(self, output_path: Optional[str] = None) -> None:
        """
        Plot performance statistics
        
        Args:
            output_path: Path to save the plot (optional)
        """
        if not self.execution_times:
            logger.warning(f"No execution data available for {self.name}")
            return
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot execution times
        x = range(len(self.execution_times))
        ax1.plot(x, self.execution_times)
        ax1.set_title(f"Execution Time: {self.name}")
        ax1.set_xlabel("Call #")
        ax1.set_ylabel("Time (s)")
        ax1.grid(True)
        
        # Add moving average
        if len(self.execution_times) > 1:
            window = min(5, len(self.execution_times))
            moving_avg = np.convolve(
                self.execution_times, 
                np.ones(window) / window, 
                mode='valid'
            )
            ax1.plot(
                range(window-1, window-1+len(moving_avg)), 
                moving_avg,
                'r--',
                label=f"{window}-call Moving Avg"
            )
            ax1.legend()
        
        # Plot memory usage if available
        if self.memory_usage:
            ax2.plot(range(len(self.memory_usage)), self.memory_usage)
            ax2.set_title(f"Memory Usage: {self.name}")
            ax2.set_xlabel("Call #")
            ax2.set_ylabel("Memory (MB)")
            ax2.grid(True)
        else:
            ax2.set_title("Memory Usage: Not Available")
            
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
            logger.info(f"Performance plot saved to {output_path}")
        else:
            plt.show()

class Profiler:
    """
    Utility class for profiling code execution
    """
    
    _instance = None
    _stats: Dict[str, ExecutionStats] = {}
    _enabled = True
    
    @classmethod
    def get_instance(cls) -> 'Profiler':
        """Get singleton instance"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
        
    @classmethod
    def enable(cls) -> None:
        """Enable profiling"""
        cls._enabled = True
        
    @classmethod
    def disable(cls) -> None:
        """Disable profiling"""
        cls._enabled = False
        
    @classmethod
    def is_enabled(cls) -> bool:
        """Check if profiling is enabled"""
        return cls._enabled
    
    @classmethod
    def get_stats(cls, function_name: Optional[str] = None) -> Union[Dict[str, ExecutionStats], Optional[ExecutionStats]]:
        """
        Get collected statistics
        
        Args:
            function_name: Name of function to get stats for (optional)
            
        Returns:
            Dictionary of ExecutionStats objects or a single ExecutionStats object
        """
        if function_name:
            return cls._stats.get(function_name)
        return cls._stats
    
    @classmethod
    def clear_stats(cls) -> None:
        """Clear all collected statistics"""
        cls._stats.clear()
        
    @classmethod
    def record_execution(cls, function_name: str, execution_time: float, memory_used: Optional[float] = None) -> None:
        """
        Record function execution statistics
        
        Args:
            function_name: Name of the function
            execution_time: Time taken for execution in seconds
            memory_used: Memory used in MB (optional)
        """
        if not cls._enabled:
            return
            
        if function_name not in cls._stats:
            cls._stats[function_name] = ExecutionStats(function_name)
            
        cls._stats[function_name].add_execution(execution_time, memory_used)
    
    @classmethod
    def generate_report(cls, output_path: Optional[str] = None) -> str:
        """
        Generate performance report
        
        Args:
            output_path: Path to save the report (optional)
            
        Returns:
            Report as a string
        """
        if not cls._stats:
            return "No profiling data available"
            
        report = ["Performance Report", "=================", ""]
        
        # Sort functions by total time
        sorted_stats = sorted(
            cls._stats.values(),
            key=lambda x: x.total_time,
            reverse=True
        )
        
        # Add table header
        report.append(f"{'Function':<30} {'Calls':<10} {'Total (s)':<12} {'Avg (s)':<12} {'Min (s)':<12} {'Max (s)':<12}")
        report.append("-" * 90)
        
        # Add function statistics
        for stat in sorted_stats:
            report.append(
                f"{stat.name:<30} {stat.call_counts:<10} {stat.total_time:<12.6f} "
                f"{stat.avg_time:<12.6f} {stat.min_time:<12.6f} {stat.max_time:<12.6f}"
            )
            
        report_str = "\n".join(report)
        
        # Save to file if output path is provided
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_str)
            logger.info(f"Performance report saved to {output_path}")
            
        return report_str
    
    @classmethod
    def plot_all(cls, output_dir: Optional[str] = None) -> None:
        """
        Plot all collected statistics
        
        Args:
            output_dir: Directory to save plots (optional)
        """
        if not cls._stats:
            logger.warning("No profiling data available for plotting")
            return
            
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        for name, stat in cls._stats.items():
            output_path = None
            if output_dir:
                output_path = os.path.join(output_dir, f"{name.replace(' ', '_')}_profile.png")
            stat.plot_performance(output_path)

class PerformanceProfiler:
    """
    Utility for profiling code performance and identifying bottlenecks
    """
    
    def __init__(self, output_dir: str = "profiling_results"):
        """
        Initialize the profiler
        
        Args:
            output_dir: Directory to save profiling results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.profiler = cProfile.Profile()
        self.memory_usage = []
        self.timing_results = {}
        self.resource_usage = {}
        
    def profile_function(self, func: Callable, *args, **kwargs) -> Any:
        """
        Profile a function and return its result
        
        Args:
            func: Function to profile
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            Result of the function
        """
        # Enable profiling
        self.profiler.enable()
        
        # Start memory tracking in a separate thread
        memory_tracking_active = True
        
        def track_memory():
            process = psutil.Process(os.getpid())
            while memory_tracking_active:
                self.memory_usage.append(process.memory_info().rss / 1024 / 1024)  # MB
                time.sleep(0.1)
        
        import threading
        memory_thread = threading.Thread(target=track_memory)
        memory_thread.daemon = True
        memory_thread.start()
        
        # Execute the function
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
        finally:
            # Stop profiling
            end_time = time.time()
            self.profiler.disable()
            memory_tracking_active = False
            memory_thread.join(timeout=1.0)
        
        # Store timing results
        func_name = func.__name__
        self.timing_results[func_name] = {
            'elapsed_time': end_time - start_time,
            'args': str(args),
            'kwargs': str(kwargs)
        }
        
        # Store resource usage
        process = psutil.Process(os.getpid())
        self.resource_usage[func_name] = {
            'cpu_percent': process.cpu_percent(),
            'memory_percent': process.memory_percent(),
            'peak_memory_mb': max(self.memory_usage) if self.memory_usage else 0,
            'memory_profile': self.memory_usage.copy()
        }
        self.memory_usage.clear()
        
        return result
    
    def generate_profile_stats(self, top_n: int = 20) -> Dict[str, Any]:
        """
        Generate profiling statistics
        
        Args:
            top_n: Number of top functions to include in statistics
            
        Returns:
            Dictionary of profiling statistics
        """
        s = io.StringIO()
        ps = pstats.Stats(self.profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(top_n)
        
        # Parse the text output to get stats in a structured format
        stats_text = s.getvalue()
        lines = stats_text.strip().split('\n')
        
        # Find the start of the stats
        for i, line in enumerate(lines):
            if 'ncalls' in line and 'tottime' in line and 'percall' in line:
                start_idx = i + 1
                break
        else:
            start_idx = 0
        
        # Parse the stats
        function_stats = []
        for line in lines[start_idx:]:
            if line.strip() and not line.startswith(' '):
                parts = line.split()
                if len(parts) >= 6:
                    function_stats.append({
                        'ncalls': parts[0],
                        'tottime': float(parts[1]),
                        'percall': float(parts[2]),
                        'cumtime': float(parts[3]),
                        'percall_cum': float(parts[4]),
                        'function': ' '.join(parts[5:])
                    })
        
        # Create the stats dictionary
        stats = {
            'timing_results': self.timing_results,
            'resource_usage': self.resource_usage,
            'function_stats': function_stats[:top_n],
            'total_functions': len(pstats.Stats(self.profiler).stats),
            'timestamp': datetime.now().isoformat()
        }
        
        return stats
    
    def save_profile_results(self, filename_prefix: str, stats: Optional[Dict[str, Any]] = None) -> str:
        """
        Save profiling results to files
        
        Args:
            filename_prefix: Prefix for output filenames
            stats: Stats dictionary, if None it will be generated
            
        Returns:
            Path to the stats file
        """
        if stats is None:
            stats = self.generate_profile_stats()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{filename_prefix}_{timestamp}"
        
        # Save stats to JSON
        stats_file = self.output_dir / f"{base_filename}_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Generate and save a visual profile
        self._generate_profile_visualization(stats, base_filename)
        
        return str(stats_file)
    
    def _generate_profile_visualization(self, stats: Dict[str, Any], base_filename: str) -> None:
        """Generate visual profile charts"""
        # Create figures directory
        figures_dir = self.output_dir / "figures"
        figures_dir.mkdir(exist_ok=True)
        
        # 1. Function time distribution
        plt.figure(figsize=(12, 8))
        
        # Sort by cumulative time
        func_stats = sorted(stats['function_stats'], key=lambda x: x['cumtime'], reverse=True)[:10]
        func_names = [x['function'].split('/')[-1] for x in func_stats]  # Simplified names
        cum_times = [x['cumtime'] for x in func_stats]
        
        plt.barh(func_names, cum_times)
        plt.xlabel('Cumulative Time (s)')
        plt.ylabel('Function')
        plt.title('Top 10 Functions by Cumulative Time')
        plt.tight_layout()
        plt.savefig(figures_dir / f"{base_filename}_time_distribution.png")
        plt.close()
        
        # 2. Memory profile over time
        for func_name, resources in stats['resource_usage'].items():
            if 'memory_profile' in resources and resources['memory_profile']:
                plt.figure(figsize=(10, 6))
                plt.plot(resources['memory_profile'])
                plt.xlabel('Measurement (0.1s intervals)')
                plt.ylabel('Memory Usage (MB)')
                plt.title(f'Memory Usage: {func_name}')
                plt.grid(True)
                plt.savefig(figures_dir / f"{base_filename}_{func_name}_memory.png")
                plt.close()
    
    def profile_code_block(self, name: str) -> Callable:
        """
        Decorator to profile a code block
        
        Args:
            name: Name for the profiling results
            
        Returns:
            Decorator function
        """
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                result = self.profile_function(func, *args, **kwargs)
                stats = self.generate_profile_stats()
                self.save_profile_results(name, stats)
                return result
            return wrapper
        return decorator

class PerformanceMetrics:
    """
    Class for collecting and reporting performance metrics
    """
    
    def __init__(self):
        self.metrics = {
            'function_calls': {},
            'execution_times': {},
            'memory_usage': {},
            'throughput': {}
        }
    
    def record_execution_time(self, function_name: str, execution_time: float) -> None:
        """Record function execution time"""
        if function_name not in self.metrics['execution_times']:
            self.metrics['execution_times'][function_name] = []
        
        self.metrics['execution_times'][function_name].append(execution_time)
    
    def record_memory_usage(self, function_name: str, memory_mb: float) -> None:
        """Record memory usage"""
        if function_name not in self.metrics['memory_usage']:
            self.metrics['memory_usage'][function_name] = []
        
        self.metrics['memory_usage'][function_name].append(memory_mb)
    
    def record_function_call(self, function_name: str) -> None:
        """Record function call count"""
        if function_name not in self.metrics['function_calls']:
            self.metrics['function_calls'][function_name] = 0
        
        self.metrics['function_calls'][function_name] += 1
    
    def record_throughput(self, operation_name: str, items_processed: int, time_taken: float) -> None:
        """Record throughput (items per second)"""
        if operation_name not in self.metrics['throughput']:
            self.metrics['throughput'][operation_name] = []
        
        throughput = items_processed / time_taken if time_taken > 0 else 0
        self.metrics['throughput'][operation_name].append({
            'items': items_processed,
            'time': time_taken,
            'throughput': throughput
        })
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all metrics"""
        summary = {}
        
        # Function call counts
        summary['function_calls'] = dict(self.metrics['function_calls'])
        
        # Execution times
        summary['execution_times'] = {}
        for func, times in self.metrics['execution_times'].items():
            if times:
                summary['execution_times'][func] = {
                    'min': min(times),
                    'max': max(times),
                    'avg': sum(times) / len(times),
                    'p95': np.percentile(times, 95) if len(times) > 1 else times[0]
                }
        
        # Memory usage
        summary['memory_usage'] = {}
        for func, usages in self.metrics['memory_usage'].items():
            if usages:
                summary['memory_usage'][func] = {
                    'min': min(usages),
                    'max': max(usages),
                    'avg': sum(usages) / len(usages)
                }
        
        # Throughput
        summary['throughput'] = {}
        for op, data in self.metrics['throughput'].items():
            if data:
                throughputs = [d['throughput'] for d in data]
                summary['throughput'][op] = {
                    'min': min(throughputs),
                    'max': max(throughputs),
                    'avg': sum(throughputs) / len(throughputs),
                    'total_items': sum(d['items'] for d in data),
                    'total_time': sum(d['time'] for d in data)
                }
        
        return summary
    
    def reset(self) -> None:
        """Reset all metrics"""
        for category in self.metrics:
            self.metrics[category] = {}
    
    def to_json(self, path: str) -> None:
        """Save metrics to a JSON file"""
        with open(path, 'w') as f:
            json.dump(self.get_summary(), f, indent=2)

# Global performance metrics instance
performance_metrics = PerformanceMetrics()

def timer(func: Callable) -> Callable:
    """
    Decorator to time function execution
    
    Args:
        func: Function to time
        
    Returns:
        Wrapped function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        performance_metrics.record_function_call(func.__name__)
        
        # Track memory before and after
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        performance_metrics.record_execution_time(func.__name__, elapsed_time)
        
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        performance_metrics.record_memory_usage(func.__name__, mem_after - mem_before)
        
        logger.debug(f"{func.__name__} took {elapsed_time:.4f} seconds")
        return result
    
    return wrapper

def profile(output_prefix: Optional[str] = None) -> Callable:
    """
    Decorator to profile a function
    
    Args:
        output_prefix: Prefix for output filenames, defaults to function name
        
    Returns:
        Decorator function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            profiler = PerformanceProfiler()
            result = profiler.profile_function(func, *args, **kwargs)
            prefix = output_prefix or func.__name__
            profiler.save_profile_results(prefix)
            return result
        return wrapper
    
    return decorator

def profile(func=None, *, detailed=False):
    """
    Decorator to profile function execution
    
    Args:
        func: Function to profile
        detailed: Whether to run detailed profiling with cProfile
        
    Returns:
        Decorated function
    """
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            if not Profiler.is_enabled():
                return f(*args, **kwargs)
                
            function_name = f.__qualname__
                
            if detailed:
                # Use cProfile for detailed profiling
                pr = cProfile.Profile()
                pr.enable()
                
            # Measure execution time
            start_time = time.time()
            result = f(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Record basic timing information
            Profiler.record_execution(function_name, execution_time)
            
            if detailed:
                # Process detailed profiling information
                pr.disable()
                s = io.StringIO()
                ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
                ps.print_stats()
                
                logger.debug(f"Detailed profile for {function_name}:\n{s.getvalue()}")
                
            return result
        return wrapper
        
    if func is None:
        return decorator
    return decorator(func)
    
@contextmanager
def profile_block(name: str):
    """
    Context manager to profile a block of code
    
    Args:
        name: Name of the block for reporting
        
    Example:
        with profile_block("data_processing"):
            # Code to profile
            process_data()
    """
    if not Profiler.is_enabled():
        yield
        return
        
    start_time = time.time()
    yield
    execution_time = time.time() - start_time
    
    Profiler.record_execution(name, execution_time)
