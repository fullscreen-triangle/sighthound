"""
Visualization utilities for performance profiling
"""

import os
import time
import logging
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime

from utils.profiler import Profiler, ExecutionStats

# Configure logger
logger = logging.getLogger(__name__)

class ProfilerVisualizer:
    """
    Visualizer for profiling data to identify bottlenecks
    """

    @staticmethod
    def generate_performance_report(output_dir: str) -> str:
        """
        Generate comprehensive performance report with visualizations
        
        Args:
            output_dir: Directory to save the report and visualizations
            
        Returns:
            Path to the generated report
        """
        # Ensure directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Get profiling data
        profiler = Profiler.get_instance()
        stats = profiler.get_stats()
        
        if not stats:
            logger.warning("No profiling data available")
            return ""
        
        # Generate text report
        report_path = os.path.join(output_dir, "performance_report.txt")
        report_content = profiler.generate_report(report_path)
        
        # Generate visualizations
        ProfilerVisualizer.plot_execution_times(stats, output_dir)
        ProfilerVisualizer.plot_function_comparison(stats, output_dir)
        ProfilerVisualizer.plot_memory_usage(stats, output_dir)
        ProfilerVisualizer.plot_call_frequency(stats, output_dir)
        
        # Add plots to HTML report
        html_report_path = os.path.join(output_dir, "performance_report.html")
        ProfilerVisualizer.generate_html_report(stats, html_report_path)
        
        return report_path
    
    @staticmethod
    def plot_execution_times(stats: Dict[str, ExecutionStats], output_dir: str) -> None:
        """
        Plot execution times for all profiled functions
        
        Args:
            stats: Dictionary mapping function names to ExecutionStats
            output_dir: Directory to save the plot
        """
        # Prepare data
        sorted_stats = sorted(
            stats.values(),
            key=lambda x: x.total_time,
            reverse=True
        )
        
        # Take top 20 functions to avoid cluttered plot
        top_functions = sorted_stats[:20]
        
        # Create plot
        fig, ax = plt.figure(figsize=(12, 8)), plt.gca()
        
        names = [stat.name for stat in top_functions]
        total_times = [stat.total_time for stat in top_functions]
        avg_times = [stat.avg_time for stat in top_functions]
        
        # Create horizontal bar chart
        y_pos = np.arange(len(names))
        ax.barh(y_pos, total_times, align='center', alpha=0.5, label='Total Time')
        
        # Add average time as dots
        ax.scatter(avg_times, y_pos, color='red', label='Avg Time per Call')
        
        # Add labels and legend
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names)
        ax.invert_yaxis()  # Labels read top-to-bottom
        ax.set_xlabel('Time (seconds)')
        ax.set_title('Function Execution Times')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "execution_times.png"))
        plt.close()
    
    @staticmethod
    def plot_function_comparison(stats: Dict[str, ExecutionStats], output_dir: str) -> None:
        """
        Plot comparison of function calls, showing the relative time spent in each function
        
        Args:
            stats: Dictionary mapping function names to ExecutionStats
            output_dir: Directory to save the plot
        """
        # Prepare data
        func_names = []
        call_counts = []
        total_times = []
        avg_times = []
        
        for name, stat in stats.items():
            func_names.append(name)
            call_counts.append(stat.call_counts)
            total_times.append(stat.total_time)
            avg_times.append(stat.avg_time)
        
        # Create DataFrame for easier manipulation
        df = pd.DataFrame({
            'function': func_names,
            'calls': call_counts,
            'total_time': total_times,
            'avg_time': avg_times
        })
        
        # Sort by total time
        df = df.sort_values('total_time', ascending=False).head(15)
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Pie chart for total time
        ax1.pie(df['total_time'], labels=df['function'], autopct='%1.1f%%',
                shadow=True, startangle=90)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        ax1.set_title('Share of Total Execution Time')
        
        # Bar chart for avg time vs calls
        ax2.bar(df['function'], df['avg_time'], label='Avg Time per Call (s)')
        ax2_twin = ax2.twinx()
        ax2_twin.plot(df['function'], df['calls'], 'r-', label='Number of Calls')
        
        ax2.set_xticklabels(df['function'], rotation=45, ha='right')
        ax2.set_ylabel('Average Time (s)')
        ax2_twin.set_ylabel('Number of Calls')
        ax2.set_title('Average Time vs Number of Calls')
        
        # Add legend
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "function_comparison.png"))
        plt.close()
    
    @staticmethod
    def plot_memory_usage(stats: Dict[str, ExecutionStats], output_dir: str) -> None:
        """
        Plot memory usage for functions that recorded memory stats
        
        Args:
            stats: Dictionary mapping function names to ExecutionStats
            output_dir: Directory to save the plot
        """
        # Filter to functions with memory data
        memory_stats = {}
        for name, stat in stats.items():
            if stat.memory_usage:
                memory_stats[name] = stat
        
        if not memory_stats:
            logger.warning("No memory usage data available")
            return
        
        # Create plot
        fig, ax = plt.figure(figsize=(12, 8)), plt.gca()
        
        # Prepare data
        func_names = []
        avg_memory = []
        
        for name, stat in memory_stats.items():
            func_names.append(name)
            avg_memory.append(np.mean(stat.memory_usage))
        
        # Sort by average memory usage
        sorted_indices = np.argsort(avg_memory)[::-1]
        func_names = [func_names[i] for i in sorted_indices]
        avg_memory = [avg_memory[i] for i in sorted_indices]
        
        # Take top 15 to avoid cluttered plot
        func_names = func_names[:15]
        avg_memory = avg_memory[:15]
        
        # Create horizontal bar chart
        y_pos = np.arange(len(func_names))
        ax.barh(y_pos, avg_memory, align='center', alpha=0.7)
        
        # Add labels
        ax.set_yticks(y_pos)
        ax.set_yticklabels(func_names)
        ax.invert_yaxis()  # Labels read top-to-bottom
        ax.set_xlabel('Average Memory Usage (MB)')
        ax.set_title('Function Memory Usage')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "memory_usage.png"))
        plt.close()
    
    @staticmethod
    def plot_call_frequency(stats: Dict[str, ExecutionStats], output_dir: str) -> None:
        """
        Plot call frequency distribution
        
        Args:
            stats: Dictionary mapping function names to ExecutionStats
            output_dir: Directory to save the plot
        """
        # Prepare data
        func_names = []
        call_counts = []
        
        for name, stat in stats.items():
            func_names.append(name)
            call_counts.append(stat.call_counts)
        
        # Sort by call count
        sorted_indices = np.argsort(call_counts)[::-1]
        func_names = [func_names[i] for i in sorted_indices]
        call_counts = [call_counts[i] for i in sorted_indices]
        
        # Take top 20 to avoid cluttered plot
        func_names = func_names[:20]
        call_counts = call_counts[:20]
        
        # Create plot
        fig, ax = plt.figure(figsize=(12, 8)), plt.gca()
        
        # Create horizontal bar chart
        y_pos = np.arange(len(func_names))
        ax.barh(y_pos, call_counts, align='center', alpha=0.7)
        
        # Add labels
        ax.set_yticks(y_pos)
        ax.set_yticklabels(func_names)
        ax.invert_yaxis()  # Labels read top-to-bottom
        ax.set_xlabel('Number of Calls')
        ax.set_title('Function Call Frequency')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "call_frequency.png"))
        plt.close()
    
    @staticmethod
    def generate_html_report(stats: Dict[str, ExecutionStats], output_path: str) -> None:
        """
        Generate HTML report with interactive visualizations
        
        Args:
            stats: Dictionary mapping function names to ExecutionStats
            output_path: Path to save the HTML report
        """
        # HTML report template
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Performance Profiling Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #333; text-align: center; }
                .section { margin-bottom: 30px; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                tr:nth-child(even) { background-color: #f9f9f9; }
                .image-container { text-align: center; margin: 20px 0; }
                .image-container img { max-width: 100%; height: auto; }
                .summary { background-color: #f0f0f0; padding: 15px; border-radius: 5px; }
            </style>
        </head>
        <body>
            <h1>Performance Profiling Report</h1>
            <div class="section summary">
                <h2>Summary</h2>
                <p><strong>Total Functions Profiled:</strong> {total_functions}</p>
                <p><strong>Total Execution Time:</strong> {total_time:.2f} seconds</p>
                <p><strong>Most Called Function:</strong> {most_called_function} ({most_called_count} calls)</p>
                <p><strong>Slowest Function:</strong> {slowest_function} ({slowest_time:.2f} seconds total)</p>
                <p><strong>Generated:</strong> {generation_time}</p>
            </div>
            
            <div class="section">
                <h2>Execution Times</h2>
                <div class="image-container">
                    <img src="execution_times.png" alt="Execution Times Graph">
                </div>
            </div>
            
            <div class="section">
                <h2>Function Comparison</h2>
                <div class="image-container">
                    <img src="function_comparison.png" alt="Function Comparison Graph">
                </div>
            </div>
            
            <div class="section">
                <h2>Memory Usage</h2>
                <div class="image-container">
                    <img src="memory_usage.png" alt="Memory Usage Graph">
                </div>
            </div>
            
            <div class="section">
                <h2>Call Frequency</h2>
                <div class="image-container">
                    <img src="call_frequency.png" alt="Call Frequency Graph">
                </div>
            </div>
            
            <div class="section">
                <h2>Detailed Statistics</h2>
                <table>
                    <tr>
                        <th>Function</th>
                        <th>Calls</th>
                        <th>Total Time (s)</th>
                        <th>Avg Time (s)</th>
                        <th>Min Time (s)</th>
                        <th>Max Time (s)</th>
                    </tr>
                    {table_rows}
                </table>
            </div>
        </body>
        </html>
        """
        
        # Prepare summary data
        total_functions = len(stats)
        total_time = sum(stat.total_time for stat in stats.values())
        
        # Find most called function
        most_called_function = ""
        most_called_count = 0
        for name, stat in stats.items():
            if stat.call_counts > most_called_count:
                most_called_count = stat.call_counts
                most_called_function = name
        
        # Find slowest function (by total time)
        slowest_function = ""
        slowest_time = 0
        for name, stat in stats.items():
            if stat.total_time > slowest_time:
                slowest_time = stat.total_time
                slowest_function = name
        
        # Generate table rows
        table_rows = ""
        sorted_stats = sorted(stats.items(), key=lambda x: x[1].total_time, reverse=True)
        for name, stat in sorted_stats:
            table_rows += f"""
            <tr>
                <td>{name}</td>
                <td>{stat.call_counts}</td>
                <td>{stat.total_time:.6f}</td>
                <td>{stat.avg_time:.6f}</td>
                <td>{stat.min_time:.6f}</td>
                <td>{stat.max_time:.6f}</td>
            </tr>
            """
        
        # Generate HTML
        html_content = html_template.format(
            total_functions=total_functions,
            total_time=total_time,
            most_called_function=most_called_function,
            most_called_count=most_called_count,
            slowest_function=slowest_function,
            slowest_time=slowest_time,
            generation_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            table_rows=table_rows
        )
        
        # Save HTML report
        with open(output_path, 'w') as f:
            f.write(html_content)
            
        logger.info(f"HTML performance report saved to {output_path}")

def profile_application(output_dir: str = "profiling_results") -> str:
    """
    Profile the entire application and generate a report
    
    Args:
        output_dir: Directory to save profiling results
        
    Returns:
        Path to the generated report
    """
    logger.info("Generating performance profiling report...")
    return ProfilerVisualizer.generate_performance_report(output_dir)
