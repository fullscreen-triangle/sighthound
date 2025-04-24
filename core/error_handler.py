#!/usr/bin/env python3
import sys
import traceback
import logging
from typing import Dict, Optional, Any, List, Union, Callable

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("sighthound")

# Error type definitions
class ErrorType:
    """Error type constants."""
    INPUT_ERROR = "input_error"
    VALIDATION_ERROR = "validation_error"
    PROCESSING_ERROR = "processing_error"
    OUTPUT_ERROR = "output_error"
    SYSTEM_ERROR = "system_error"
    CONFIGURATION_ERROR = "configuration_error"
    NETWORK_ERROR = "network_error"
    PERMISSION_ERROR = "permission_error"
    DEPENDENCY_ERROR = "dependency_error"
    UNEXPECTED_ERROR = "unexpected_error"

# Verbosity levels
VERBOSITY_MINIMAL = 0
VERBOSITY_BASIC = 1
VERBOSITY_DETAILED = 2

class SighthoundError(Exception):
    """Base exception class for Sighthound errors with improved error messages."""
    
    def __init__(
        self, 
        message: str, 
        error_type: str = ErrorType.UNEXPECTED_ERROR,
        details: Optional[Dict[str, Any]] = None,
        troubleshooting_tips: Optional[List[str]] = None,
        original_exception: Optional[Exception] = None
    ):
        """Initialize the error with message and additional context."""
        self.message = message
        self.error_type = error_type
        self.details = details or {}
        self.troubleshooting_tips = troubleshooting_tips or []
        self.original_exception = original_exception
        super().__init__(message)
    
    def __str__(self) -> str:
        """Return the string representation of the error."""
        return self.message


class ErrorHandler:
    """Handler for processing and displaying user-friendly error messages."""
    
    def __init__(self, verbosity: int = VERBOSITY_BASIC, show_troubleshooting: bool = True):
        """
        Initialize the error handler.
        
        Args:
            verbosity: Level of detail in error messages
                0: Minimal - Just the error message
                1: Basic - Error message with brief context
                2: Detailed - Full error context including technical details
            show_troubleshooting: Whether to show troubleshooting tips
        """
        self.verbosity = verbosity
        self.show_troubleshooting = show_troubleshooting
        self.error_definitions: Dict[str, Dict[str, Union[str, List[str]]]] = {
            ErrorType.INPUT_ERROR: {
                "title": "Input Error",
                "description": "There was a problem with the input data or files.",
                "troubleshooting": [
                    "Check that all input files exist and are readable",
                    "Verify that input files are in the expected format",
                    "Ensure input data includes required fields (e.g., latitude/longitude)"
                ]
            },
            ErrorType.VALIDATION_ERROR: {
                "title": "Validation Error",
                "description": "The input data failed validation checks.",
                "troubleshooting": [
                    "Verify that your input data has the correct format",
                    "Check for missing required fields",
                    "Ensure values are within expected ranges"
                ]
            },
            ErrorType.PROCESSING_ERROR: {
                "title": "Processing Error",
                "description": "An error occurred while processing the data.",
                "troubleshooting": [
                    "Check if your dataset is too large or complex",
                    "Verify that the processing parameters are valid",
                    "Try processing a smaller subset of the data first"
                ]
            },
            ErrorType.OUTPUT_ERROR: {
                "title": "Output Error",
                "description": "Failed to generate or save output files.",
                "troubleshooting": [
                    "Check write permissions for the output directory",
                    "Verify that output directory exists",
                    "Ensure you have sufficient disk space"
                ]
            },
            ErrorType.SYSTEM_ERROR: {
                "title": "System Error",
                "description": "A system-level error occurred.",
                "troubleshooting": [
                    "Check available system resources (memory, disk space)",
                    "Verify that required system dependencies are installed",
                    "Try restarting the application"
                ]
            },
            ErrorType.CONFIGURATION_ERROR: {
                "title": "Configuration Error",
                "description": "There is an issue with the configuration.",
                "troubleshooting": [
                    "Check the configuration file syntax",
                    "Verify that all required configuration parameters are set",
                    "Ensure configuration values are within valid ranges"
                ]
            },
            ErrorType.NETWORK_ERROR: {
                "title": "Network Error",
                "description": "A network-related error occurred.",
                "troubleshooting": [
                    "Check your internet connection",
                    "Verify that required services are accessible",
                    "Check firewall or proxy settings"
                ]
            },
            ErrorType.PERMISSION_ERROR: {
                "title": "Permission Error",
                "description": "Insufficient permissions to perform the operation.",
                "troubleshooting": [
                    "Check file and directory permissions",
                    "Verify that you have necessary access rights",
                    "Try running the application with elevated privileges"
                ]
            },
            ErrorType.DEPENDENCY_ERROR: {
                "title": "Dependency Error",
                "description": "A required dependency is missing or incompatible.",
                "troubleshooting": [
                    "Install missing dependencies",
                    "Verify that installed dependencies meet version requirements",
                    "Check for conflicts between installed packages"
                ]
            },
            ErrorType.UNEXPECTED_ERROR: {
                "title": "Unexpected Error",
                "description": "An unexpected error occurred.",
                "troubleshooting": [
                    "Try restarting the application",
                    "Check application logs for more information",
                    "Report the issue with detailed steps to reproduce"
                ]
            }
        }
    
    def format_error(self, error: SighthoundError) -> str:
        """
        Format an error message based on verbosity level.
        
        Args:
            error: The SighthoundError instance
            
        Returns:
            str: Formatted error message
        """
        error_def = self.error_definitions.get(
            error.error_type, 
            self.error_definitions[ErrorType.UNEXPECTED_ERROR]
        )
        
        # Start with title and message
        title = error_def["title"]
        message = error.message
        
        if self.verbosity == VERBOSITY_MINIMAL:
            # Just the message
            result = f"{message}"
        elif self.verbosity == VERBOSITY_BASIC:
            # Title and message
            result = f"{title}: {message}"
        else:  # VERBOSITY_DETAILED
            # Title, message, and details
            result = f"{title}: {message}\n\n"
            
            if error.details:
                result += "Details:\n"
                for key, value in error.details.items():
                    result += f"  - {key}: {value}\n"
                result += "\n"
            
            if error.original_exception:
                result += f"Original error: {str(error.original_exception)}\n\n"
        
        # Add troubleshooting tips if enabled
        if self.show_troubleshooting:
            # Use custom tips if available, otherwise use default ones
            tips = error.troubleshooting if error.troubleshooting else error_def["troubleshooting"]
            if tips:
                result += "Troubleshooting tips:\n"
                for i, tip in enumerate(tips, 1):
                    result += f"  {i}. {tip}\n"
        
        return result
    
    def handle_error(
        self, 
        error: Union[SighthoundError, Exception], 
        exit_on_error: bool = False,
        log_error: bool = True
    ) -> None:
        """
        Handle an error by formatting and displaying it appropriately.
        
        Args:
            error: The error to handle
            exit_on_error: Whether to exit the program after handling the error
            log_error: Whether to log the error
        """
        # Convert standard exceptions to SighthoundError
        if not isinstance(error, SighthoundError):
            sighthound_error = SighthoundError(
                message=str(error),
                error_type=ErrorType.UNEXPECTED_ERROR,
                original_exception=error
            )
        else:
            sighthound_error = error
        
        # Format the error message
        error_message = self.format_error(sighthound_error)
        
        # Log the error if requested
        if log_error:
            logger.error(
                f"Error ({sighthound_error.error_type}): {sighthound_error.message}",
                exc_info=sighthound_error.original_exception
            )
        
        # Print the error message to stderr
        print(error_message, file=sys.stderr)
        
        # Exit if requested
        if exit_on_error:
            sys.exit(1)


def safe_execute(
    func: Callable, 
    error_handler: ErrorHandler = None,
    error_type: str = ErrorType.UNEXPECTED_ERROR,
    error_message: str = "An error occurred during execution",
    exit_on_error: bool = False,
    log_error: bool = True,
    *args, 
    **kwargs
) -> Any:
    """
    Execute a function safely with error handling.
    
    Args:
        func: Function to execute
        error_handler: ErrorHandler instance (creates default if None)
        error_type: Type of error to use if an exception occurs
        error_message: Message to use if an exception occurs
        exit_on_error: Whether to exit on error
        log_error: Whether to log the error
        *args: Arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        The return value of the function, or None if an error occurred
    """
    if error_handler is None:
        error_handler = ErrorHandler()
    
    try:
        return func(*args, **kwargs)
    except Exception as e:
        # Create a SighthoundError with the original exception
        error = SighthoundError(
            message=f"{error_message}: {str(e)}",
            error_type=error_type,
            original_exception=e,
            details={"traceback": traceback.format_exc()}
        )
        
        # Handle the error
        error_handler.handle_error(error, exit_on_error=exit_on_error, log_error=log_error)
        return None 