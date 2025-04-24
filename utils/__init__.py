"""Utility functions and classes for the Sighthound project"""

from typing import Dict, Any, Optional, Callable, TypeVar, Generic, Type
import logging
import traceback
import functools
import inspect
import os
from pathlib import Path

# Setup structured logging
logger = logging.getLogger("sighthound")

# Type variables for generics
T = TypeVar('T')
R = TypeVar('R')
"""Utility functions and classes for the Sighthound project"""



class ErrorContext:
    """Utility class to help build error context information"""
    
    @staticmethod
    def build_context(
        function_name: str, 
        args: Dict[str, Any] = None, 
        exception: Optional[Exception] = None,
        additional_info: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Build context dictionary for error reporting
        
        Args:
            function_name: Name of the function where error occurred
            args: Function arguments
            exception: Exception that was raised
            additional_info: Any additional information to include in context
            
        Returns:
            Dictionary with error context information
        """
        context = {
            "function": function_name,
        }
        
        if args:
            # Filter out sensitive information
            safe_args = {k: v for k, v in args.items() 
                        if not k.lower() in ['password', 'token', 'key', 'secret']}
            context["args"] = safe_args
            
        if exception:
            context["exception_type"] = type(exception).__name__
            context["exception_msg"] = str(exception)
            
        if additional_info:
            context.update(additional_info)
            
        return context
    
    @staticmethod
    def from_exception(exception: Exception) -> Dict[str, Any]:
        """
        Extract context information from exception
        
        Args:
            exception: Exception to extract context from
            
        Returns:
            Dictionary with error context
        """
        context = {
            "exception_type": type(exception).__name__,
            "exception_msg": str(exception),
        }
        
        # Add traceback information
        tb = traceback.extract_tb(exception.__traceback__)
        if tb:
            last_frame = tb[-1]
            context["file"] = last_frame.filename
            context["line"] = last_frame.lineno
            context["function"] = last_frame.name
            
        # Add custom context if available
        if hasattr(exception, 'context'):
            context.update(exception.context)
            
        return context

def with_error_context(
    error_handler: Optional[Callable[[Exception, Dict[str, Any]], Any]] = None
) -> Callable[[Callable[..., R]], Callable[..., R]]:
    """
    Decorator to add error context to function calls
    
    Args:
        error_handler: Optional function to handle errors with context
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable[..., R]) -> Callable[..., R]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Build context from function signature and arguments
                sig = inspect.signature(func)
                bound_args = sig.bind(*args, **kwargs)
                arg_dict = {k: v for k, v in bound_args.arguments.items()}
                
                context = ErrorContext.build_context(
                    function_name=func.__name__,
                    args=arg_dict,
                    exception=e
                )
                
                # Log the error with context
                logger.error(f"Error in {func.__name__}: {str(e)}", extra={"context": context})
                
                # Call error handler if provided
                if error_handler:
                    return error_handler(e, context)
                    
                # Re-raise with context if possible
                if hasattr(type(e), 'context'):
                    e.context = context
                
                raise
        return wrapper
    return decorator

class Singleton:
    """
    A non-thread-safe singleton implementation
    """
    _instances = {}
    
    def __new__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__new__(cls)
        return cls._instances[cls]

from typing import Dict, Any, Optional, Callable, TypeVar, Generic, Type, Union, List
import logging
import traceback
import functools
import inspect
import os
import json
from pathlib import Path

# Setup structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('sighthound.log')
    ]
)
logger = logging.getLogger("sighthound")

# Type variables for generics
T = TypeVar('T')
R = TypeVar('R')
S = TypeVar('S')

class ApplicationError(Exception):
    """Base exception for all application errors"""
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        self.context = context or {}
        super().__init__(message)

class ConfigurationError(ApplicationError):
    """Exception raised for configuration errors"""
    pass

class ValidationError(ApplicationError):
    """Exception raised for input validation errors"""
    pass

class ProcessingError(ApplicationError):
    """Exception raised for data processing errors"""
    pass

class ErrorContext:
    """Utility class to help build error context information"""
    
    @staticmethod
    def build_context(
        function_name: str, 
        args: Dict[str, Any] = None, 
        exception: Optional[Exception] = None,
        additional_info: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Build context dictionary for error reporting
        
        Args:
            function_name: Name of the function where error occurred
            args: Function arguments
            exception: Exception that was raised
            additional_info: Any additional information to include in context
            
        Returns:
            Dictionary with error context information
        """
        context = {
            "function": function_name,
        }
        
        if args:
            # Filter out sensitive information
            safe_args = {k: v for k, v in args.items() 
                        if not k.lower() in ['password', 'token', 'key', 'secret']}
            context["args"] = safe_args
            
        if exception:
            context["exception_type"] = type(exception).__name__
            context["exception_msg"] = str(exception)
            
        if additional_info:
            context.update(additional_info)
            
        return context
    
    @staticmethod
    def from_exception(exception: Exception) -> Dict[str, Any]:
        """
        Extract context information from exception
        
        Args:
            exception: Exception to extract context from
            
        Returns:
            Dictionary with error context
        """
        context = {
            "exception_type": type(exception).__name__,
            "exception_msg": str(exception),
        }
        
        # Add traceback information
        tb = traceback.extract_tb(exception.__traceback__)
        if tb:
            last_frame = tb[-1]
            context["file"] = last_frame.filename
            context["line"] = last_frame.lineno
            context["function"] = last_frame.name
            
        # Add custom context if available
        if hasattr(exception, 'context'):
            context.update(exception.context)
            
        return context

def with_error_context(
    error_handler: Optional[Callable[[Exception, Dict[str, Any]], Any]] = None
) -> Callable[[Callable[..., R]], Callable[..., R]]:
    """
    Decorator to add error context to function calls
    
    Args:
        error_handler: Optional function to handle errors with context
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable[..., R]) -> Callable[..., R]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Build context from function signature and arguments
                sig = inspect.signature(func)
                bound_args = sig.bind(*args, **kwargs)
                arg_dict = {k: v for k, v in bound_args.arguments.items()}
                
                context = ErrorContext.build_context(
                    function_name=func.__name__,
                    args=arg_dict,
                    exception=e
                )
                
                # Log the error with context
                logger.error(f"Error in {func.__name__}: {str(e)}", extra={"context": context})
                
                # Call error handler if provided
                if error_handler:
                    return error_handler(e, context)
                    
                # Re-raise with context if possible
                if hasattr(type(e), 'context'):
                    e.context = context
                
                raise
        return wrapper
    return decorator

class Singleton:
    """
    A non-thread-safe singleton implementation
    """
    _instances = {}
    
    def __new__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__new__(cls)
        return cls._instances[cls]

class Registry(Generic[T]):
    """
    Generic registry for components
    
    Args:
        base_class: Base class that all registered items must inherit from
    """
    def __init__(self, base_class: Type[T]):
        self._registry: Dict[str, Type[T]] = {}
        self._base_class = base_class
        
    def register(self, name: str, cls: Type[T]) -> None:
        """
        Register a class with a name
        
        Args:
            name: Name to register the class under
            cls: Class to register
        """
        if not issubclass(cls, self._base_class):
            raise TypeError(f"Class {cls.__name__} is not a subclass of {self._base_class.__name__}")
        self._registry[name] = cls
        
    def get(self, name: str) -> Optional[Type[T]]:
        """
        Get a registered class by name
        
        Args:
            name: Name of the registered class
            
        Returns:
            Registered class or None if not found
        """
        return self._registry.get(name)
        
    def create(self, name: str, *args, **kwargs) -> Optional[T]:
        """
        Create an instance of a registered class
        
        Args:
            name: Name of the registered class
            args: Positional arguments to pass to the constructor
            kwargs: Keyword arguments to pass to the constructor
            
        Returns:
            Instance of the registered class or None if not found
        """
        cls = self.get(name)
        if cls:
            return cls(*args, **kwargs)
        return None
        
    def get_all(self) -> Dict[str, Type[T]]:
        """
        Get all registered classes
        
        Returns:
            Dictionary mapping names to registered classes
        """
        return self._registry.copy()

class DependencyContainer(Singleton):
    """
    Simple dependency injection container
    """
    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._factories: Dict[str, Callable[['DependencyContainer'], Any]] = {}
        self._configs: Dict[str, Any] = {}
        
    def register_instance(self, name: str, instance: Any) -> None:
        """
        Register an instance with a name
        
        Args:
            name: Name to register the instance under
            instance: Instance to register
        """
        self._services[name] = instance
        
    def register_factory(self, name: str, factory: Callable[['DependencyContainer'], T]) -> None:
        """
        Register a factory function with a name
        
        Args:
            name: Name to register the factory under
            factory: Factory function that creates instances
        """
        self._factories[name] = factory
        
    def register_config(self, config_dict: Dict[str, Any]) -> None:
        """
        Register configuration values
        
        Args:
            config_dict: Dictionary of configuration values
        """
        self._configs.update(config_dict)
        
    def get_config(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value
        
        Args:
            key: Configuration key
            default: Default value if key is not found
            
        Returns:
            Configuration value or default
        """
        return self._configs.get(key, default)
        
    def get(self, name: str) -> Any:
        """
        Get a service by name
        
        Args:
            name: Name of the service
            
        Returns:
            Service instance
            
        Raises:
            KeyError: If service is not found
        """
        if name in self._services:
            return self._services[name]
            
        if name in self._factories:
            instance = self._factories[name](self)
            self._services[name] = instance
            return instance
            
        raise KeyError(f"Service '{name}' not registered")
        
    def has(self, name: str) -> bool:
        """
        Check if a service is registered
        
        Args:
            name: Name of the service
            
        Returns:
            True if service is registered, False otherwise
        """
        return name in self._services or name in self._factories

    def clear(self) -> None:
        """Clear all registered services and factories"""
        self._services.clear()
        self._factories.clear()

# Global dependency container instance
container = DependencyContainer()

def inject(*service_names: str) -> Callable[[Callable[..., R]], Callable[..., R]]:
    """
    Decorator to inject dependencies into a function
    
    Args:
        service_names: Names of services to inject
        
    Returns:
        Decorated function that receives injected dependencies
    """
    def decorator(func: Callable[..., R]) -> Callable[..., R]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Inject dependencies
            for name in service_names:
                if name not in kwargs:
                    try:
                        kwargs[name] = container.get(name)
                    except KeyError:
                        logger.warning(f"Service '{name}' not found for injection into {func.__name__}")
            return func(*args, **kwargs)
        return wrapper
    return decorator

class Config:
    """Configuration management with environment variable support"""
    _config: Dict[str, Any] = {}
    
    @classmethod
    def load_from_file(cls, file_path: str) -> None:
        """
        Load configuration from file
        
        Args:
            file_path: Path to configuration file
        """
        if not os.path.exists(file_path):
            raise ConfigurationError(f"Configuration file not found: {file_path}")
            
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.json':
            with open(file_path, 'r') as f:
                cls._config.update(json.load(f))
        elif ext in ['.yaml', '.yml']:
            try:
                import yaml
                with open(file_path, 'r') as f:
                    cls._config.update(yaml.safe_load(f))
            except ImportError:
                raise ConfigurationError("PyYAML package is required for loading YAML files")
        else:
            raise ConfigurationError(f"Unsupported configuration file format: {ext}")
    
    @classmethod
    def load_from_env(cls, prefix: str = "SIGHTHOUND_") -> None:
        """
        Load configuration from environment variables
        
        Args:
            prefix: Prefix for environment variables
        """
        for key, val in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix):].lower()
                
                # Try to parse as int, float, bool
                if val.isdigit():
                    cls._config[config_key] = int(val)
                elif val.replace('.', '', 1).isdigit() and val.count('.') < 2:
                    cls._config[config_key] = float(val)
                elif val.lower() in ['true', 'false']:
                    cls._config[config_key] = val.lower() == 'true'
                else:
                    cls._config[config_key] = val
    
    @classmethod
    def get(cls, key: str, default: Any = None) -> Any:
        """
        Get configuration value
        
        Args:
            key: Configuration key
            default: Default value if key is not found
            
        Returns:
            Configuration value or default
        """
        return cls._config.get(key, default)
    
    @classmethod
    def set(cls, key: str, value: Any) -> None:
        """
        Set configuration value
        
        Args:
            key: Configuration key
            value: Configuration value
        """
        cls._config[key] = value
    
    @classmethod
    def as_dict(cls) -> Dict[str, Any]:
        """
        Get configuration as dictionary
        
        Returns:
            Dictionary of configuration values
        """
        return cls._config.copy()

def validate_inputs(**validators: Callable[[Any], bool]) -> Callable[[Callable[..., R]], Callable[..., R]]:
    """
    Decorator to validate function inputs
    
    Args:
        validators: Dict mapping parameter names to validator functions
        
    Returns:
        Decorated function with input validation
    """
    def decorator(func: Callable[..., R]) -> Callable[..., R]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            
            for param_name, validator in validators.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    if not validator(value):
                        raise ValidationError(
                            f"Invalid value for parameter '{param_name}': {value}",
                            {"parameter": param_name, "value": value}
                        )
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Common validators
def not_none(value: Any) -> bool:
    """Validate that value is not None"""
    return value is not None

def not_empty(value: Any) -> bool:
    """Validate that value is not empty"""
    if value is None:
        return False
    if hasattr(value, '__len__'):
        return len(value) > 0
    return True

def is_positive(value: Union[int, float]) -> bool:
    """Validate that value is positive"""
    return value > 0

def is_non_negative(value: Union[int, float]) -> bool:
    """Validate that value is non-negative"""
    return value >= 0

def is_in_range(min_val: Union[int, float], max_val: Union[int, float]) -> Callable[[Union[int, float]], bool]:
    """
    Create validator for range check
    
    Args:
        min_val: Minimum value
        max_val: Maximum value
        
    Returns:
        Validator function
    """
    def validator(value: Union[int, float]) -> bool:
        return min_val <= value <= max_val
    return validator

def is_instance_of(cls: Type) -> Callable[[Any], bool]:
    """
    Create validator for type check
    
    Args:
        cls: Expected type
        
    Returns:
        Validator function
    """
    def validator(value: Any) -> bool:
        return isinstance(value, cls)
    return validator

def has_attrs(*attrs: str) -> Callable[[Any], bool]:
    """
    Create validator that checks if object has attributes
    
    Args:
        attrs: Required attributes
        
    Returns:
        Validator function
    """
    def validator(value: Any) -> bool:
        return all(hasattr(value, attr) for attr in attrs)
    return validator
class Registry(Generic[T]):
    """
    Generic registry for components
    
    Args:
        base_class: Base class that all registered items must inherit from
    """
    def __init__(self, base_class: Type[T]):
        self._registry: Dict[str, Type[T]] = {}
        self._base_class = base_class
        
    def register(self, name: str, cls: Type[T]) -> None:
        """
        Register a class with a name
        
        Args:
            name: Name to register the class under
            cls: Class to register
        """
        if not issubclass(cls, self._base_class):
            raise TypeError(f"Class {cls.__name__} is not a subclass of {self._base_class.__name__}")
        self._registry[name] = cls
        
    def get(self, name: str) -> Optional[Type[T]]:
        """
        Get a registered class by name
        
        Args:
            name: Name of the registered class
            
        Returns:
            Registered class or None if not found
        """
        return self._registry.get(name)
        
    def create(self, name: str, *args, **kwargs) -> Optional[T]:
        """
        Create an instance of a registered class
        
        Args:
            name: Name of the registered class
            args: Positional arguments to pass to the constructor
            kwargs: Keyword arguments to pass to the constructor
            
        Returns:
            Instance of the registered class or None if not found
        """
        cls = self.get(name)
        if cls:
            return cls(*args, **kwargs)
        return None
        
    def get_all(self) -> Dict[str, Type[T]]:
        """
        Get all registered classes
        
        Returns:
            Dictionary mapping names to registered classes
        """
        return self._registry.copy()

class DependencyContainer:
    """
    Simple dependency injection container
    """
    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._factories: Dict[str, Callable[..., Any]] = {}
        
    def register_instance(self, name: str, instance: Any) -> None:
        """
        Register an instance with a name
        
        Args:
            name: Name to register the instance under
            instance: Instance to register
        """
        self._services[name] = instance
        
    def register_factory(self, name: str, factory: Callable[..., T]) -> None:
        """
        Register a factory function with a name
        
        Args:
            name: Name to register the factory under
            factory: Factory function that creates instances
        """
        self._factories[name] = factory
        
    def get(self, name: str) -> Any:
        """
        Get a service by name
        
        Args:
            name: Name of the service
            
        Returns:
            Service instance
            
        Raises:
            KeyError: If service is not found
        """
        if name in self._services:
            return self._services[name]
            
        if name in self._factories:
            instance = self._factories[name](self)
            self._services[name] = instance
            return instance
            
        raise KeyError(f"Service '{name}' not registered")
        
    def has(self, name: str) -> bool:
        """
        Check if a service is registered
        
        Args:
            name: Name of the service
            
        Returns:
            True if service is registered, False otherwise
        """
        return name in self._services or name in self._factories

# Global dependency container instance
container = DependencyContainer()

def inject(*service_names: str) -> Callable[[Callable[..., R]], Callable[..., R]]:
    """
    Decorator to inject dependencies into a function
    
    Args:
        service_names: Names of services to inject
        
    Returns:
        Decorated function that receives injected dependencies
    """
    def decorator(func: Callable[..., R]) -> Callable[..., R]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Inject dependencies
            for name in service_names:
                if name not in kwargs:
                    try:
                        kwargs[name] = container.get(name)
                    except KeyError:
                        logger.warning(f"Service '{name}' not found for injection into {func.__name__}")
            return func(*args, **kwargs)
        return wrapper
    return decorator
class ErrorContext:
    """Utility class to help build error context information"""
    
    @staticmethod
    def build_context(
        function_name: str, 
        args: Dict[str, Any] = None, 
        exception: Optional[Exception] = None,
        additional_info: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Build context dictionary for error reporting
        
        Args:
            function_name: Name of the function where error occurred
            args: Function arguments
            exception: Exception that was raised
            additional_info: Any additional information to include in context
            
        Returns:
            Dictionary with error context information
        """
        context = {
            "function": function_name,
        }
        
        if args:
            context["args"] = args
            
        if exception:
            context["exception_type"] = type(exception).__name__
            context["exception_msg"] = str(exception)
            
        if additional_info:
            context.update(additional_info)
            
        return context
    
    @staticmethod
    def from_exception(exception: Exception) -> Dict[str, Any]:
        """
        Extract context information from exception
        
        Args:
            exception: Exception to extract context from
            
        Returns:
            Dictionary with error context
        """
        context = {
            "exception_type": type(exception).__name__,
            "exception_msg": str(exception),
        }
        
        # Add traceback information
        tb = traceback.extract_tb(exception.__traceback__)
        if tb:
            last_frame = tb[-1]
            context["file"] = last_frame.filename
            context["line"] = last_frame.lineno
            context["function"] = last_frame.name
            
        # Add custom context if available
        if hasattr(exception, 'context'):
            context.update(exception.context)
            
        return context

def with_error_context(
    error_handler: Optional[Callable[[Exception, Dict[str, Any]], Any]] = None
) -> Callable[[Callable[..., R]], Callable[..., R]]:
    """
    Decorator to add error context to function calls
    
    Args:
        error_handler: Optional function to handle errors with context
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable[..., R]) -> Callable[..., R]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Build context from function signature and arguments
                sig = inspect.signature(func)
                bound_args = sig.bind(*args, **kwargs)
                arg_dict = {k: v for k, v in bound_args.arguments.items()}
                
                context = ErrorContext.build_context(
                    function_name=func.__name__,
                    args=arg_dict,
                    exception=e
                )
                
                # Log the error with context
                logger.error(f"Error in {func.__name__}: {str(e)}", extra={"context": context})
                
                # Call error handler if provided
                if error_handler:
                    return error_handler(e, context)
                    
                # Re-raise with context if possible
                if hasattr(type(e), 'context'):
                    e.context = context
                
                raise
        return wrapper
    return decorator

class Singleton:
    """
    A non-thread-safe singleton implementation.
    """
    _instances = {}
    
    def __new__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__new__(cls)
        return cls._instances[cls]

class Registry(Generic[T]):
    """
    Generic registry for components
    
    Args:
        base_class: Base class that all registered items must inherit from
    """
    def __init__(self, base_class: Type[T]):
        self._registry: Dict[str, Type[T]] = {}
        self._base_class = base_class
        
    def register(self, name: str, cls: Type[T]) -> None:
        """
        Register a class with a name
        
        Args:
            name: Name to register the class under
            cls: Class to register
        """
        if not issubclass(cls, self._base_class):
            raise TypeError(f"Class {cls.__name__} is not a subclass of {self._base_class.__name__}")
        self._registry[name] = cls
        
    def get(self, name: str) -> Optional[Type[T]]:
        """
        Get a registered class by name
        
        Args:
            name: Name of the registered class
            
        Returns:
            Registered class or None if not found
        """
        return self._registry.get(name)
        
    def create(self, name: str, *args, **kwargs) -> Optional[T]:
        """
        Create an instance of a registered class
        
        Args:
            name: Name of the registered class
            args: Positional arguments to pass to the constructor
            kwargs: Keyword arguments to pass to the constructor
            
        Returns:
            Instance of the registered class or None if not found
        """
        cls = self.get(name)
        if cls:
            return cls(*args, **kwargs)
        return None
        
    def get_all(self) -> Dict[str, Type[T]]:
        """
        Get all registered classes
        
        Returns:
            Dictionary mapping names to registered classes
        """
        return self._registry.copy()