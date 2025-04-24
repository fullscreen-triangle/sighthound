from typing import Dict, Type, Optional, List, Any
import os
from .base_parser import BaseParser, ParserResult
"""Parser module providing a plugin architecture for different file formats"""

from typing import Dict, Type, Optional, List, Any
import os
import importlib
import pkgutil
from .base_parser import BaseParser, ParserResult

class ParserRegistry:
    """
    Registry for parser classes that implements a plugin architecture
    Handles automatic registration of parsers that inherit from BaseParser
    """
    _parsers: Dict[str, Type[BaseParser]] = {}
    _extension_map: Dict[str, Type[BaseParser]] = {}
    
    @classmethod
    def register(cls, parser_class: Type[BaseParser]) -> None:
        """
        Register a parser class with its supported extensions
        
        Args:
            parser_class: Parser class to register
        """
        # Register by name
        cls._parsers[parser_class.PARSER_NAME] = parser_class
        
        # Register by extension
        for ext in parser_class.SUPPORTED_EXTENSIONS:
            ext = ext.lower()
            if ext in cls._extension_map:
                print(f"Warning: Extension {ext} already registered to "
                      f"{cls._extension_map[ext].PARSER_NAME}, now overridden by {parser_class.PARSER_NAME}")
            cls._extension_map[ext] = parser_class
    
    @classmethod
    def get_parser_by_name(cls, name: str) -> Optional[Type[BaseParser]]:
        """
        Get parser class by name
        
        Args:
            name: Name of the parser
            
        Returns:
            Parser class or None if not found
        """
        return cls._parsers.get(name)
    
    @classmethod
    def get_parser_for_file(cls, file_path: str) -> Optional[BaseParser]:
        """
        Get appropriate parser instance for the given file
        
        Args:
            file_path: Path to the file to be parsed
            
        Returns:
            Instance of appropriate parser or None if no parser is found
        """
        if not file_path:
            return None
            
        ext = os.path.splitext(file_path)[1].lower()
        parser_class = cls._extension_map.get(ext)
        
        if parser_class:
            return parser_class()
        return None
    
    @classmethod
    def parse_file(cls, file_path: str) -> ParserResult:
        """
        Parse a file using the appropriate parser
        
        Args:
            file_path: Path to the file to be parsed
            
        Returns:
            ParserResult containing parsing results or error information
        """
        parser = cls.get_parser_for_file(file_path)
        
        if not parser:
            ext = os.path.splitext(file_path)[1].lower()
            return ParserResult(
                success=False,
                error_message=f"No parser available for extension: {ext}",
                error_context={"file_path": file_path, "extension": ext}
            )
            
        return parser.parse(file_path)
    
    @classmethod
    def get_supported_extensions(cls) -> List[str]:
        """
        Get list of all supported file extensions
        
        Returns:
            List of supported extensions
        """
        return list(cls._extension_map.keys())
    
    @classmethod
    def get_all_parsers(cls) -> Dict[str, Type[BaseParser]]:
        """
        Get all registered parsers
        
        Returns:
            Dictionary of parser name to parser class
        """
        return cls._parsers.copy()


def autodiscover_parsers() -> None:
    """
    Automatically discover and register all parser classes in the parsers package
    """
    package_name = __name__
    for _, name, is_pkg in pkgutil.iter_modules(__path__, package_name + '.'):
        if not is_pkg and name != __name__:
            try:
                module = importlib.import_module(name)
                for item_name in dir(module):
                    item = getattr(module, item_name)
                    # Check if it's a class that inherits from BaseParser
                    if (isinstance(item, type) and 
                        issubclass(item, BaseParser) and 
                        item is not BaseParser):
                        ParserRegistry.register(item)
            except ImportError as e:
                print(f"Failed to import {name}: {e}")

# Auto-discover parsers when module is imported
autodiscover_parsers()
class ParserFactory:
    """
    Factory for creating parser instances based on file extension
    Supports dependency injection and plugin architecture
    """
    _parsers: Dict[str, Type[BaseParser]] = {}
    
    @classmethod
    def register_parser(cls, parser_class: Type[BaseParser]) -> None:
        """
        Register a parser class with its supported extensions
        
        Args:
            parser_class: Parser class to register
        """
        for ext in parser_class.SUPPORTED_EXTENSIONS:
            cls._parsers[ext.lower()] = parser_class
            
    @classmethod
    def get_parser(cls, file_path: str) -> Optional[BaseParser]:
        """
        Get appropriate parser instance for the given file
        
        Args:
            file_path: Path to the file to be parsed
            
        Returns:
            Instance of appropriate parser or None if no parser is found
        """
        if not file_path:
            return None
            
        ext = os.path.splitext(file_path)[1].lower()
        parser_class = cls._parsers.get(ext)
        
        if parser_class:
            return parser_class()
        return None
    
    @classmethod
    def parse_file(cls, file_path: str) -> ParserResult:
        """
        Parse a file using the appropriate parser
        
        Args:
            file_path: Path to the file to be parsed
            
        Returns:
            ParserResult containing parsing results or error information
        """
        parser = cls.get_parser(file_path)
        
        if not parser:
            ext = os.path.splitext(file_path)[1].lower()
            return ParserResult(
                success=False,
                error_message=f"No parser available for extension: {ext}",
                error_context={"file_path": file_path, "extension": ext}
            )
            
        return parser.parse(file_path)
    
    @classmethod
    def get_supported_extensions(cls) -> List[str]:
        """
        Get list of all supported file extensions
        
        Returns:
            List of supported extensions
        """
        return list(cls._parsers.keys())
"""Parser module providing a plugin architecture for different file formats"""

from typing import Dict, Type, Optional, List, Any
import os
import importlib
import pkgutil
import logging
from .base_parser import BaseParser, ParserResult

# Configure logger
logger = logging.getLogger(__name__)

class ParserRegistry:
    """
    Registry for parser classes that implements a plugin architecture
    Handles automatic registration of parsers that inherit from BaseParser
    """
    _parsers: Dict[str, Type[BaseParser]] = {}
    _extension_map: Dict[str, Type[BaseParser]] = {}
    
    @classmethod
    def register(cls, parser_class: Type[BaseParser]) -> None:
        """
        Register a parser class with its supported extensions
        
        Args:
            parser_class: Parser class to register
        """
        # Register by name
        cls._parsers[parser_class.PARSER_NAME] = parser_class
        logger.info(f"Registered parser: {parser_class.PARSER_NAME}")
        
        # Register by extension
        for ext in parser_class.SUPPORTED_EXTENSIONS:
            ext = ext.lower()
            if ext in cls._extension_map:
                logger.warning(f"Extension {ext} already registered to "
                              f"{cls._extension_map[ext].PARSER_NAME}, now overridden by {parser_class.PARSER_NAME}")
            cls._extension_map[ext] = parser_class
            logger.info(f"Registered extension {ext} to parser {parser_class.PARSER_NAME}")
    
    @classmethod
    def get_parser_by_name(cls, name: str) -> Optional[Type[BaseParser]]:
        """
        Get parser class by name
        
        Args:
            name: Name of the parser
            
        Returns:
            Parser class or None if not found
        """
        return cls._parsers.get(name)
    
    @classmethod
    def get_parser_for_file(cls, file_path: str) -> Optional[BaseParser]:
        """
        Get appropriate parser instance for the given file
        
        Args:
            file_path: Path to the file to be parsed
            
        Returns:
            Instance of appropriate parser or None if no parser is found
        """
        if not file_path:
            return None
            
        ext = os.path.splitext(file_path)[1].lower()
        parser_class = cls._extension_map.get(ext)
        
        if parser_class:
            return parser_class()
        return None
    
    @classmethod
    def parse_file(cls, file_path: str) -> ParserResult:
        """
        Parse a file using the appropriate parser
        
        Args:
            file_path: Path to the file to be parsed
            
        Returns:
            ParserResult containing parsing results or error information
        """
        parser = cls.get_parser_for_file(file_path)
        
        if not parser:
            ext = os.path.splitext(file_path)[1].lower()
            return ParserResult(
                success=False,
                error_message=f"No parser available for extension: {ext}",
                error_context={"file_path": file_path, "extension": ext}
            )
            
        return parser.parse(file_path)
    
    @classmethod
    def get_supported_extensions(cls) -> List[str]:
        """
        Get list of all supported file extensions
        
        Returns:
            List of supported extensions
        """
        return list(cls._extension_map.keys())
    
    @classmethod
    def get_all_parsers(cls) -> Dict[str, Type[BaseParser]]:
        """
        Get all registered parsers
        
        Returns:
            Dictionary of parser name to parser class
        """
        return cls._parsers.copy()


def autodiscover_parsers() -> None:
    """
    Automatically discover and register all parser classes in the parsers package
    """
    package_name = __name__
    for _, name, is_pkg in pkgutil.iter_modules(__path__, package_name + '.'):
        if not is_pkg and name != __name__:
            try:
                module = importlib.import_module(name)
                for item_name in dir(module):
                    item = getattr(module, item_name)
                    # Check if it's a class that inherits from BaseParser
                    if (isinstance(item, type) and 
                        issubclass(item, BaseParser) and 
                        item is not BaseParser):
                        ParserRegistry.register(item)
            except ImportError as e:
                logger.error(f"Failed to import {name}: {e}")

# Auto-discover parsers when module is imported
autodiscover_parsers()

# Decorator to register parsers
def register_parser(cls: Type[BaseParser]) -> Type[BaseParser]:
    """
    Decorator to register parser classes
    
    Args:
        cls: Parser class to register
        
    Returns:
        The same parser class (allows for stacking decorators)
    """
    ParserRegistry.register(cls)
    return cls
# Import and register all parsers
# This will be populated as parsers are imported elsewhere in the application