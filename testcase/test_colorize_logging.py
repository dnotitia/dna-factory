"""
Test cases for dna_factory/utils/colorize_logging.py

This module tests the logging colorization functionality.
"""

import logging
import sys
from pathlib import Path

import pytest

# Add parent directory to path to import dna_factory
sys.path.insert(0, str(Path(__file__).parent.parent))

from dna_factory.utils.colorize_logging import ColoredFormatter, format_logs_with_colors


class TestColoredFormatter:
    """Test cases for ColoredFormatter class"""
    
    def test_formatter_initialization(self):
        """Test that ColoredFormatter can be initialized"""
        formatter = ColoredFormatter('%(levelname)s - %(message)s')
        assert isinstance(formatter, logging.Formatter)
    
    def test_debug_level_color(self):
        """Test DEBUG level gets cyan color"""
        formatter = ColoredFormatter('%(levelname)s - %(message)s')
        record = logging.LogRecord(
            name='test',
            level=logging.DEBUG,
            pathname='test.py',
            lineno=1,
            msg='Test message',
            args=(),
            exc_info=None
        )
        
        formatted = formatter.format(record)
        # Should contain cyan color code
        assert '\033[36m' in formatted
        # Should contain reset code
        assert '\033[0m' in formatted
        # Should contain the message
        assert 'Test message' in formatted
    
    def test_info_level_color(self):
        """Test INFO level gets white color"""
        formatter = ColoredFormatter('%(levelname)s - %(message)s')
        record = logging.LogRecord(
            name='test',
            level=logging.INFO,
            pathname='test.py',
            lineno=1,
            msg='Info message',
            args=(),
            exc_info=None
        )
        
        formatted = formatter.format(record)
        # Should contain white color code
        assert '\033[37m' in formatted
        assert '\033[0m' in formatted
        assert 'Info message' in formatted
    
    def test_warning_level_color(self):
        """Test WARNING level gets yellow color"""
        formatter = ColoredFormatter('%(levelname)s - %(message)s')
        record = logging.LogRecord(
            name='test',
            level=logging.WARNING,
            pathname='test.py',
            lineno=1,
            msg='Warning message',
            args=(),
            exc_info=None
        )
        
        formatted = formatter.format(record)
        # Should contain yellow color code
        assert '\033[33m' in formatted
        assert '\033[0m' in formatted
        assert 'Warning message' in formatted
    
    def test_error_level_color(self):
        """Test ERROR level gets red color"""
        formatter = ColoredFormatter('%(levelname)s - %(message)s')
        record = logging.LogRecord(
            name='test',
            level=logging.ERROR,
            pathname='test.py',
            lineno=1,
            msg='Error message',
            args=(),
            exc_info=None
        )
        
        formatted = formatter.format(record)
        # Should contain red color code
        assert '\033[31m' in formatted
        assert '\033[0m' in formatted
        assert 'Error message' in formatted
    
    def test_critical_level_color(self):
        """Test CRITICAL level gets magenta color"""
        formatter = ColoredFormatter('%(levelname)s - %(message)s')
        record = logging.LogRecord(
            name='test',
            level=logging.CRITICAL,
            pathname='test.py',
            lineno=1,
            msg='Critical message',
            args=(),
            exc_info=None
        )
        
        formatted = formatter.format(record)
        # Should contain magenta color code
        assert '\033[35m' in formatted
        assert '\033[0m' in formatted
        assert 'Critical message' in formatted
    
    def test_formatter_with_different_format_string(self):
        """Test formatter with different format strings"""
        formatter = ColoredFormatter('[%(levelname)s] %(name)s: %(message)s')
        record = logging.LogRecord(
            name='test_logger',
            level=logging.INFO,
            pathname='test.py',
            lineno=1,
            msg='Test message',
            args=(),
            exc_info=None
        )
        
        formatted = formatter.format(record)
        assert 'test_logger' in formatted
        assert 'Test message' in formatted
        assert '\033[37m' in formatted  # INFO color


class TestFormatLogsWithColors:
    """Test cases for format_logs_with_colors function"""

    @pytest.mark.parametrize(
        ('variable_name', 'value'),
        [
            pytest.param('WORLD_SIZE', '4', id='multiple_processes'),
            pytest.param('WORLD_SIZE', '2', id='two_processes'),
            pytest.param('WORLD_SIZE', '1', id='single_process'),
            pytest.param('TEST_VAR', '100', id='large_value'),
            pytest.param('CUDA_VISIBLE_DEVICES', '0,1', id='gpu_list'),
        ],
    )
    def test_format_with_nonzero_value(self, monkeypatch, variable_name, value):
        """Nonzero environment values are highlighted, including GPU lists."""
        monkeypatch.setenv(variable_name, value)

        result = format_logs_with_colors(variable_name)

        assert result == f'\033[33m{variable_name}: {value}\033[0m'

    def test_format_with_nonexistent_variable(self, monkeypatch):
        """Test formatting when environment variable doesn't exist"""
        monkeypatch.delenv('NONEXISTENT_VAR', raising=False)

        result = format_logs_with_colors('NONEXISTENT_VAR')

        assert result == 'NONEXISTENT_VAR: None'

    def test_format_with_zero_value(self, monkeypatch):
        """Test formatting with zero value"""
        monkeypatch.setenv('TEST_VAR', '0')

        result = format_logs_with_colors('TEST_VAR')

        assert result == 'TEST_VAR: 0'


class TestColoredFormatterIntegration:
    """Integration tests for ColoredFormatter with actual logger"""
    
    def test_logger_with_colored_formatter(self):
        """Test using ColoredFormatter with an actual logger"""
        # Create a logger with colored formatter
        logger = logging.getLogger('test_colored_logger')
        logger.setLevel(logging.DEBUG)
        
        # Create a string handler to capture output
        handler = logging.StreamHandler()
        formatter = ColoredFormatter('%(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        # This should not raise any exceptions
        logger.debug('Debug message')
        logger.info('Info message')
        logger.warning('Warning message')
        logger.error('Error message')
        logger.critical('Critical message')
        
        # Cleanup
        logger.removeHandler(handler)
    
    def test_formatter_preserves_message_content(self):
        """Test that formatter preserves the actual message content"""
        formatter = ColoredFormatter('%(message)s')
        test_message = 'This is a test message with special characters: @#$%'
        
        record = logging.LogRecord(
            name='test',
            level=logging.INFO,
            pathname='test.py',
            lineno=1,
            msg=test_message,
            args=(),
            exc_info=None
        )
        
        formatted = formatter.format(record)
        # Should contain the original message (possibly with color codes)
        assert test_message in formatted


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
