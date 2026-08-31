"""
Test cases for dna_factory/utils/colorize_args.py

This module tests the argument parsing and colorization functionality.
"""

import os
import sys
import tempfile
import pytest
from pathlib import Path

# Add parent directory to path to import dna_factory
sys.path.insert(0, str(Path(__file__).parent.parent))

from dna_factory.utils.colorize_args import (
    parse_user_args,
    parse_yaml_config,
    colorize_user_args,
    format_args_with_colors
)


class TestParseUserArgs:
    """Test cases for parse_user_args function"""
    
    def test_parse_long_options(self):
        """Test parsing long options with -- prefix"""
        args = ['--learning_rate', '0.001', '--batch_size', '32']
        result = parse_user_args(args)
        assert 'learning_rate' in result
        assert 'batch_size' in result
    
    def test_parse_options_with_dashes(self):
        """Test that dashes in option names are converted to underscores"""
        args = ['--learning-rate', '0.001', '--per-device-batch-size', '32']
        result = parse_user_args(args)
        assert 'learning_rate' in result
        assert 'per_device_batch_size' in result
    
    def test_parse_short_options(self):
        """Test parsing short options with - prefix"""
        args = ['-n', '100', '-v']
        result = parse_user_args(args)
        assert 'n' in result
        assert 'v' in result
    
    def test_parse_boolean_flags(self):
        """Test parsing boolean flags without values"""
        args = ['--verbose', '--debug']
        result = parse_user_args(args)
        assert 'verbose' in result
        assert 'debug' in result
    
    def test_parse_mixed_arguments(self):
        """Test parsing mixed short and long options"""
        args = ['--epochs', '10', '-v', '--learning-rate', '0.001']
        result = parse_user_args(args)
        assert 'epochs' in result
        assert 'v' in result
        assert 'learning_rate' in result
    
    def test_parse_config_file_option(self):
        """Test parsing config file option and extracting its keys"""
        # Create a temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write('model_name: test\n')
            f.write('learning_rate: 0.001\n')
            config_path = f.name
        
        try:
            args = ['--config', config_path, '--batch_size', '32']
            result = parse_user_args(args)
            
            # Should include config option itself
            assert 'config' in result
            # Should include keys from the config file
            assert 'model_name' in result
            assert 'learning_rate' in result
            # Should include command line args
            assert 'batch_size' in result
        finally:
            os.unlink(config_path)
    
    def test_empty_args(self):
        """Test with empty arguments list"""
        args = []
        result = parse_user_args(args)
        assert len(result) == 0


class TestParseYamlConfig:
    """Test cases for parse_yaml_config function"""
    
    def test_parse_flat_yaml(self):
        """Test parsing flat YAML structure"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write('key1: value1\n')
            f.write('key2: value2\n')
            f.write('key3: 123\n')
            config_path = f.name
        
        try:
            result = parse_yaml_config(config_path)
            assert 'key1' in result
            assert 'key2' in result
            assert 'key3' in result
        finally:
            os.unlink(config_path)
    
    def test_parse_nested_yaml(self):
        """Test parsing nested YAML structure"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write('model:\n')
            f.write('  name: test_model\n')
            f.write('  size: 100\n')
            f.write('training:\n')
            f.write('  epochs: 10\n')
            f.write('  learning_rate: 0.001\n')
            config_path = f.name
        
        try:
            result = parse_yaml_config(config_path)
            # Should include top-level keys
            assert 'model' in result
            assert 'training' in result
            # Should include nested keys with prefix
            assert 'model_name' in result
            assert 'model_size' in result
            assert 'training_epochs' in result
            assert 'training_learning_rate' in result
        finally:
            os.unlink(config_path)
    
    def test_parse_deeply_nested_yaml(self):
        """Test parsing deeply nested YAML structure"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write('level1:\n')
            f.write('  level2:\n')
            f.write('    level3:\n')
            f.write('      key: value\n')
            config_path = f.name
        
        try:
            result = parse_yaml_config(config_path)
            assert 'level1' in result
            assert 'level1_level2' in result
            assert 'level1_level2_level3' in result
            assert 'level1_level2_level3_key' in result
        finally:
            os.unlink(config_path)
    
    def test_parse_empty_yaml(self):
        """Test parsing empty YAML file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write('')
            config_path = f.name
        
        try:
            result = parse_yaml_config(config_path)
            assert len(result) == 0
        finally:
            os.unlink(config_path)
    
    def test_parse_nonexistent_file(self):
        """Test parsing non-existent file returns empty set"""
        result = parse_yaml_config('/nonexistent/path/to/config.yaml')
        assert len(result) == 0
    
    def test_parse_invalid_yaml(self):
        """Test parsing invalid YAML returns empty set with warning"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write('invalid: yaml: structure: :\n')
            config_path = f.name
        
        try:
            # Should not raise exception, just print warning
            result = parse_yaml_config(config_path)
            # Result may be empty or partial depending on YAML parser
            assert isinstance(result, set)
        finally:
            os.unlink(config_path)


class TestColorizeUserArgs:
    """Test cases for colorize_user_args function"""
    
    def test_colorize_single_line(self):
        """Test colorizing a single line with user argument"""
        text = "'learning_rate': 0.001"
        user_args = {'learning_rate'}
        result = colorize_user_args(text, user_args)
        
        # Should contain ANSI color codes
        assert '\033[1;95m' in result  # PINK color
        assert '\033[0m' in result      # RESET
    
    def test_colorize_multiple_lines(self):
        """Test colorizing multiple lines with different arguments"""
        text = "'learning_rate': 0.001\n'batch_size': 32\n'epochs': 10"
        user_args = {'learning_rate', 'epochs'}
        result = colorize_user_args(text, user_args)
        
        lines = result.split('\n')
        # First and third lines should be colored
        assert '\033[1;95m' in lines[0]
        assert '\033[1;95m' in lines[2]
        # Second line should not be colored
        assert '\033[1;95m' not in lines[1]
    
    def test_colorize_no_matching_args(self):
        """Test text with no matching user arguments"""
        text = "'other_param': value"
        user_args = {'learning_rate'}
        result = colorize_user_args(text, user_args)
        
        # Should not contain color codes
        assert '\033[1;95m' not in result
    
    def test_colorize_unquoted_keys(self):
        """Test colorizing unquoted field names"""
        text = "learning_rate: 0.001"
        user_args = {'learning_rate'}
        result = colorize_user_args(text, user_args)
        
        # Should contain color codes
        assert '\033[1;95m' in result
    
    def test_colorize_empty_text(self):
        """Test colorizing empty text"""
        text = ""
        user_args = {'learning_rate'}
        result = colorize_user_args(text, user_args)
        assert result == ""
    
    def test_colorize_empty_user_args(self):
        """Test with empty user arguments set"""
        text = "'learning_rate': 0.001"
        user_args = set()
        result = colorize_user_args(text, user_args)
        
        # Should not contain color codes
        assert '\033[1;95m' not in result


class TestFormatArgsWithColors:
    """Test cases for format_args_with_colors function"""
    
    def test_format_simple_dict(self):
        """Test formatting simple dictionary with colors"""
        args_dict = {'learning_rate': 0.001, 'batch_size': 32, 'epochs': 10}
        user_args = {'learning_rate', 'epochs'}
        
        result = format_args_with_colors(args_dict, user_args)
        
        # Should be formatted as string
        assert isinstance(result, str)
        # Should contain the values
        assert '0.001' in result
        assert '32' in result
        assert '10' in result
        # Should contain color codes for user args
        assert '\033[1;95m' in result
    
    def test_format_nested_dict(self):
        """Test formatting nested dictionary"""
        args_dict = {
            'model': {'name': 'test', 'size': 100},
            'training': {'epochs': 10, 'lr': 0.001}
        }
        user_args = {'model', 'epochs'}
        
        result = format_args_with_colors(args_dict, user_args)
        
        assert isinstance(result, str)
        assert 'model' in result
        assert 'training' in result
    
    def test_format_empty_dict(self):
        """Test formatting empty dictionary"""
        args_dict = {}
        user_args = set()
        
        result = format_args_with_colors(args_dict, user_args)
        assert isinstance(result, str)
    
    def test_format_with_various_types(self):
        """Test formatting dictionary with various value types"""
        args_dict = {
            'string_val': 'test',
            'int_val': 42,
            'float_val': 3.14,
            'bool_val': True,
            'none_val': None,
            'list_val': [1, 2, 3],
            'dict_val': {'nested': 'value'}
        }
        user_args = {'string_val', 'bool_val'}
        
        result = format_args_with_colors(args_dict, user_args)
        
        assert isinstance(result, str)
        assert 'test' in result
        assert '42' in result
        assert 'True' in result
        assert 'None' in result


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

