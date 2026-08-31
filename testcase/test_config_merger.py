"""
Test cases for dna_factory/utils/config_merger.py

This module tests the configuration file merging functionality.
"""

import os
import sys
import tempfile
import yaml
import pytest
from pathlib import Path

# Add parent directory to path to import dna_factory
sys.path.insert(0, str(Path(__file__).parent.parent))

from dna_factory.utils.config_merger import merge_config_files


class TestMergeConfigFiles:
    """Test cases for merge_config_files function"""
    
    def test_merge_with_only_default_config(self):
        """Test merging with only default config (no user config)"""
        # Create default config
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            default_config = {
                'model_name': 'default_model',
                'learning_rate': 0.001,
                'batch_size': 32
            }
            yaml.safe_dump(default_config, f)
            default_path = f.name
        
        try:
            # Merge without user config
            merged_path = merge_config_files(default_path, user_config_path=None)
            
            # Read merged config
            with open(merged_path, 'r') as f:
                merged = yaml.safe_load(f)
            
            # Should be identical to default config
            assert merged == default_config
            
            # Cleanup
            os.unlink(merged_path)
        finally:
            os.unlink(default_path)
    
    def test_merge_flat_configs(self):
        """Test merging two flat configuration files"""
        # Create default config
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            default_config = {
                'model_name': 'default_model',
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 10
            }
            yaml.safe_dump(default_config, f)
            default_path = f.name
        
        # Create user config (overrides some values)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            user_config = {
                'model_name': 'user_model',
                'batch_size': 64
            }
            yaml.safe_dump(user_config, f)
            user_path = f.name
        
        try:
            # Merge configs
            merged_path = merge_config_files(default_path, user_path)
            
            # Read merged config
            with open(merged_path, 'r') as f:
                merged = yaml.safe_load(f)
            
            # User values should override defaults
            assert merged['model_name'] == 'user_model'
            assert merged['batch_size'] == 64
            # Default values should be preserved
            assert merged['learning_rate'] == 0.001
            assert merged['epochs'] == 10
            
            # Cleanup
            os.unlink(merged_path)
        finally:
            os.unlink(default_path)
            os.unlink(user_path)
    
    def test_merge_nested_configs(self):
        """Test merging nested configuration structures"""
        # Create default config
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            default_config = {
                'model': {
                    'name': 'default_model',
                    'size': 100,
                    'layers': 12
                },
                'training': {
                    'epochs': 10,
                    'learning_rate': 0.001
                }
            }
            yaml.safe_dump(default_config, f)
            default_path = f.name
        
        # Create user config
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            user_config = {
                'model': {
                    'name': 'user_model',
                    'size': 200
                },
                'training': {
                    'epochs': 20
                }
            }
            yaml.safe_dump(user_config, f)
            user_path = f.name
        
        try:
            # Merge configs
            merged_path = merge_config_files(default_path, user_path)
            
            # Read merged config
            with open(merged_path, 'r') as f:
                merged = yaml.safe_load(f)
            
            # Check nested overrides
            assert merged['model']['name'] == 'user_model'
            assert merged['model']['size'] == 200
            # Check preserved defaults
            assert merged['model']['layers'] == 12
            assert merged['training']['learning_rate'] == 0.001
            # Check user overrides
            assert merged['training']['epochs'] == 20
            
            # Cleanup
            os.unlink(merged_path)
        finally:
            os.unlink(default_path)
            os.unlink(user_path)
    
    def test_merge_deeply_nested_configs(self):
        """Test merging deeply nested configuration structures"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            default_config = {
                'level1': {
                    'level2': {
                        'level3': {
                            'key1': 'default1',
                            'key2': 'default2'
                        }
                    }
                }
            }
            yaml.safe_dump(default_config, f)
            default_path = f.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            user_config = {
                'level1': {
                    'level2': {
                        'level3': {
                            'key1': 'user1'
                        }
                    }
                }
            }
            yaml.safe_dump(user_config, f)
            user_path = f.name
        
        try:
            merged_path = merge_config_files(default_path, user_path)
            
            with open(merged_path, 'r') as f:
                merged = yaml.safe_load(f)
            
            # User override should work at deep level
            assert merged['level1']['level2']['level3']['key1'] == 'user1'
            # Default should be preserved
            assert merged['level1']['level2']['level3']['key2'] == 'default2'
            
            os.unlink(merged_path)
        finally:
            os.unlink(default_path)
            os.unlink(user_path)
    
    def test_merge_with_list_values(self):
        """Test merging configs with list values"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            default_config = {
                'datasets': [
                    {'id': 'ds1', 'path': '/path1', 'weight': 1.0},
                    {'id': 'ds2', 'path': '/path2', 'weight': 0.5}
                ],
                'simple_list': [1, 2, 3]
            }
            yaml.safe_dump(default_config, f)
            default_path = f.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            user_config = {
                'datasets': [
                    {'id': 'ds1', 'weight': 2.0},  # Override weight for ds1
                    {'id': 'ds3', 'path': '/path3', 'weight': 1.5}  # Add new dataset
                ],
                'simple_list': [4, 5, 6]  # Override entire list
            }
            yaml.safe_dump(user_config, f)
            user_path = f.name
        
        try:
            merged_path = merge_config_files(default_path, user_path)
            
            with open(merged_path, 'r') as f:
                merged = yaml.safe_load(f)
            
            # Check datasets list merge by id
            datasets = merged['datasets']
            ds1 = next(d for d in datasets if d.get('id') == 'ds1')
            assert ds1['weight'] == 2.0  # Override
            assert ds1['path'] == '/path1'  # Preserved
            
            # Check new dataset added
            ds3 = next((d for d in datasets if d.get('id') == 'ds3'), None)
            assert ds3 is not None
            assert ds3['weight'] == 1.5
            
            # Simple list should be replaced
            assert merged['simple_list'] == [4, 5, 6]
            
            os.unlink(merged_path)
        finally:
            os.unlink(default_path)
            os.unlink(user_path)
    
    def test_merge_with_new_keys_in_user_config(self):
        """Test that new keys from user config are added"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            default_config = {
                'existing_key': 'value1'
            }
            yaml.safe_dump(default_config, f)
            default_path = f.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            user_config = {
                'new_key': 'value2',
                'another_key': 'value3'
            }
            yaml.safe_dump(user_config, f)
            user_path = f.name
        
        try:
            merged_path = merge_config_files(default_path, user_path)
            
            with open(merged_path, 'r') as f:
                merged = yaml.safe_load(f)
            
            # All keys should be present
            assert 'existing_key' in merged
            assert 'new_key' in merged
            assert 'another_key' in merged
            assert merged['existing_key'] == 'value1'
            assert merged['new_key'] == 'value2'
            assert merged['another_key'] == 'value3'
            
            os.unlink(merged_path)
        finally:
            os.unlink(default_path)
            os.unlink(user_path)
    
    def test_merge_with_nonexistent_user_config(self):
        """Test merging when user config path doesn't exist"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            default_config = {'key': 'value'}
            yaml.safe_dump(default_config, f)
            default_path = f.name
        
        try:
            # Use non-existent path
            merged_path = merge_config_files(default_path, '/nonexistent/path.yaml')
            
            with open(merged_path, 'r') as f:
                merged = yaml.safe_load(f)
            
            # Should just return default config
            assert merged == default_config
            
            os.unlink(merged_path)
        finally:
            os.unlink(default_path)
    
    def test_merge_with_empty_configs(self):
        """Test merging empty configuration files"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write('')
            default_path = f.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write('')
            user_path = f.name
        
        try:
            merged_path = merge_config_files(default_path, user_path)
            
            with open(merged_path, 'r') as f:
                merged = yaml.safe_load(f)
            
            # Should be empty or None
            assert merged is None or merged == {}
            
            os.unlink(merged_path)
        finally:
            os.unlink(default_path)
            os.unlink(user_path)
    
    def test_merge_creates_valid_yaml(self):
        """Test that merged config is valid YAML"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            default_config = {'key1': 'value1', 'key2': {'nested': 'value2'}}
            yaml.safe_dump(default_config, f)
            default_path = f.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            user_config = {'key3': 'value3'}
            yaml.safe_dump(user_config, f)
            user_path = f.name
        
        try:
            merged_path = merge_config_files(default_path, user_path)
            
            # Should be able to load without error
            with open(merged_path, 'r') as f:
                merged = yaml.safe_load(f)
            
            assert isinstance(merged, dict)
            
            os.unlink(merged_path)
        finally:
            os.unlink(default_path)
            os.unlink(user_path)
    
    def test_merge_preserves_unicode(self):
        """Test that Unicode characters are preserved in merge"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8') as f:
            default_config = {'message': '안녕하세요', 'count': 42}
            yaml.safe_dump(default_config, f, allow_unicode=True)
            default_path = f.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8') as f:
            user_config = {'greeting': '반갑습니다', 'message': '환영합니다'}
            yaml.safe_dump(user_config, f, allow_unicode=True)
            user_path = f.name
        
        try:
            merged_path = merge_config_files(default_path, user_path)
            
            with open(merged_path, 'r', encoding='utf-8') as f:
                merged = yaml.safe_load(f)
            
            assert merged['message'] == '환영합니다'
            assert merged['greeting'] == '반갑습니다'
            assert merged['count'] == 42
            
            os.unlink(merged_path)
        finally:
            os.unlink(default_path)
            os.unlink(user_path)
    
    def test_merge_returns_temp_file_path(self):
        """Test that merge returns a valid temporary file path"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.safe_dump({'key': 'value'}, f)
            default_path = f.name
        
        try:
            merged_path = merge_config_files(default_path)
            
            # Should be a string
            assert isinstance(merged_path, str)
            # File should exist
            assert os.path.exists(merged_path)
            # Should be a yaml file
            assert merged_path.endswith('.yaml')
            # Should be able to read it
            with open(merged_path, 'r') as f:
                content = yaml.safe_load(f)
                assert content is not None
            
            os.unlink(merged_path)
        finally:
            os.unlink(default_path)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

