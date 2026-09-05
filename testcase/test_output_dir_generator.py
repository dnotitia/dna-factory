"""
Test cases for dna_factory/utils/output_dir_generator.py

This module tests the automatic output directory name generation functionality.
"""

import sys
import pytest
from types import SimpleNamespace
from pathlib import Path

# Add parent directory to path to import dna_factory
sys.path.insert(0, str(Path(__file__).parent.parent))

from dna_factory.utils.output_dir_generator import generate_auto_output_dir


class TestGenerateAutoOutputDir:
    """Test cases for generate_auto_output_dir function"""

    def test_basic_model_name_only(self):
        """Test with only model name, no user args"""
        model_name = "Qwen/Qwen3-0.6B"
        user_args = set()

        script_args = SimpleNamespace()
        training_args = SimpleNamespace()
        model_args = SimpleNamespace()
        dataset_mixture_args = SimpleNamespace()
        dnotitia_args = SimpleNamespace()

        result = generate_auto_output_dir(
            model_name, user_args, script_args, training_args,
            model_args, dataset_mixture_args, dnotitia_args, "SFT"
        )

        # Should just be model name + SFT
        assert result == "Qwen3-0.6B-SFT"

    def test_model_name_extraction_with_slash(self):
        """Test extracting model name from path with slash"""
        model_name = "HuggingFace/SmolLM-135M"
        user_args = set()

        script_args = SimpleNamespace()
        training_args = SimpleNamespace()
        model_args = SimpleNamespace()
        dataset_mixture_args = SimpleNamespace()
        dnotitia_args = SimpleNamespace()

        result = generate_auto_output_dir(
            model_name, user_args, script_args, training_args,
            model_args, dataset_mixture_args, dnotitia_args, "SFT"
        )

        # Should extract only the part after the slash
        assert result == "SmolLM-135M-SFT"

    def test_with_multiple_user_args(self):
        """Test with multiple user-specified arguments"""
        model_name = "test/model"
        user_args = {'learning_rate', 'per_device_train_batch_size', 'num_train_epochs'}

        script_args = SimpleNamespace()
        training_args = SimpleNamespace(learning_rate=0.001, per_device_train_batch_size=32, num_train_epochs=10)
        model_args = SimpleNamespace()
        dataset_mixture_args = SimpleNamespace()
        dnotitia_args = SimpleNamespace()

        result = generate_auto_output_dir(
            model_name, user_args, script_args, training_args,
            model_args, dataset_mixture_args, dnotitia_args, "SFT"
        )

        assert "model-SFT" in result
        assert "lr-0.001" in result
        assert "bs-32" in result
        assert "ep-10" in result

    def test_excludes_output_dir_arg(self):
        """Test that output_dir is excluded from auto-generated name"""
        model_name = "test/model"
        user_args = {'output_dir', 'learning_rate'}

        script_args = SimpleNamespace()
        training_args = SimpleNamespace(output_dir='/some/path', learning_rate=0.001)
        model_args = SimpleNamespace()
        dataset_mixture_args = SimpleNamespace()
        dnotitia_args = SimpleNamespace()

        result = generate_auto_output_dir(
            model_name, user_args, script_args, training_args,
            model_args, dataset_mixture_args, dnotitia_args, "SFT"
        )

        # Should not contain output_dir
        assert 'output_dir' not in result
        assert '/some/path' not in result
        # Should contain learning_rate
        assert 'lr-0.001' in result

    def test_excludes_model_name_or_path_arg(self):
        """Test that model_name_or_path is excluded from auto-generated name"""
        model_name = "test/model"
        user_args = {'model_name_or_path', 'num_train_epochs'}

        script_args = SimpleNamespace()
        training_args = SimpleNamespace(num_train_epochs=5)
        model_args = SimpleNamespace(model_name_or_path='test/model')
        dataset_mixture_args = SimpleNamespace()
        dnotitia_args = SimpleNamespace()

        result = generate_auto_output_dir(
            model_name, user_args, script_args, training_args,
            model_args, dataset_mixture_args, dnotitia_args, "SFT"
        )

        # Should not contain model_name_or_path
        assert 'model-name-or-path' not in result
        # Should contain epochs
        assert 'ep-5' in result

    def test_with_boolean_values(self):
        """Test with boolean argument values"""
        model_name = "test/model"
        user_args = {'use_lora', 'gradient_checkpointing'}

        script_args = SimpleNamespace()
        training_args = SimpleNamespace(gradient_checkpointing=True)
        model_args = SimpleNamespace(use_lora=False)
        dataset_mixture_args = SimpleNamespace()
        dnotitia_args = SimpleNamespace()

        result = generate_auto_output_dir(
            model_name, user_args, script_args, training_args,
            model_args, dataset_mixture_args, dnotitia_args, "SFT"
        )

        # Boolean should be lowercase
        assert 'use_lora-false' in result
        assert 'gc-true' in result

    def test_with_list_values(self):
        """Test with list argument values"""
        model_name = "test/model"
        user_args = {'layers'}

        script_args = SimpleNamespace(layers=[1, 2, 3])
        training_args = SimpleNamespace()
        model_args = SimpleNamespace()
        dataset_mixture_args = SimpleNamespace()
        dnotitia_args = SimpleNamespace()

        result = generate_auto_output_dir(
            model_name, user_args, script_args, training_args,
            model_args, dataset_mixture_args, dnotitia_args, "SFT"
        )

        # List should be joined with dashes
        assert 'layers-1-2-3' in result

    def test_with_tuple_values(self):
        """Test with tuple argument values"""
        model_name = "test/model"
        user_args = {'dimensions'}

        script_args = SimpleNamespace(dimensions=(256, 512))
        training_args = SimpleNamespace()
        model_args = SimpleNamespace()
        dataset_mixture_args = SimpleNamespace()
        dnotitia_args = SimpleNamespace()

        result = generate_auto_output_dir(
            model_name, user_args, script_args, training_args,
            model_args, dataset_mixture_args, dnotitia_args, "SFT"
        )

        assert 'dimensions-256-512' in result

    def test_complex_output_dir_name(self):
        """Test generating a complex output directory name with many parameters"""
        model_name = "HuggingFace/Qwen3-0.6B"
        user_args = {
            'learning_rate', 'per_device_train_batch_size', 'num_train_epochs',
            'use_lora', 'lora_rank', 'gradient_checkpointing',
            'max_length', 'packing', 'assistant_only_loss',
            'debug_first_n_batches', 'run_name', 'use_liger_kernel',
        }

        script_args = SimpleNamespace()
        training_args = SimpleNamespace(
            learning_rate=0.0001,
            per_device_train_batch_size=16,
            num_train_epochs=3,
            gradient_checkpointing=True,
            max_length=16000,
            packing=True,
            assistant_only_loss=True,
            run_name="test",
            use_liger_kernel=True,
        )
        model_args = SimpleNamespace(
            use_lora=True,
            lora_rank=8
        )
        dataset_mixture_args = SimpleNamespace()
        dnotitia_args = SimpleNamespace(
            debug_first_n_batches=10,
        )

        result = generate_auto_output_dir(
            model_name, user_args, script_args, training_args,
            model_args, dataset_mixture_args, dnotitia_args, "SFT"
        )

        assert ('Qwen3-0.6B-SFT.run-test.max-16000.pack-true.ao_loss-true.ep-3.bs-16.lr-0.0001.'
                'gc-true.use_liger_kernel-true.dfnb-10.use_lora-true.lora_rank-8') == result

    def test_datasets_with_multiple_datasets(self):
        """Test with multiple datasets to check that count is shown instead of full names"""
        model_name = "dnotitia/Qwen3-4B"
        user_args = {'datasets'}

        # Create mock dataset objects
        dataset1 = SimpleNamespace(path="dnotitia/dpo_claude3.5-sonnet_15k_v3")
        dataset2 = SimpleNamespace(path="dnotitia/dpo-kmmlu-180k-v2")
        dataset3 = SimpleNamespace(path="dnotitia/dpo-law-counsel-v1")
        dataset4 = SimpleNamespace(path="dnotitia/dpo_ethical_safety_claude35_10k_v1")

        script_args = SimpleNamespace()
        training_args = SimpleNamespace()
        model_args = SimpleNamespace()
        dataset_mixture_args = SimpleNamespace(datasets=[dataset1, dataset2, dataset3, dataset4])
        dnotitia_args = SimpleNamespace()

        result = generate_auto_output_dir(
            model_name, user_args, script_args, training_args,
            model_args, dataset_mixture_args, dnotitia_args, "DPO"
        )

        # Should show organization name and count instead of full dataset names
        assert result == "Qwen3-4B-DPO.datasets-dnotitia-4ea"
        # Should NOT contain full dataset names
        assert "dpo_claude3.5-sonnet_15k_v3" not in result
        assert "dpo-kmmlu-180k-v2" not in result

    def test_distillation_teacher_model(self):
        """Test the DISTILL training type and the teacher model abbreviation"""
        model_name = "dnotitia/Qwen3-0.6B"
        user_args = {'teacher_model_name_or_path', 'beta', 'max_completion_length'}

        script_args = SimpleNamespace()
        training_args = SimpleNamespace(
            teacher_model_name_or_path="dnotitia/Qwen3-1.7B",
            beta=1.0,
            max_completion_length=512,
        )
        model_args = SimpleNamespace()
        dataset_mixture_args = SimpleNamespace()
        dnotitia_args = SimpleNamespace()

        result = generate_auto_output_dir(
            model_name, user_args, script_args, training_args,
            model_args, dataset_mixture_args, dnotitia_args, "DISTILL"
        )

        # The '/' in the teacher id is normalized to '-', as for every other value
        assert result == "Qwen3-0.6B-DISTILL.teacher-dnotitia-Qwen3-1.7B.mcl-512.beta-1.0"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
