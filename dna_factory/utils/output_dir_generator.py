def generate_auto_output_dir(model_name_or_path, user_specified_args, script_args, training_args, model_args,
                             dataset_mixture_args, dnotitia_args, training_type):
    """
    Auto-generate output directory name based on model name and user-specified arguments.

    Args:
        model_name_or_path (str): Model name or path
        user_specified_args (set): Set of user-specified argument names
        script_args: Script arguments object
        training_args: Training arguments object
        model_args: Model arguments object
        dataset_mixture_args: Dataset mixture arguments object
        dnotitia_args: Dnotitia arguments object

    Returns:
        str: Auto-generated output directory name
    """
    # Mapping of parameter names to their abbreviations
    # The order of parameters in this dictionary determines the order in the output directory name
    PARAM_ABBREVIATIONS = {
        'dataset_name': 'dataset',
        'dataset_config': 'dataset_config',
        'datasets': 'datasets',

        'run_name': 'run',
        'attn_implementation': 'attn',

        'teacher_model_name_or_path': 'teacher',

        'max_length': 'max',
        'packing': 'pack',
        'assistant_only_loss': 'ao_loss',
        'loss_type': 'loss',

        'reward_funcs': 'rf',
        'reward_model_name_or_path': 'rm',
        'num_generations': 'ng',
        'max_completion_length': 'mcl',
        'beta': 'beta',
        'temperature': 'temp',
        'epsilon': 'eps',
        'scale_rewards': 'scale',
        'use_vllm': 'vllm',
        'vllm_mode': 'vmode',

        'num_train_epochs': 'ep',

        'per_device_train_batch_size': 'bs',
        'learning_rate': 'lr',

        'gradient_checkpointing': 'gc',
        'use_liger_kernel': 'use_liger_kernel',

        'debug_first_n_batches': 'dfnb',
    }

    # Extract model name from model_name_or_path
    model_path = model_name_or_path.rstrip('/')
    if '/' in model_path:
        # Extract model name after the last slash (e.g., "Qwen/Qwen3-0.6B" -> "Qwen3-0.6B")
        model_name = model_path.split('/')[-1]
    else:
        # Use the entire path if no slash (local model)
        model_name = model_path

    # Get all argument dictionaries
    all_args = [
        ('script', vars(script_args)),
        ('training', vars(training_args)),
        ('model', vars(model_args)),
        ('dataset', vars(dataset_mixture_args)),
        ('dnotitia', vars(dnotitia_args)),
    ]

    # Create a unified dictionary of all user-specified arguments
    user_args_dict = {}
    for prefix, args_dict in all_args:
        for arg_name, arg_value in args_dict.items():
            if arg_name in user_specified_args and arg_name not in ['output_dir', 'model_name_or_path']:
                user_args_dict[arg_name] = arg_value

    # Collect user-specified argument names and values in the order defined in PARAM_ABBREVIATIONS
    auto_dir_parts = []
    
    # First, add parameters in the order they appear in PARAM_ABBREVIATIONS
    for arg_name in PARAM_ABBREVIATIONS.keys():
        if arg_name in user_args_dict:
            arg_value = user_args_dict[arg_name]
            
            # Special handling for datasets field
            if arg_name == 'datasets' and isinstance(arg_value, list):
                # Count the number of datasets
                dataset_count = len(arg_value)
                if dataset_count > 0:
                    # Extract the first dataset's organization/owner name (e.g., "dnotitia" from "dnotitia/dataset-name")
                    first_dataset_org = None
                    for dataset_config in arg_value:
                        dataset_path = None
                        if hasattr(dataset_config, 'path') and dataset_config.path:
                            dataset_path = dataset_config.path
                        elif hasattr(dataset_config, 'id') and dataset_config.id:
                            dataset_path = dataset_config.id
                        
                        if dataset_path and '/' in dataset_path:
                            first_dataset_org = dataset_path.split('/')[0]
                            break
                    
                    # Format: datasets-{org}-{count}
                    if first_dataset_org:
                        auto_dir_parts.append(f"{arg_name}-{first_dataset_org}-{dataset_count}ea")
                    else:
                        auto_dir_parts.append(f"{arg_name}-{dataset_count}ea")
                continue
            
            # Convert value to string and handle special cases
            if isinstance(arg_value, (list, tuple)):
                value_str = '-'.join(str(v) for v in arg_value)
            elif isinstance(arg_value, bool):
                value_str = str(arg_value).lower()
            elif arg_value is None:
                value_str = 'none'
            else:
                value_str = str(arg_value)

            # Clean the value string (remove special characters, spaces, but preserve underscores)
            value_str = value_str.replace('/', '-').replace(' ', '-')
            
            # Use abbreviation if available, otherwise use full parameter name
            param_name = PARAM_ABBREVIATIONS.get(arg_name, arg_name)
            auto_dir_parts.append(f"{param_name}-{value_str}")
    
    # Then, add any remaining user-specified parameters that are not in PARAM_ABBREVIATIONS
    for arg_name, arg_value in user_args_dict.items():
        if arg_name not in PARAM_ABBREVIATIONS:
            # Convert value to string and handle special cases
            if isinstance(arg_value, (list, tuple)):
                value_str = '-'.join(str(v) for v in arg_value)
            elif isinstance(arg_value, bool):
                value_str = str(arg_value).lower()
            elif arg_value is None:
                value_str = 'none'
            else:
                value_str = str(arg_value)

            # Clean the value string (remove special characters, spaces, but preserve underscores)
            value_str = value_str.replace('/', '-').replace(' ', '-')
            
            # Use full parameter name
            auto_dir_parts.append(f"{arg_name}-{value_str}")

    # Generate the auto directory name starting with model name
    # Use '.' as delimiter between parameters
    if auto_dir_parts:
        auto_output_dir = f"{model_name}-{training_type}.{'.'.join(auto_dir_parts)}"
    else:
        # Fallback if no user args specified
        auto_output_dir = f"{model_name}-{training_type}"

    # Limit the output directory name to 200 characters
    return auto_output_dir[:200]
