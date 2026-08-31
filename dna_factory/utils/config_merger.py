import os
import tempfile
import yaml
from pathlib import Path


def merge_config_files(default_config_path, user_config_path=None):
    """
    Merge default config with user config and create a temporary merged config file.
    
    Args:
        default_config_path (str): Path to the default config file
        user_config_path (str): Path to the user config file (optional)
    
    Returns:
        str: Path to the temporary merged config file
    """
    # Load default config
    with open(default_config_path, 'r', encoding='utf-8') as f:
        default_config = yaml.safe_load(f) or {}

    # Start with default config 
    merged_config = default_config.copy()

    # If user config exists, merge it (user config overrides defaults)
    if user_config_path and Path(user_config_path).exists():
        with open(user_config_path, 'r', encoding='utf-8') as f:
            user_config = yaml.safe_load(f) or {}

        # Deep merge user config into default config
        def deep_merge(base, override):
            """
            Recursively merge override into base and return the merged structure.
            """
            # Merge dictionaries
            if isinstance(base, dict) and isinstance(override, dict):
                for k, override_value in override.items():
                    if k in base:
                        base[k] = deep_merge(base[k], override_value)
                    else:
                        base[k] = override_value
                return base

            # Merge lists
            if isinstance(base, list) and isinstance(override, list):
                # Attempt keyed merge for lists of dicts
                def find_key_field(sample_list):
                    if not sample_list:
                        return None
                    candidate_keys = ("id", "name", "key")
                    # Key is acceptable if present in all dict elements
                    for candidate in candidate_keys:
                        if all(isinstance(elem, dict) and candidate in elem for elem in sample_list):
                            return candidate
                    return None

                key_field = find_key_field(base) or find_key_field(override)

                if key_field is not None:
                    # Build map from base for quick lookup
                    base_map = {}
                    base_order = []
                    for elem in base:
                        if isinstance(elem, dict) and key_field in elem:
                            base_map[elem[key_field]] = elem
                            base_order.append(elem[key_field])
                        else:
                            base_order.append(None)

                    # Merge/append override elements
                    for o_elem in override:
                        if isinstance(o_elem, dict) and key_field in o_elem:
                            elem_id = o_elem[key_field]
                            if elem_id in base_map:
                                base_map[elem_id] = deep_merge(base_map[elem_id], o_elem)
                            else:
                                base_map[elem_id] = o_elem
                                base_order.append(elem_id)
                        else:
                            # Append non-dict or missing-key items as-is
                            base_order.append(None)
                            base.append(o_elem)

                    # Reconstruct list preserving the original order where possible,
                    # followed by any new keyed items not yet placed.
                    merged_list = []
                    placed_ids = set()
                    for order_key in base_order:
                        if order_key is None:
                            # This corresponds to a non-keyed element already in base
                            # Elements without keys were already appended above; keep in place
                            continue
                        if order_key in base_map and order_key not in placed_ids:
                            merged_list.append(base_map[order_key])
                            placed_ids.add(order_key)

                    # Add any remaining keyed items from base_map that were not in base_order
                    for elem_id, elem in base_map.items():
                        if elem_id not in placed_ids:
                            merged_list.append(elem)

                    # Add trailing non-dict items appended earlier
                    for elem in base:
                        if not isinstance(elem, dict) or key_field not in elem:
                            merged_list.append(elem)

                    return merged_list

                # Fallback: replace the entire list
                return override

            # For all other types, return the override
            return override

        merged_config = deep_merge(merged_config, user_config)

    # Create temporary file with merged config
    temp_fd, temp_path = tempfile.mkstemp(suffix='.yaml', prefix='merged_config_')
    try:
        with os.fdopen(temp_fd, 'w', encoding='utf-8') as temp_file:
            # Preserve key insertion order for readability and stable diffs
            yaml.safe_dump(
                merged_config,
                temp_file,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )
    except Exception as e:
        # Clean up temp file if writing fails
        os.unlink(temp_path)
        raise e

    return temp_path
