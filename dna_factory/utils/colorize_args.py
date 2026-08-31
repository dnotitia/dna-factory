import pprint
import yaml
import os


def parse_user_args(args):
    """Parse command line arguments to identify user-specified options including config files"""
    user_specified_args = set()
    i = 0
    while i < len(args):
        arg = args[i]
        if arg.startswith('--'):
            # Remove '--' prefix and convert to field name format
            field_name = arg[2:].replace('-', '_')
            user_specified_args.add(field_name)
            
            # Special handling for --config option
            if field_name == 'config' and i + 1 < len(args) and not args[i + 1].startswith('--'):
                config_file = args[i + 1]
                # Parse the YAML config file and add its keys to user_specified_args
                yaml_args = parse_yaml_config(config_file)
                user_specified_args.update(yaml_args)
            
            # Skip the next argument if it's a value (not starting with --)
            if i + 1 < len(args) and not args[i + 1].startswith('--'):
                i += 1
        elif arg.startswith('-') and len(arg) > 1:
            # Handle short options like -n, -v, etc.
            field_name = arg[1:]
            user_specified_args.add(field_name)
            if i + 1 < len(args) and not args[i + 1].startswith('-'):
                i += 1
        i += 1
    return user_specified_args


def parse_yaml_config(config_file):
    """Parse YAML configuration file and return set of keys"""
    yaml_args = set()
    try:
        if os.path.exists(config_file):
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
                if config_data:
                    # Recursively collect all keys from nested dictionaries
                    def collect_keys(data, prefix=''):
                        if isinstance(data, dict):
                            for key, value in data.items():
                                full_key = f"{prefix}_{key}" if prefix else key
                                yaml_args.add(full_key)
                                if isinstance(value, dict):
                                    collect_keys(value, full_key)
                    
                    collect_keys(config_data)
    except Exception as e:
        # If there's an error reading the config file, just continue without adding yaml args
        print(f"Warning: Could not parse config file {config_file}: {e}")
    
    return yaml_args


def colorize_user_args(text, user_specified_args):
    """Colorize entire lines containing user-specified arguments"""
    PINK = '\033[1;95m'  # Bold bright magenta/pink color
    RESET = '\033[0m'

    lines = text.split('\n')
    colored_lines = []

    for line in lines:
        # Check if this line contains any user-specified arguments
        line_contains_user_arg = False
        for arg in user_specified_args:
            # Pattern for quoted field names or unquoted field names
            if f"'{arg}':" in line or (f"{arg}:" in line and f"'{arg}':" not in line):
                line_contains_user_arg = True
                break

        # If line contains user argument, color the entire line pink
        if line_contains_user_arg:
            line = f"{PINK}{line}{RESET}"

        colored_lines.append(line)

    return '\n'.join(colored_lines)


def format_args_with_colors(args_dict, user_specified_args):
    """Format arguments dictionary with colors for user-specified values"""
    formatted = pprint.pformat(args_dict, indent=2, width=80)
    return colorize_user_args(formatted, user_specified_args)
