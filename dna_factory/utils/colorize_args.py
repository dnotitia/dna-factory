import os
import pprint

import yaml


def parse_user_args(args):
    """Parse command line arguments to identify user-specified options including config files"""
    user_specified_args = set()
    i = 0
    while i < len(args):
        arg = args[i]
        if arg.startswith("--"):
            # `--name=value` carries its value in the same token, `--name value` in the next one.
            name, inline, inline_value = arg[2:].partition("=")
            field_name = name.replace("-", "_")
            user_specified_args.add(field_name)

            if inline:
                value = inline_value
            elif i + 1 < len(args) and not args[i + 1].startswith("--"):
                value = args[i + 1]
                i += 1
            else:
                value = None

            # Keys of a user-supplied config file count as user-specified too
            if field_name == "config" and value:
                user_specified_args.update(parse_yaml_config(value))
        elif arg.startswith("-") and len(arg) > 1:
            # Handle short options like -n, -v, etc.
            field_name = arg[1:]
            user_specified_args.add(field_name)
            if i + 1 < len(args) and not args[i + 1].startswith("-"):
                i += 1
        i += 1
    return user_specified_args


def parse_yaml_config(config_file):
    """Parse YAML configuration file and return set of keys"""
    yaml_args = set()
    try:
        if os.path.exists(config_file):
            with open(config_file, encoding="utf-8") as f:
                config_data = yaml.safe_load(f)
                if config_data:
                    # Recursively collect all keys from nested dictionaries
                    def collect_keys(data, prefix=""):
                        if isinstance(data, dict):
                            for key, value in data.items():
                                full_key = f"{prefix}_{key}" if prefix else key
                                yaml_args.add(full_key)
                                if isinstance(value, dict):
                                    collect_keys(value, full_key)

                    collect_keys(config_data)
    except Exception as e:  # noqa: BLE001 - cosmetic only; a bad config must not abort the run
        # If there's an error reading the config file, just continue without adding yaml args
        print(f"Warning: Could not parse config file {config_file}: {e}")

    return yaml_args


def _same_value(a, b):
    """Whether two parsed argument values are equal; values that cannot be compared are not."""
    try:
        return bool(a == b)
    except Exception:  # noqa: BLE001 - e.g. array-like values with no single truth value
        return False


def find_redundant_args(user_specified_args, user_values, default_values):
    """
    Names the user specified with the value they already have by default.

    `user_values` and `default_values` map argument names to their parsed values, for the
    user's run and for a run with no user config or CLI arguments. Comparing parsed values
    means `1e-5` and `1.0e-5`, or `true` and `True`, count as the same value; lists and
    dicts compare element by element. A name missing from either mapping is never redundant.
    """
    return {
        name
        for name in user_specified_args
        if name in user_values
        and name in default_values
        and _same_value(user_values[name], default_values[name])
    }


def _line_mentions(line, args):
    """Whether the line shows any of the arguments (quoted or unquoted field name)."""
    return any(
        f"'{arg}':" in line or (f"{arg}:" in line and f"'{arg}':" not in line)
        for arg in args
    )


def colorize_user_args(text, user_specified_args, redundant_args=()):
    """
    Colorize entire lines containing user-specified arguments.

    Lines of arguments that differ from the default are pink; lines of arguments the user
    specified with their default value (`redundant_args`, effectively ignored) are gray.
    """
    PINK = "\033[1;95m"  # Bold bright magenta/pink color
    GRAY = "\033[90m"  # Bright black (gray)
    RESET = "\033[0m"

    lines = text.split("\n")
    colored_lines = []

    for line in lines:
        if _line_mentions(line, user_specified_args):
            line = f"{PINK}{line}{RESET}"
        elif _line_mentions(line, redundant_args):
            line = f"{GRAY}{line}{RESET}"

        colored_lines.append(line)

    return "\n".join(colored_lines)


def format_args_with_colors(args_dict, user_specified_args, redundant_args=()):
    """Format arguments dictionary with colors for user-specified values"""
    formatted = pprint.pformat(args_dict, indent=2, width=80)
    return colorize_user_args(formatted, user_specified_args, redundant_args)
