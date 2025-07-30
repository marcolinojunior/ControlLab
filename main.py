import argparse
import inspect
from typing import List, get_origin

import controllab.analysis
import controllab.core
import controllab.design
import controllab.modeling
import controllab.numerical

def build_cli():
    """
    Dynamically builds a CLI for the controllab library.
    """
    parser = argparse.ArgumentParser(description="A dynamically generated CLI for the controllab library.")
    subparsers = parser.add_subparsers(dest="command", help="Available commands", required=True)

    modules_to_inspect = [
        controllab.analysis,
        controllab.core,
        controllab.design,
        controllab.modeling,
        controllab.numerical,
    ]

    added_commands = set()

    for module in modules_to_inspect:
        # Discover and add functions
        for func_name, func_obj in inspect.getmembers(module, inspect.isfunction):
            if not func_name.startswith('_') and func_obj.__module__.startswith(module.__name__):
                if func_name not in added_commands:
                    func_parser = subparsers.add_parser(func_name, help=f"Execute the '{func_name}' function from {module.__name__}.")
                    sig = inspect.signature(func_obj)
                    for param in sig.parameters.values():
                        if param.name == 'self':
                            continue
                        arg_name = f'--{param.name}'
                        arg_type = param.annotation if param.annotation != inspect.Parameter.empty else str

                        # Use str for complex types for now
                        if arg_type not in [str, int, float, bool]:
                            arg_type = str

                        is_list = get_origin(param.annotation) in (list, List)
                        nargs = '+' if is_list else None

                        if param.default == inspect.Parameter.empty:
                            func_parser.add_argument(arg_name, type=arg_type, required=True, nargs=nargs)
                        else:
                            if isinstance(param.default, bool) and not param.default:
                                func_parser.add_argument(arg_name, action='store_true')
                            else:
                                func_parser.add_argument(arg_name, type=arg_type, default=param.default, nargs=nargs)
                    func_parser.set_defaults(func=func_obj, is_method=False)
                    added_commands.add(func_name)

        # Discover and add classes and their methods
        for class_name, class_obj in inspect.getmembers(module, inspect.isclass):
            if not class_name.startswith('_') and class_obj.__module__.startswith(module.__name__):
                for method_name, method_obj in inspect.getmembers(class_obj, inspect.isfunction):
                    if not method_name.startswith('_'):
                        command_name = f"{class_name}.{method_name}"
                        if command_name not in added_commands:
                            method_parser = subparsers.add_parser(command_name, help=f"Execute the '{method_name}' method of the '{class_name}' class.")
                            method_parser.add_argument('--instance', type=str, required=True, help=f"String representation of the '{class_name}' instance.")
                            sig = inspect.signature(method_obj)
                            for param in sig.parameters.values():
                                if param.name == 'self':
                                    continue
                                arg_name = f'--{param.name}'
                                arg_type = param.annotation if param.annotation != inspect.Parameter.empty else str

                                # Use str for complex types for now
                                if arg_type not in [str, int, float, bool]:
                                    arg_type = str

                                is_list = get_origin(param.annotation) in (list, List)
                                nargs = '+' if is_list else None

                                if param.default == inspect.Parameter.empty:
                                    method_parser.add_argument(arg_name, type=arg_type, required=True, nargs=nargs)
                                else:
                                    if isinstance(param.default, bool) and not param.default:
                                        method_parser.add_argument(arg_name, action='store_true')
                                    else:
                                        method_parser.add_argument(arg_name, type=arg_type, default=param.default, nargs=nargs)
                            method_parser.set_defaults(func=method_obj, is_method=True, class_obj=class_obj)
                            added_commands.add(command_name)

    return parser

from type_handler import deserialize_arg, serialize_result

def main():
    """
    Main function to run the CLI.
    """
    parser = build_cli()
    args = parser.parse_args()

    sig = inspect.signature(args.func)
    func_args = {}
    for param in sig.parameters.values():
        if param.name == 'self':
            continue
        arg_value = getattr(args, param.name, None)
        if arg_value is not None:
            param_type = param.annotation
            is_list = get_origin(param_type) in (list, List)
            if is_list:
                item_type = param_type.__args__[0] if param_type.__args__ else str
                deserialized_value = [deserialize_arg(item, item_type) for item in arg_value]
            else:
                deserialized_value = deserialize_arg(arg_value, param.annotation)
            func_args[param.name] = deserialized_value

    if args.is_method:
        # For method calls, we need to instantiate the class.
        # We use the --instance argument for this.
        instance = deserialize_arg(args.instance, args.class_obj)
        result = args.func(instance, **func_args)
    else:
        result = args.func(**func_args)

    serialized_result = serialize_result(result)
    print(serialized_result)


if __name__ == "__main__":
    main()
