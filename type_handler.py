import sympy as sp
from controllab.core.symbolic_tf import SymbolicTransferFunction
from controllab.core.symbolic_ss import SymbolicStateSpace

def deserialize_arg(value: str, arg_type: type):
    """
    Deserializes a string argument to the specified type.
    """
    if arg_type == SymbolicTransferFunction:
        s = sp.Symbol('s')
        expr = sp.sympify(value)
        num, den = sp.fraction(expr)
        return SymbolicTransferFunction(num, den)
    elif arg_type == SymbolicStateSpace:
        # Assuming the input is a string representation of a tuple of matrices
        # e.g., "[[1, 2], [3, 4]]; [[5], [6]]; [[7, 8]]; [[9]]"
        parts = value.split(';')
        A = sp.Matrix(eval(parts[0]))
        B = sp.Matrix(eval(parts[1]))
        C = sp.Matrix(eval(parts[2]))
        D = sp.Matrix(eval(parts[3]))
        return SymbolicStateSpace(A, B, C, D)
    # Add more handlers for other complex types here
    else:
        # For simple types, just return the value as is,
        # argparse has already converted it.
        return value

def serialize_result(result) -> str:
    """
    Serializes a result object to a string.
    """
    if hasattr(result, 'get_full_report'):
        return result.get_full_report()
    if hasattr(result, 'get_formatted_report'):
        return result.get_formatted_report()
    if isinstance(result, (SymbolicTransferFunction, SymbolicStateSpace)):
        return str(result)
    # Add more handlers for other complex types here
    else:
        return str(result)
