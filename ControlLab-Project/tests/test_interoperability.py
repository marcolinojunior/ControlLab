# DENTRO DE: tests/test_interoperability.py
import pytest
import sympy as sp
import control as ct
from controllab.core.symbolic_tf import SymbolicTransferFunction

s = sp.symbols('s')

def test_from_numeric_siso_standard():
    """Testa a conversão de um sistema SISO padrão."""
    # Sistema numérico: G(s) = (s + 1) / (s^2 + 2s + 5)
    num_coeffs = [1, 1]
    den_coeffs = [1, 2, 5]
    numeric_sys = ct.TransferFunction(num_coeffs, den_coeffs)

    symbolic_sys = SymbolicTransferFunction.from_numeric(numeric_sys)

    expected_num = s + 1
    expected_den = s**2 + 2*s + 5

    assert symbolic_sys.numerator.equals(expected_num)
    assert symbolic_sys.denominator.equals(expected_den)

def test_from_numeric_pure_gain():
    """Testa a conversão de um sistema que é apenas um ganho."""
    numeric_sys = ct.TransferFunction([5], [1]) # G(s) = 5

    symbolic_sys = SymbolicTransferFunction.from_numeric(numeric_sys)

    assert symbolic_sys.numerator.equals(5)
    assert symbolic_sys.denominator.equals(1)

def test_from_numeric_raises_error_for_mimo():
    """Testa se a função levanta um erro para sistemas MIMO, como esperado."""
    # Sistema MIMO 2x2
    num_mimo = [[[1], [0]], [[0], [1]]]
    den_mimo = [[[1, 1], [1, 1]], [[1, 1], [1, 1]]]
    mimo_sys = ct.TransferFunction(num_mimo, den_mimo)

    with pytest.raises(NotImplementedError, match="A conversão de sistemas MIMO ainda não é suportada."):
        SymbolicTransferFunction.from_numeric(mimo_sys)

def test_from_numeric_raises_error_for_invalid_type():
    """Testa se a função levanta um erro para tipos de input inválidos."""
    invalid_input = [1, 2, 3] # Uma lista simples não é um objeto TF

    with pytest.raises(TypeError, match="O input deve ser do tipo 'control.TransferFunction'"):
        SymbolicTransferFunction.from_numeric(invalid_input)
