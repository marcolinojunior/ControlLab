import sympy as sp
import numpy as np

try:
    from control import TransferFunction, step_response
except ImportError:
    TransferFunction = None
    step_response = None

from controllab.core.symbolic_tf import SymbolicTransferFunction

class PatchedNumericTransferFunction:
    """
    Representação numérica de uma função de transferência, com um método de
    conversão robusto que funciona para qualquer polinômio.
    """
    def __init__(self, num, den):
        if not all(isinstance(c, (int, float, complex)) for c in num) or \
           not all(isinstance(c, (int, float, complex)) for c in den):
            raise TypeError("Os coeficientes devem ser tipos numéricos nativos.")

        self.num = num
        self.den = den

        if TransferFunction:
            self.control_tf = TransferFunction(self.num, self.den)
        else:
            self.control_tf = None

    @classmethod
    def from_symbolic(cls, symbolic_tf: SymbolicTransferFunction):
        """
        Cria uma instância numérica a partir de uma simbólica usando um
        método de extração de coeficientes robusto.
        """
        s = symbolic_tf.variable

        # Expande as expressões para garantir a forma polinomial padrão
        num_expr = sp.expand(symbolic_tf.numerator)
        den_expr = sp.expand(symbolic_tf.denominator)

        # Determina o grau dos polinômios
        num_degree = sp.degree(num_expr, gen=s)
        den_degree = sp.degree(den_expr, gen=s)

        # Extrai coeficientes de forma robusta usando o método .coeff()
        num_coeffs = []
        for i in range(num_degree, -1, -1):
            coeff = num_expr.coeff(s, i)
            num_coeffs.append(float(coeff))

        den_coeffs = []
        for i in range(den_degree, -1, -1):
            coeff = den_expr.coeff(s, i)
            den_coeffs.append(float(coeff))

        return cls(num_coeffs, den_coeffs)

    def __str__(self):
        num_str = [f"{c:.3f}" for c in self.num]
        den_str = [f"{c:.3f}" for c in self.den]
        return f"PatchedNumericTransferFunction(num={num_str}, den={den_str})"

def patched_simulate_system_response(numeric_sys: PatchedNumericTransferFunction, input_type='step', time_vector=None):
    """
    Simula a resposta temporal usando a classe corrigida.
    """
    if not step_response:
        raise ImportError("A biblioteca 'python-control' é necessária para a simulação.")

    if input_type == 'step':
        if time_vector is None:
            time_vector = np.linspace(0, 10, 500)

        T, yout = step_response(numeric_sys.control_tf, T=time_vector)
        return T, yout
    else:
        raise ValueError(f"Tipo de entrada '{input_type}' não suportado.")
