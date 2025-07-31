# -*- coding: utf-8 -*-

"""
================================================================================
ControlLab - Interface Simbólico-Numérica (VERSÃO CORRIGIDA)
================================================================================
Este módulo serve como a ponte principal entre o núcleo simbólico (SymPy) e
as bibliotecas de computação numérica (NumPy, SciPy, python-control).
"""

import numpy as np
import sympy as sp

# É uma boa prática usar um bloco try-except para dependências opcionais
try:
    from control import TransferFunction, step_response
except ImportError:
    print("Aviso: A biblioteca 'python-control' não foi encontrada. As funcionalidades numéricas serão limitadas.")
    TransferFunction = None
    step_response = None

from controllab.core.symbolic_tf import SymbolicTransferFunction

class NumericTransferFunction:
    """
    Representação numérica de uma função de transferência, servindo como um
    wrapper para objetos do python-control.
    """
    def __init__(self, num, den):
        if not all(isinstance(c, (int, float, complex)) for c in num) or \
           not all(isinstance(c, (int, float, complex)) for c in den):
            raise TypeError("Os coeficientes do numerador e denominador devem ser tipos numéricos nativos (int, float, complex).")

        self.num = num
        self.den = den

        if TransferFunction:
            self.control_tf = TransferFunction(self.num, self.den)
        else:
            self.control_tf = None

    @classmethod
    def from_symbolic(cls, symbolic_tf: SymbolicTransferFunction):
        """
        Cria uma instância NumericTransferFunction a partir de uma
        SymbolicTransferFunction. Este método contém a correção definitiva.
        """
        s = symbolic_tf.variable

        num_poly = sp.Poly(symbolic_tf.numerator, s)
        den_poly = sp.Poly(symbolic_tf.denominator, s)

        # =============================== CORREÇÃO DEFINITIVA ===============================
        # Usamos o método .evalf() em cada coeficiente para obter uma
        # representação numérica de ponto flutuante que pode ser convertida
        # com segurança para um float nativo do Python.
        try:
            num_coeffs = [float(coeff.evalf()) for coeff in num_poly.all_coeffs()]
            den_coeffs = [float(coeff.evalf()) for coeff in den_poly.all_coeffs()]
        except Exception as e:
            print(f"❌ Erro de conversão em from_symbolic: {e}")
            print(f"   Numerador com problema: {num_poly.expr}")
            print(f"   Denominador com problema: {den_poly.expr}")
            raise
        # ===============================================================================

        return cls(num_coeffs, den_coeffs)

    def __str__(self):
        # Arredonda os coeficientes para uma melhor visualização
        num_str = [f"{c:.3f}" for c in self.num]
        den_str = [f"{c:.3f}" for c in self.den]
        return f"NumericTransferFunction(num={num_str}, den={den_str})"

def simulate_system_response(numeric_sys: NumericTransferFunction, input_type='step', time_vector=None):
    """
    Simula a resposta temporal de um sistema numérico usando a biblioteca 'control'.
    """
    if not step_response:
        raise ImportError("A biblioteca 'python-control' é necessária para a simulação.")

    if input_type == 'step':
        if time_vector is None:
            # Cria um vetor de tempo padrão se nenhum for fornecido
            time_vector = np.linspace(0, 10, 500)

        # Usa a função step_response da biblioteca 'control'
        T, yout = step_response(numeric_sys.control_tf, T=time_vector)
        return T, yout
    else:
        raise ValueError(f"Tipo de entrada '{input_type}' não suportado para simulação.")
