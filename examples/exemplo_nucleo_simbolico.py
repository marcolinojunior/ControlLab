"""
Exemplo do Núcleo Simbólico
============================

Este script demonstra as funcionalidades do núcleo simbólico do ControlLab.
"""

# --- INÍCIO DO CÓDIGO DE CORREÇÃO ---
import sys
import os

# Adiciona o diretório raiz do projeto (um nível acima de 'examples') ao path do Python
# para que ele possa encontrar a pasta 'src'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ControlLab-Project'))
sys.path.insert(0, project_root)
# --- FIM DO CÓDIGO DE CORREÇÃO ---

import sympy as sp
from src.controllab.core.symbolic_tf import SymbolicTransferFunction
from src.controllab.core.symbolic_ss import SymbolicStateSpace

# Criação da função de transferência
print("="*60)
print("Exemplo de Funcionalidades da Função de Transferência")
print("="*60)
s = sp.Symbol('s')
G = SymbolicTransferFunction(10, s**2 + 3*s + 2, s)
print(f"Função de Transferência: {G}")

# Extração de coeficientes
print("\n1. Extração de coeficientes:")
coeffs = G.get_coefficients()
print(f"   Numerador: {coeffs['num']}")
print(f"   Denominador: {coeffs['den']}")

# Transformação de Tustin
print("\n2. Transformação de Tustin (T=0.1):")
T = 0.1
G_discrete = G.apply_tustin_transform(T)
print(f"   G(z) = {G_discrete}")

# Equação característica
print("\n3. Equação característica:")
char_eq = G.characteristic_equation()
print(f"   Equação: {char_eq} = 0")

# Análise de polos e zeros
print("\n4. Análise de Polos e Zeros:")
poles = G.poles()
zeros = G.zeros()
print(f"   Polos: {poles}")
print(f"   Zeros: {zeros}")

# Avaliação em um ponto
print("\n5. Avaliação em s=1:")
value = G.evaluate(1)
print(f"   G(1) = {value}")

# Espaço de estados
print("\n" + "="*60)
print("Exemplo de Funcionalidades do Espaço de Estados")
print("="*60)
A = sp.Matrix([[0, 1], [-2, -3]])
B = sp.Matrix([[0], [1]])
C = sp.Matrix([[1, 0]])
D = sp.Matrix([[0]])
sys = SymbolicStateSpace(A, B, C, D)
print("1. Sistema criado:")
print(f"   A = {A}")
print(f"   B = {B}")
print(f"   C = {C}")
print(f"   D = {D}")

# Conversão para função de transferência
print("\n2. Função de transferência:")
tf = sys.transfer_function()
print(f"   G(s) = {tf}")

# Autovalores
print("\n3. Autovalores:")
eigenvals = sys.eigenvalues()
print(f"   Autovalores: {eigenvals}")

# Controlabilidade e Observabilidade
print("\n4. Controlabilidade e Observabilidade:")
controllable = sys.is_controllable()
observable = sys.is_observable()
print(f"   Controlável: {controllable}")
print(f"   Observável: {observable}")

# Operações simbólicas
print("\n" + "="*60)
print("Exemplo de Operações Simbólicas")
print("="*60)
G1 = SymbolicTransferFunction(1, s + 1, s)
G2 = SymbolicTransferFunction(2, s + 2, s)
print("1. Sistemas criados:")
print(f"   G1(s) = {G1}")
print(f"   G2(s) = {G2}")

# Multiplicação
print("\n2. Multiplicação:")
G_mult = G1 * G2
print(f"   G1 * G2 = {G_mult}")

# Adição
print("\n3. Adição:")
G_add = G1 + G2
print(f"   G1 + G2 = {G_add}")

# Realimentação
print("\n4. Realimentação:")
G_feedback = G1.feedback(G2)
print(f"   G1 / (1 + G1*G2) = {G_feedback}")

# Histórico
print("\n5. Histórico de operações:")
history_report = G_feedback.history.get_formatted_report()
print(history_report)
