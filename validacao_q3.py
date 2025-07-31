import sympy
import sys
import os

# Adiciona o diretório src ao sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'ControlLab-Project/src/')))

try:
    from controllab.core.symbolic_tf import SymbolicTransferFunction
    # As funções de interconexão estão em locais diferentes do esperado pelo snippet
    from controllab.modeling.conversions import feedback_connection
except ImportError as e:
    print(f"Erro de importação: {e}")
    sys.exit(1)

# Define o símbolo 's'
s = sympy.Symbol('s')

print("--- Questão 3: Validação de Álgebra de Diagramas de Blocos ---")

# --- Adaptação da API ---
# O snippet do usuário assume a existência de funções 'series' e 'parallel'.
# A biblioteca implementa isso via sobrecarga de operadores.
# Vamos definir wrappers para manter o código de validação próximo ao original.
def series(tf1, tf2):
    return tf1 * tf2

def parallel(tf1, tf2):
    return tf1 + tf2

# A função de feedback está em 'conversions' e retorna uma tupla (tf, history)
def feedback(forward_tf, feedback_tf, sign=-1):
    # O sinal na função da biblioteca é o oposto do sinal no somador
    # sign=-1 (padrão) para realimentação negativa
    tf_obj, history = feedback_connection(forward_tf, feedback_tf, sign=sign)
    return tf_obj

# --- Validação do Módulo Interconnection ---

# Definir as funções de transferência dos componentes
K = 5
C = SymbolicTransferFunction([K], [1], s=s)
P1 = SymbolicTransferFunction([1], [1, 1], s=s)
P2 = SymbolicTransferFunction([10], [1, 5], s=s)
H = SymbolicTransferFunction([1], [0.5, 1], s=s)
# G_par(s) = 2/s. O denominador é [1, 0]
G_par = SymbolicTransferFunction([2], [1, 0], s=s)

print("\n--- Componentes do Sistema ---")
print(f"C(s) = {C}")
print(f"P1(s) = {P1}")
print(f"P2(s) = {P2}")
print(f"H(s) = {H}")
print(f"G_par(s) = {G_par}")

# Passo 1: Validar a conexão em série
P_series = series(P1, P2)
# Alternativamente, usando sobrecarga de operador para validação
P_series_op = P1 * P2
print(f"\nPlanta Equivalente P(s) = P1 * P2 = {P_series.simplify()}")
assert sympy.simplify(P_series.expression - P_series_op.expression) == 0
print("Validação de conexão em série: SUCESSO")

# Passo 3: Validar a conexão de realimentação
G_direct = C * P_series
G_cl = feedback(G_direct, H, sign=-1)
print(f"\nMalha Fechada G_cl(s) = {G_cl.simplify()}")

# Passo 4: Validar a conexão em paralelo
G_final = parallel(G_cl, G_par)
# Alternativamente, usando sobrecarga de operador
G_final_op = G_cl + G_par
print(f"\nFunção de Transferência Final G_final(s) = {G_final.simplify()}")
assert sympy.simplify(G_final.expression - G_final_op.expression) == 0
print("Validação de conexão em paralelo: SUCESSO")

# Verificação analítica
# Gfinal(s) = (s^3 + 33s^2 + 67s + 110) / (0.5s^4 + 4s^3 + 8.5s^2 + 55s)
num_final_manual_expr = s**3 + 33*s**2 + 67*s + 110
den_final_manual_expr = 0.5*s**4 + 4*s**3 + 8.5*s**2 + 55*s
G_final_manual = SymbolicTransferFunction(num_final_manual_expr, den_final_manual_expr, s=s)

print(f"\nResultado Analítico Manual = {G_final_manual.simplify()}")

# A asserção final compara o resultado da biblioteca com a derivação manual
diff = sympy.simplify(G_final.expression - G_final_manual.expression)
assert diff == 0, f"A diferença deveria ser 0, mas foi {diff}"
print("\nValidação BEM-SUCEDIDA: A simplificação do diagrama de blocos está correta.")
