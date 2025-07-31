import sympy
import sys
import os

# Adiciona o diretório src ao sys.path para encontrar o módulo controllab
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'ControlLab-Project/src/')))

try:
    from controllab.core.symbolic_tf import SymbolicTransferFunction
    from controllab.core.history import OperationHistory
    from controllab.core.transforms import LaplaceTransform
except ImportError as e:
    print(f"Erro de importação: {e}")
    print("Verifique se o caminho para o projeto ControlLab está correto.")
    sys.exit(1)

# Define o símbolo 's' para as expressões de Laplace
s = sympy.Symbol('s')
t = sympy.Symbol('t', real=True, positive=True)

print("--- Questão 1: Validação de Fundamentos de Modelagem e Análise ---")

# --- Validação do Módulo Core e Transforms ---

# Histórico de operações para rastreamento pedagógico
# Nota: A classe SymbolicTransferFunction cria seu próprio histórico.
# Para ter um histórico unificado, passaríamos a mesma instância,
# mas para este teste, vamos seguir o fluxo da classe.

# Passo 2 e 5: Criar a Função de Transferência Simbólica
R, L, C = 12, 0.15, 100e-6

# G(s) = (1/(LC)) / (s^2 + (R/L)s + 1/(LC))
num_expr = 1 / (L * C)
den_expr = s**2 + (R/L)*s + 1/(L*C)

# A capacidade de definir o sistema simbolicamente é a base para todas as análises.
# Valida a classe SymbolicTransferFunction e sua inicialização.
G = SymbolicTransferFunction(num_expr, den_expr, variable=s)

print("\n--- Função de Transferência do Circuito RLC ---")
print(f"G(s) = {G}")
# Acessando o histórico da instância de G
G.history.add_step("VERIFICAÇÃO", "Criação da FT do circuito RLC validada.", None, str(G))

# Passo 6: Validar o cálculo de polos e zeros
# Esta etapa conecta o conceito teórico de polos/zeros com a implementação do software.
poles = G.poles()
zeros = G.zeros()
print(f"\nPolos do sistema: {poles}")
print(f"Zeros do sistema: {zeros}")
G.history.add_step("ANÁLISE", "Cálculo de polos e zeros.", f"Polos: {poles}, Zeros: {zeros}", "Polos determinam a estabilidade e a resposta natural.")

# Passo 7: Validar a Transformada de Laplace e a Inversa
# Valida a capacidade da biblioteca de trabalhar entre o domínio do tempo e da frequência.

# Instanciar a classe de transformada
lt = LaplaceTransform()

# Transformada de Laplace da entrada (degrau unitário)
u_t = sympy.Heaviside(t)
U_s = lt.transform(u_t, t, s)
print(f"\nTransformada de Laplace da entrada degrau u(t): U(s) = {U_s}")
G.history.add_step("TRANSFORMADA", "Transformada de Laplace do degrau unitário.", f"u(t) -> U(s) = {U_s}", "L{u(t)} = 1/s")

# Saída no domínio de Laplace
# Usamos (G.numerator / G.denominator) como o atributo .expression
Vc_s = (G.numerator / G.denominator) * U_s
print(f"Saída no domínio de Laplace: Vc(s) = {Vc_s}")
G.history.add_step("CÁLCULO", "Cálculo da saída Vc(s) no domínio de Laplace.", str(Vc_s), "Vc(s) = G(s) * U(s)")

# Transformada Inversa de Laplace para obter a resposta no tempo
print("\nCalculando a resposta ao degrau Vc(t) via transformada inversa...")
vc_t = lt.inverse_transform(Vc_s, s, t)
G.history.add_step("TRANSFORMADA_INVERSA", "Cálculo da resposta ao degrau Vc(t).", str(vc_t), "Vc(t) = L⁻¹{Vc(s)}")

print("\n--- Resposta ao Degrau Unitário no Domínio do Tempo ---")
# Simplificar a expressão para melhor visualização
vc_t_simplified = sympy.simplify(vc_t)
print(f"Vc(t) = {vc_t_simplified}")

# Validação das funcionalidades pedagógicas (Histórico)
print("\n--- Histórico de Operações (Validação Pedagógica) ---")
# O histórico está dentro do objeto G, conforme a implementação atual
# Corrigindo o nome do método para get_formatted_history
print(G.history.get_formatted_history())
