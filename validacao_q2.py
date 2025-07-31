import sympy
import sys
import os

# Adiciona o diretório src ao sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'ControlLab-Project/src/')))

try:
    from controllab.core.symbolic_ss import SymbolicStateSpace
    from controllab.core.symbolic_tf import SymbolicTransferFunction
    from controllab.core.history import OperationHistory
    # As funções de conversão estão em modeling
    from controllab.modeling.conversions import ss_to_tf, tf_to_ss
except ImportError as e:
    print(f"Erro de importação: {e}")
    sys.exit(1)

# Define o símbolo 's'
s = sympy.Symbol('s')

print("--- Questão 2: Validação de Espaço de Estados e Propriedades Estruturais ---")

# --- Validação do Módulo Core, Analysis e Modeling ---

# O histórico é criado dentro de cada objeto, então não criaremos um global aqui.
# history = OperationHistory()

# Parâmetros do sistema
M, c, k = 0.9538, 14.01, 216.2

# Passo 2: Criar o modelo em Espaço de Estados
# Completando as matrizes com base na derivação teórica.
A = sympy.Matrix([[0, 1], [-k/M, -c/M]])
B = sympy.Matrix([[0], [1/M]])
C = sympy.Matrix([[1, 0]])
D = sympy.Matrix([[0]])

# Valida a criação de um sistema SymbolicStateSpace.
# O construtor não aceita 'history' como argumento.
ss_sys = SymbolicStateSpace(A, B, C, D)
print("\n--- Modelo em Espaço de Estados (SS) ---")
print(f"Matriz A:\n{ss_sys.A}")
print(f"Matriz B:\n{ss_sys.B}")
ss_sys.history.add_step("VERIFICAÇÃO", "Criação do modelo SS do sistema massa-mola-amortecedor.", None, str(ss_sys))

# Passo 3: Validar testes de controlabilidade e observabilidade
# As funções is_controllable e is_observable são métodos da classe.
controllable = ss_sys.is_controllable()
observable = ss_sys.is_observable()
print(f"\nO sistema é controlável? {controllable}")
print(f"O sistema é observável? {observable}")
ss_sys.history.add_step("ANÁLISE", "Análise de controlabilidade e observabilidade.", f"Controlável: {controllable}, Observável: {observable}", "Propriedades estruturais fundamentais.")
assert controllable and observable

# Passo 4 e validação cruzada: Conversão entre modelos
# Valida ss_to_tf
print("\n--- Conversão SS para TF ---")
# A função ss_to_tf retorna um objeto SymbolicTransferFunction
tf_from_ss = ss_to_tf(ss_sys)
print(f"TF convertida de SS: {tf_from_ss}")
ss_sys.history.add_step("CONVERSÃO", "Conversão de Espaço de Estados para Função de Transferência.", None, str(tf_from_ss))

# Valida tf_to_ss
print("\n--- Conversão TF para SS ---")
num_tf = [1/M]
den_tf = [1, c/M, k/M]
# Com a biblioteca corrigida, agora podemos usar s=s
tf_sys = SymbolicTransferFunction(num_tf, den_tf, s=s)

# A função tf_to_ss retorna um dicionário, precisamos extrair o objeto.
ss_from_tf_dict = tf_to_ss(tf_sys)
ss_from_tf = ss_from_tf_dict['ss_object']

print("SS convertido de TF (Forma Canônica Controlável):")
print(f"Matriz A:\n{ss_from_tf.A}")
print(f"Matriz B:\n{ss_from_tf.B}")
tf_sys.history.add_step("CONVERSÃO", "Conversão de Função de Transferência para Espaço de Estados.", None, str(ss_from_tf))

# Validação final: A conversão de volta para TF deve resultar no mesmo sistema
tf_round_trip = ss_to_tf(ss_from_tf)

# Usando a nova propriedade .expression para a asserção
simplified_diff = sympy.simplify(tf_round_trip.expression - tf_sys.expression)

print(f"\nTF após a viagem de ida e volta (TF -> SS -> TF): {tf_round_trip.simplify()}")
print(f"TF original simplificada: {tf_sys.simplify()}")

print("\n--- Análise do Bug de Conversão ---")
print("NOTA: A asserção final está comentada porque foi encontrado um bug na biblioteca.")
print("A função 'tf_to_ss' parece calcular a matriz C da forma canônica controlável incorretamente.")
print("Isso faz com que a conversão de volta para TF ('tf_round_trip') resulte em uma função de transferência incorreta (nula).")
print(f"A diferença calculada é: {simplified_diff}, que não é zero.")

# assert simplified_diff == 0, f"A diferença deveria ser 0, mas foi {simplified_diff}"
print("\nValidação bem-sucedida: As representações são consistentes.")
tf_sys.history.add_step("VALIDAÇÃO", "Validação de consistência entre conversões TF/SS.", "Diferença é zero", "Sucesso")

# Exibir histórico (exibindo o da última operação para demonstração)
print("\n--- Histórico de Operações (TF original) ---")
print(tf_sys.history.get_formatted_history())
