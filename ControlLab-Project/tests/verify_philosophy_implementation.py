# tests/verify_philosophy_implementation.py

import sys
import os
import pytest
import sympy as sp
import numpy as np

# Adiciona o diretório 'src' ao path para que possamos importar os módulos do ControlLab
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Importações necessárias para os testes
from controllab.core.symbolic_tf import SymbolicTransferFunction
from controllab.design.antiwindup import design_antiwindup_compensation, SaturationLimits
from controllab.design.pole_placement import StateSpaceController
from controllab.numerical.factory import NumericalSystemFactory

# --- Início do Script de Verificação ---

print("="*80)
print("==  AUDITORIA DA ARQUITETURA DE DIAGNÓSTICO INTELIGENTE DO CONTROLLAB  ==")
print("="*80)

# --- Teste 1: Verificação da Fundação - O "Gravador de Voo" está funcionando? ---

def test_history_recording():
    """
    Verifica se a classe SymbolicTransferFunction está corretamente
    registrando seu histórico de criação e modificação.
    """
    print("\n[FASE 1]: Verificando a gravação do histórico no `core`...")
    s = sp.Symbol('s')

    # 1.1. Verifica o registro na criação do objeto
    tf1 = SymbolicTransferFunction(1, s + 1, s)
    assert hasattr(tf1, 'history'), "FALHA: O objeto não possui o atributo 'history'."
    assert len(tf1.history.steps) == 1, "FALHA: O número de passos no histórico inicial está incorreto."
    assert tf1.history.steps[0].operation == "Criação do Objeto", "FALHA: A operação de criação não foi registrada corretamente."
    print("  [✓] SUCESSO: Objeto registra sua 'Criação' no histórico.")

    # 1.2. Verifica o registro em uma operação (ex: simplificação)
    tf2_unsimplified = SymbolicTransferFunction((s+1)/(s+1), 1, s)
    tf2_simplified = tf2_unsimplified.simplify()

    # O histórico do novo objeto deve conter os passos do pai + o novo passo
    assert len(tf2_simplified.history.steps) > len(tf2_unsimplified.history.steps), "FALHA: A operação de simplificação não adicionou um passo ao histórico."
    assert tf2_simplified.history.steps[-1].operation == "Simplificação", "FALHA: A operação de simplificação não foi registrada corretamente."
    print("  [✓] SUCESSO: Operações (como simplify) adicionam novos passos ao histórico.")


# --- Teste 2: Verificação do "Detetive" - O diagnóstico de Anti-Windup ---

def test_anti_windup_intelligent_diagnosis():
    """
    Verifica se a função de anti-windup falha de forma inteligente,
    fornecendo um diagnóstico completo usando o histórico.
    Este é o teste principal que valida a correção do bug original.
    """
    print("\n[FASE 2]: Verificando o diagnóstico inteligente no `design.antiwindup`...")
    s = sp.Symbol('s')
    Kp_pi, Ki_pi = 5, 2

    # 2.1. Criação INTENCIONALMENTE FALHA do controlador, como no teste original
    # C(s) = ((Kp*s + Ki)/s) / s  ->  (5s + 2) / s^2
    flawed_controller = SymbolicTransferFunction(Kp_pi + Ki_pi/s, s)

    # 2.2. Prepara os outros argumentos para a função
    plant = SymbolicTransferFunction(1, s + 1, s)
    saturation = SaturationLimits(u_min=-10, u_max=10)

    # 2.3. Verifica se a função levanta um ValueError e inspeciona a mensagem
    with pytest.raises(ValueError) as excinfo:
        design_antiwindup_compensation(
            controller=flawed_controller,
            plant=plant,
            saturation_limits=saturation,
            method='back_calculation'
        )

    # 2.4. A "autópsia": Inspeciona a mensagem de erro para garantir que ela é diagnóstica
    error_message = str(excinfo.value)

    assert "FALHA NO PROJETO ANTI-WINDUP" in error_message, "FALHA: A mensagem de erro não tem o cabeçalho esperado."
    assert "MOTIVO TÉCNICO DA FALHA" in error_message, "FALHA: A mensagem de erro não explica o motivo técnico."
    assert "não é válido para um controlador PID padrão" in error_message, "FALHA: A causa técnica específica (denominador s**2) não foi mencionada."
    assert "DIAGNÓSTICO DA CAUSA RAIZ" in error_message, "FALHA: A mensagem de erro não inclui a seção de diagnóstico do histórico."
    assert "Estado Antes: {'numerador_inicial': 5 + 2/s, 'denominador_inicial': s}" in error_message, "FALHA: O relatório do histórico dentro do erro está incorreto ou ausente."
    assert "AÇÃO RECOMENDADA" in error_message, "FALHA: A mensagem de erro não fornece uma sugestão de correção para o usuário."

    print("  [✓] SUCESSO: Anti-windup falha de forma inteligente e fornece um relatório de diagnóstico completo.")

# --- Teste 3: Verificação de Outros Módulos "Detetives" ---

def test_pole_placement_intelligent_diagnosis():
    """
    Verifica se a alocação de polos valida a controlabilidade e fornece um
    diagnóstico claro em caso de falha.
    """
    print("\n[FASE 3]: Verificando o diagnóstico inteligente no `design.pole_placement`...")

    # 3.1. Cria um sistema NÃO-CONTROLÁVEL
    A = sp.Matrix([[1, 1, 0], [0, 1, 0], [0, 0, 2]])
    B = sp.Matrix([[0], [1], [0]]) # A entrada não afeta o terceiro estado
    desired_poles = [-1, -2, -3]

    controller = StateSpaceController()

    # 3.2. Verifica se a função levanta o erro correto e inspeciona a mensagem
    with pytest.raises(np.linalg.LinAlgError) as excinfo:
        controller.pole_placement(A, B, desired_poles)

    error_message = str(excinfo.value)

    assert "FALHA NA ALOCAÇÃO DE POLOS" in error_message
    assert "O sistema não é controlável" in error_message
    assert "o posto calculado foi 2" in error_message # O posto é 2, não 3
    assert "MATRIZ DE CONTROLABILIDADE CALCULADA" in error_message

    print("  [✓] SUCESSO: Alocação de polos identifica sistema não-controlável e fornece diagnóstico.")


def test_numerical_conversion_intelligent_diagnosis():
    """
    Verifica se a conversão para numérico falha de forma inteligente se a
    expressão ainda contiver símbolos livres.
    """
    print("\n[FASE 4]: Verificando o diagnóstico inteligente no `numerical.factory`...")
    s = sp.Symbol('s')
    K = sp.Symbol('K') # Símbolo livre

    # 4.1. Cria uma função de transferência com um parâmetro simbólico
    symbolic_tf_with_param = SymbolicTransferFunction(K, s**2 + s + K, s)

    factory = NumericalSystemFactory()

    # 4.2. Verifica se a conversão levanta o erro correto
    with pytest.raises(ValueError) as excinfo:
        factory.create_from_symbolic(symbolic_tf_with_param)

    error_message = str(excinfo.value)

    assert "FALHA NA CONVERSÃO SIMBÓLICO->NUMÉRICO" in error_message
    assert "SÍMBOLOS NÃO RESOLVIDOS ENCONTRADOS" in error_message
    assert "{K}" in error_message # Verifica se o símbolo 'K' foi identificado
    assert "AÇÃO RECOMENDADA" in error_message
    assert "Use o método `.subs(" in error_message

    print("  [✓] SUCESSO: Conversão numérica identifica parâmetros simbólicos e guia o usuário.")


# --- Execução dos Testes ---
if __name__ == "__main__":
    pytest.main(['-s', __file__])
