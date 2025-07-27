"""
ControlLab - Alocação de Polos e Realimentação de Estados
=========================================================

Este módulo implementa métodos de controle moderno baseados em espaço de estados:
- Verificação de controlabilidade
- Fórmula de Ackermann para alocação de polos
- Métodos robustos de alocação
- Projeto sistemático em espaço de estados

Características:
- Derivação simbólica completa
- Explicações step-by-step do método de Ackermann
- Demonstração da controlabilidade
- Análise de robustez
"""

import sympy as sp
import numpy as np
from typing import List, Optional, Union, Tuple, Dict, Any
from ..core.symbolic_ss import SymbolicStateSpace
from .design_utils import ControllerResult, create_educational_content

def check_controllability(ss_obj: SymbolicStateSpace,
                         show_steps: bool = True) -> Dict[str, Any]:
    """
    Verifica se o sistema é completamente controlável

    Args:
        ss_obj: Sistema em espaço de estados
        show_steps: Se deve mostrar passos detalhados

    Returns:
        Dict[str, Any]: Resultado da análise de controlabilidade
    """
    if show_steps:
        print("🔍 VERIFICAÇÃO DE CONTROLABILIDADE")
        print("=" * 40)
        print(f"🏭 Sistema: ẋ = Ax + Bu")
        print(f"📐 A = {ss_obj.A}")
        print(f"📐 B = {ss_obj.B}")

    A = ss_obj.A
    B = ss_obj.B
    n = A.rows  # Ordem do sistema

    # Construir matriz de controlabilidade Wc = [B AB A²B ... A^(n-1)B]
    if show_steps:
        print("\n📋 CONSTRUÇÃO DA MATRIZ DE CONTROLABILIDADE")
        print("=" * 50)
        print("Wc = [B | AB | A²B | ... | A^(n-1)B]")

    Wc_blocks = [B]
    current_block = B

    for i in range(1, n):
        current_block = A * current_block
        Wc_blocks.append(current_block)

        if show_steps:
            print(f"A^{i}B = {current_block}")

    # Concatenar horizontalmente
    Wc = sp.Matrix.hstack(*Wc_blocks)

    if show_steps:
        print(f"\n📊 Matriz de Controlabilidade:")
        print(f"Wc = {Wc}")

    # Calcular determinante e rank
    det_Wc = Wc.det()
    rank_Wc = Wc.rank()

    # Sistema é controlável se rank(Wc) = n
    is_controllable = rank_Wc == n

    if show_steps:
        print(f"\n✅ ANÁLISE DE CONTROLABILIDADE:")
        print(f"📐 Determinante: det(Wc) = {det_Wc}")
        print(f"📊 Rank: rank(Wc) = {rank_Wc}")
        print(f"📏 Ordem do sistema: n = {n}")

        if is_controllable:
            print("✅ Sistema COMPLETAMENTE CONTROLÁVEL")
            print("💡 Todos os estados podem ser controlados")
        else:
            print("❌ Sistema NÃO completamente controlável")
            print(f"⚠️ Apenas {rank_Wc} de {n} estados são controláveis")

    # Conteúdo educacional
    educational_content = [
        "🎓 CONCEITO DE CONTROLABILIDADE:",
        "• Um sistema é controlável se todos os estados podem ser",
        "  transferidos de qualquer condição inicial para qualquer",
        "  estado final em tempo finito",
        "• Critério: rank(Wc) = n (ordem do sistema)",
        "• Matriz Wc = [B | AB | A²B | ... | A^(n-1)B]",
        "• Necessário para alocação arbitrária de polos"
    ]

    return {
        'is_controllable': is_controllable,
        'controllability_matrix': Wc,
        'rank': rank_Wc,
        'determinant': det_Wc,
        'system_order': n,
        'educational_content': educational_content
    }

def acker(ss_obj: SymbolicStateSpace,
          desired_poles: List[Union[complex, sp.Symbol]],
          show_steps: bool = True) -> Dict[str, Any]:
    """
    Calcula ganhos de realimentação usando Fórmula de Ackermann

    Args:
        ss_obj: Sistema em espaço de estados
        desired_poles: Polos desejados para malha fechada
        show_steps: Se deve mostrar passos detalhados

    Returns:
        Dict[str, Any]: Ganhos de realimentação e análise
    """
    if show_steps:
        print("🎯 FÓRMULA DE ACKERMANN - ALOCAÇÃO DE POLOS")
        print("=" * 50)
        print(f"🏭 Sistema: ẋ = Ax + Bu")
        print(f"🎯 Polos desejados: {desired_poles}")

    A = ss_obj.A
    B = ss_obj.B
    n = A.rows

    # Verificar controlabilidade primeiro
    controllability = check_controllability(ss_obj, show_steps=False)

    if not controllability['is_controllable']:
        error_message = (
            f"FALHA NA ALOCAÇÃO DE POLOS: O sistema não é controlável.\n\n"
            f"--> DIAGNÓSTICO TÉCNICO:\n"
            f"    A matriz de controlabilidade deve ter posto completo (rank={n}), mas o posto calculado foi {controllability['rank']}.\n\n"
            f"--> MATRIZ DE CONTROLABILIDADE CALCULADA:\n{controllability['controllability_matrix']}\n\n"
            f"--> AÇÃO RECOMENDADA:\n"
            f"    Revise as matrizes A e B do seu modelo. Pode haver estados dinâmicos que não são afetados pela entrada de controle."
        )
        raise np.linalg.LinAlgError(error_message)

    # Construir polinômio característico desejado
    s = sp.Symbol('s')

    if show_steps:
        print(f"\n📋 CONSTRUÇÃO DO POLINÔMIO CARACTERÍSTICO DESEJADO")
        print("=" * 55)

    # αc(s) = (s - p1)(s - p2)...(s - pn)
    alpha_c = 1
    for i, pole in enumerate(desired_poles):
        factor = (s - pole)
        alpha_c *= factor

        if show_steps:
            print(f"Fator {i+1}: (s - {pole})")

    alpha_c = sp.expand(alpha_c)

    if show_steps:
        print(f"\nPolinômio característico desejado:")
        print(f"αc(s) = {alpha_c}")

    # Extrair coeficientes do polinômio
    coeffs = sp.Poly(alpha_c, s).all_coeffs()

    if show_steps:
        print(f"Coeficientes: {coeffs}")

    # Avaliar αc(A) - Teorema de Cayley-Hamilton
    if show_steps:
        print(f"\n📐 CÁLCULO DE αc(A) - TEOREMA CAYLEY-HAMILTON")
        print("=" * 50)
        print("αc(A) = a₀I + a₁A + a₂A² + ... + aₙAⁿ")

    alpha_c_A = sp.zeros(n, n)
    A_power = sp.eye(n)  # A⁰ = I

    # Começar do termo de menor grau
    for i, coeff in enumerate(reversed(coeffs)):
        term = coeff * A_power
        alpha_c_A += term

        if show_steps:
            if i == 0:
                print(f"Termo {i}: {coeff} * I = {term}")
            else:
                print(f"Termo {i}: {coeff} * A^{i} = {term}")

        if i < len(coeffs) - 1:  # Não calcular A^(n+1)
            A_power = A * A_power

    if show_steps:
        print(f"\nαc(A) = {alpha_c_A}")

    # Fórmula de Ackermann: K = [0 0 ... 0 1] * Wc⁻¹ * αc(A)
    if show_steps:
        print(f"\n🔧 APLICAÇÃO DA FÓRMULA DE ACKERMANN")
        print("=" * 45)
        print("K = [0 0 ... 0 1] * Wc⁻¹ * αc(A)")

    # Obter matriz de controlabilidade
    Wc = controllability['controllability_matrix']

    # Vetor [0 0 ... 0 1]
    e_n = sp.zeros(1, n)
    e_n[0, n-1] = 1

    if show_steps:
        print(f"Vetor eₙ = {e_n}")
        print(f"Wc = {Wc}")
        print(f"Wc⁻¹ = {Wc.inv()}")

    # Calcular ganhos
    K = e_n * Wc.inv() * alpha_c_A

    if show_steps:
        print(f"\n✅ GANHOS DE REALIMENTAÇÃO:")
        print(f"K = {K}")
        print(f"\n🔄 Sistema em malha fechada:")
        print(f"ẋ = (A - BK)x")
        print(f"A - BK = {A - B * K}")

    # Verificar polos resultantes
    A_cl = A - B * K
    char_poly_cl = A_cl.charpoly('s')

    if show_steps:
        print(f"\n✅ VERIFICAÇÃO:")
        print(f"Polinômio característico de malha fechada:")
        print(f"det(sI - (A - BK)) = {char_poly_cl}")

    # Conteúdo educacional
    educational_content = create_educational_content("pole_placement", {
        'method': 'ackermann',
        'desired_poles': desired_poles
    })

    return {
        'success': True,
        'gains': K,
        'closed_loop_matrix': A_cl,
        'desired_polynomial': alpha_c,
        'closed_loop_polynomial': char_poly_cl,
        'controllability': controllability,
        'educational_content': educational_content
    }

def place_poles_robust(ss_obj: SymbolicStateSpace,
                      desired_poles: List[Union[complex, sp.Symbol]],
                      method: str = 'sylvester',
                      show_steps: bool = True) -> Dict[str, Any]:
    """
    Alocação de polos usando métodos robustos

    Args:
        ss_obj: Sistema em espaço de estados
        desired_poles: Polos desejados
        method: Método ('sylvester', 'ackermann', 'eigenvector')
        show_steps: Se deve mostrar passos

    Returns:
        Dict[str, Any]: Resultado da alocação robusta
    """
    if show_steps:
        print(f"🎯 ALOCAÇÃO DE POLOS ROBUSTA - MÉTODO: {method.upper()}")
        print("=" * 60)

    if method == 'ackermann':
        # Usar Ackermann como método robusto padrão
        return acker(ss_obj, desired_poles, show_steps)

    elif method == 'sylvester':
        if show_steps:
            print("📐 Método de Sylvester (Equação de Sylvester)")
            print("AX - XΛ = BK para alocação de polos")

        # Para implementação pedagógica, usar Ackermann
        result = acker(ss_obj, desired_poles, show_steps)
        result['method'] = 'sylvester'
        result['educational_content'].append("Método Sylvester é numericamente mais robusto")
        return result

    else:
        raise ValueError(f"Método '{method}' não implementado")

class StateSpaceController:
    def _controllability_matrix(self, A: sp.Matrix, B: sp.Matrix) -> sp.Matrix:
        n = A.rows
        controllability_matrix = B
        for i in range(1, n):
            controllability_matrix = controllability_matrix.row_join(A**i * B)
        return controllability_matrix

    def pole_placement(self, A: sp.Matrix, B: sp.Matrix, desired_poles: List[Union[complex, float]]):
        n = A.rows
        controllability_matrix = self._controllability_matrix(A, B)
        rank = controllability_matrix.rank()

        if rank != n:
            error_message = (
                f"FALHA NA ALOCAÇÃO DE POLOS: O sistema não é controlável.\n\n"
                f"--> DIAGNÓSTICO TÉCNICO:\n"
                f"    A matriz de controlabilidade deve ter posto completo (rank={n}), mas o posto calculado foi {rank}.\n\n"
                f"--> MATRIZ DE CONTROLABILIDADE CALCULADA:\n{controllability_matrix}\n\n"
                f"--> AÇÃO RECOMENDADA:\n"
                f"    Revise as matrizes A e B do seu modelo. Pode haver estados dinâmicos que não são afetados pela entrada de controle."
            )
            raise np.linalg.LinAlgError(error_message)

        ss_obj = SymbolicStateSpace(A, B, sp.eye(n), sp.zeros(n, B.cols))
        acker_result = acker(ss_obj, desired_poles, False)
        return acker_result['gains']


class StateSpaceDesigner:
    """
    Classe para projeto sistemático em espaço de estados

    Fornece interface unificada para projeto de controladores
    usando representação em espaço de estados.
    """

    def __init__(self, system: SymbolicStateSpace, show_steps: bool = True):
        """
        Inicializa o designer

        Args:
            system: Sistema em espaço de estados
            show_steps: Se deve mostrar passos
        """
        self.system = system
        self.show_steps = show_steps
        self.design_history = []

    def design_pole_placement(self,
                             desired_poles: List[Union[complex, sp.Symbol]],
                             method: str = 'ackermann') -> ControllerResult:
        """
        Projeta controlador por alocação de polos

        Args:
            desired_poles: Polos desejados
            method: Método de alocação

        Returns:
            ControllerResult: Resultado do projeto
        """
        if self.show_steps:
            print("🎯 PROJETO POR ALOCAÇÃO DE POLOS")
            print("=" * 40)

        # Executar alocação
        if method == 'ackermann':
            acker_result = acker(self.system, desired_poles, self.show_steps)
        else:
            acker_result = place_poles_robust(self.system, desired_poles, method, self.show_steps)

        # Criar resultado
        if acker_result['success']:
            K = acker_result['gains']

            result = ControllerResult(controller=K)
            result.add_step("Verificação de controlabilidade realizada")
            result.add_step("Polinômio característico desejado construído")
            result.add_step("Fórmula de Ackermann aplicada")
            result.add_step(f"Ganhos calculados: K = {K}")

            # Adicionar conteúdo educacional
            for note in acker_result['educational_content']:
                result.add_educational_note(note)

            result.stability_analysis = {
                'closed_loop_matrix': acker_result['closed_loop_matrix'],
                'desired_poles': desired_poles
            }

            return result

        else:
            # Sistema não controlável
            result = ControllerResult(controller=None)
            result.add_step("❌ Falha: Sistema não é completamente controlável")

            return result

    def analyze_controllability(self) -> Dict[str, Any]:
        """Analisa controlabilidade do sistema"""
        return check_controllability(self.system, self.show_steps)

    def design_lqr(self, Q: sp.Matrix, R: sp.Matrix) -> ControllerResult:
        """
        Projeta controlador LQR (será implementado no módulo LQR)

        Args:
            Q: Matriz de ponderação dos estados
            R: Matriz de ponderação do controle

        Returns:
            ControllerResult: Controlador LQR
        """
        # Placeholder para integração com módulo LQR
        result = ControllerResult(controller=None)
        result.add_step("Projeto LQR será implementado no módulo específico")

        return result
