"""
ControlLab - Projeto de Observadores
====================================

Este módulo implementa métodos de projeto de observadores de estado:
- Verificação de observabilidade
- Projeto de observadores usando a dualidade
- Observador de Luenberger
- Demonstração da dualidade controlador-observador

Características:
- Demonstração explícita da dualidade
- Explicações step-by-step
- Projeto sistemático de observadores
- Análise de convergência
"""

import sympy as sp
import numpy as np
from typing import List, Optional, Union, Tuple, Dict, Any
from ..core.symbolic_ss import SymbolicStateSpace
from .design_utils import ControllerResult, create_educational_content

def check_observability(ss_obj: SymbolicStateSpace,
                       show_steps: bool = True) -> Dict[str, Any]:
    """
    Verifica se o sistema é completamente observável

    Args:
        ss_obj: Sistema em espaço de estados
        show_steps: Se deve mostrar passos detalhados

    Returns:
        Dict[str, Any]: Resultado da análise de observabilidade
    """
    if show_steps:
        print("🔍 VERIFICAÇÃO DE OBSERVABILIDADE")
        print("=" * 40)
        print(f"🏭 Sistema: ẋ = Ax + Bu, y = Cx + Du")
        print(f"📐 A = {ss_obj.A}")
        print(f"📐 C = {ss_obj.C}")

    A = ss_obj.A
    C = ss_obj.C
    n = A.rows  # Ordem do sistema

    # Construir matriz de observabilidade Wo = [C; CA; CA²; ...; CA^(n-1)]
    if show_steps:
        print("\n📋 CONSTRUÇÃO DA MATRIZ DE OBSERVABILIDADE")
        print("=" * 50)
        print("Wo = [C; CA; CA²; ...; CA^(n-1)]")

    Wo_blocks = [C]
    current_block = C

    for i in range(1, n):
        current_block = current_block * A
        Wo_blocks.append(current_block)

        if show_steps:
            print(f"CA^{i} = {current_block}")

    # Concatenar verticalmente
    Wo = sp.Matrix.vstack(*Wo_blocks)

    if show_steps:
        print(f"\n📊 Matriz de Observabilidade:")
        print(f"Wo = {Wo}")

    # Calcular determinante e rank
    det_Wo = Wo.det()
    rank_Wo = Wo.rank()

    # Sistema é observável se rank(Wo) = n
    is_observable = rank_Wo == n

    if show_steps:
        print(f"\n✅ ANÁLISE DE OBSERVABILIDADE:")
        print(f"📐 Determinante: det(Wo) = {det_Wo}")
        print(f"📊 Rank: rank(Wo) = {rank_Wo}")
        print(f"📏 Ordem do sistema: n = {n}")

        if is_observable:
            print("✅ Sistema COMPLETAMENTE OBSERVÁVEL")
            print("💡 Todos os estados podem ser estimados")
        else:
            print("❌ Sistema NÃO completamente observável")
            print(f"⚠️ Apenas {rank_Wo} de {n} estados são observáveis")

    # Conteúdo educacional
    educational_content = [
        "🎓 CONCEITO DE OBSERVABILIDADE:",
        "• Um sistema é observável se todos os estados podem ser",
        "  determinados a partir das saídas e entradas medidas",
        "• Critério: rank(Wo) = n (ordem do sistema)",
        "• Matriz Wo = [C; CA; CA²; ...; CA^(n-1)]",
        "• Dual da controlabilidade: Wo = (Wc)ᵀ para sistema dual"
    ]

    return {
        'is_observable': is_observable,
        'observability_matrix': Wo,
        'rank': rank_Wo,
        'determinant': det_Wo,
        'system_order': n,
        'educational_content': educational_content
    }

def acker_observer(ss_obj: SymbolicStateSpace,
                  desired_poles: List[Union[complex, sp.Symbol]],
                  show_steps: bool = True) -> Dict[str, Any]:
    """
    Projeta observador usando Fórmula de Ackermann via dualidade

    Args:
        ss_obj: Sistema em espaço de estados
        desired_poles: Polos desejados para o observador
        show_steps: Se deve mostrar passos detalhados

    Returns:
        Dict[str, Any]: Ganhos do observador e análise
    """
    if show_steps:
        print("🎯 PROJETO DE OBSERVADOR - FÓRMULA DE ACKERMANN")
        print("=" * 55)
        print(f"🏭 Sistema: ẋ = Ax + Bu, y = Cx")
        print(f"🎯 Polos desejados para observador: {desired_poles}")

    A = ss_obj.A
    C = ss_obj.C
    n = A.rows

    # Verificar observabilidade primeiro
    observability = check_observability(ss_obj, show_steps=False)

    if not observability['is_observable']:
        error_message = (
            f"FALHA NO PROJETO DO OBSERVADOR: O sistema não é observável.\n\n"
            f"--> DIAGNÓSTICO TÉCNICO:\n"
            f"    A matriz de observabilidade deve ter posto completo (rank={n}), mas o posto calculado foi {observability['rank']}.\n\n"
            f"--> MATRIZ DE OBSERVABILIDADE CALCULADA:\n{observability['observability_matrix']}\n\n"
            f"--> AÇÃO RECOMENDADA:\n"
            f"    Revise as matrizes A e C do seu modelo. Pode haver estados que não afetam a saída."
        )
        raise ValueError(error_message)

    # DEMONSTRAÇÃO DA DUALIDADE
    if show_steps:
        print(f"\n🎓 APLICAÇÃO DO PRINCÍPIO DA DUALIDADE")
        print("=" * 50)
        print("📚 TEORIA DA DUALIDADE:")
        print("• Observador é dual do controlador")
        print("• Se (A,B) é controlável ⟺ (Aᵀ,Cᵀ) é observável")
        print("• Projeto: usar Ackermann no sistema dual")
        print("\n🔄 SISTEMA DUAL:")
        print("ẋd = Aᵀxd + Cᵀud")

    # Criar sistema dual
    A_dual = A.T  # Aᵀ
    B_dual = C.T  # Cᵀ (B do sistema dual)

    if show_steps:
        print(f"📐 A_dual = Aᵀ = {A_dual}")
        print(f"📐 B_dual = Cᵀ = {B_dual}")

    # Criar sistema dual para usar Ackermann
    from ..core.symbolic_ss import SymbolicStateSpace
    dual_system = SymbolicStateSpace(A_dual, B_dual, sp.eye(n), sp.zeros(n, 1))

    # Aplicar Ackermann no sistema dual
    if show_steps:
        print(f"\n🔧 APLICANDO ACKERMANN NO SISTEMA DUAL:")
        print("=" * 45)

    from .pole_placement import acker
    acker_result = acker(dual_system, desired_poles, show_steps)

    if not acker_result['success']:
        return acker_result

    # Ganhos do observador são a transposta dos ganhos do controlador dual
    L = acker_result['gains'].T

    if show_steps:
        print(f"\n✅ GANHOS DO OBSERVADOR:")
        print(f"L = Kᵀ = {L}")
        print(f"\n🔄 Observador de Luenberger:")
        print(f"ẋ̂ = Ax̂ + Bu + L(y - Cx̂)")
        print(f"ẋ̂ = (A - LC)x̂ + Bu + Ly")
        print(f"\n📐 Matriz do observador:")
        print(f"A - LC = {A - L * C}")

    # Verificar polos do observador
    A_obs = A - L * C
    char_poly_obs = A_obs.charpoly('s')

    if show_steps:
        print(f"\n✅ VERIFICAÇÃO:")
        print(f"Polinômio característico do observador:")
        print(f"det(sI - (A - LC)) = {char_poly_obs}")

        print(f"\n🎓 DINÂMICA DO ERRO DE ESTIMAÇÃO:")
        print(f"e = x - x̂ (erro de estimação)")
        print(f"ė = (A - LC)e")
        print(f"Erro converge se polos de (A - LC) são estáveis")

    # Conteúdo educacional específico para observadores
    educational_content = [
        "🎓 OBSERVADOR DE LUENBERGER:",
        "• Estima estados não medidos do sistema",
        "• ẋ̂ = Ax̂ + Bu + L(y - Cx̂)",
        "• L são os ganhos do observador",
        "• Dinâmica do erro: ė = (A - LC)e",
        "",
        "🎓 PRINCÍPIO DA DUALIDADE:",
        "• Observador é dual do controlador",
        "• Mesmo método (Ackermann) aplicado ao sistema dual",
        "• L = K_dual^T onde K_dual é ganho do sistema (A^T, C^T)",
        "",
        "🎓 SEPARAÇÃO:",
        "• Observador pode ser projetado independentemente",
        "• Polos do observador devem ser mais rápidos que controlador",
        "• Regra prática: polos 3-5 vezes mais rápidos"
    ]

    return {
        'success': True,
        'observer_gains': L,
        'observer_matrix': A_obs,
        'desired_polynomial': acker_result['desired_polynomial'],
        'observer_polynomial': char_poly_obs,
        'dual_system_result': acker_result,
        'observability': observability,
        'educational_content': educational_content
    }

def design_luenberger_observer(ss_obj: SymbolicStateSpace,
                              desired_poles: List[Union[complex, sp.Symbol]],
                              show_steps: bool = True) -> ControllerResult:
    """
    Projeta observador de Luenberger completo

    Args:
        ss_obj: Sistema em espaço de estados
        desired_poles: Polos desejados para convergência do observador
        show_steps: Se deve mostrar passos

    Returns:
        ControllerResult: Observador projetado
    """
    if show_steps:
        print("🎯 PROJETO DE OBSERVADOR DE LUENBERGER")
        print("=" * 45)

    # Usar método de Ackermann via dualidade
    observer_result = acker_observer(ss_obj, desired_poles, show_steps)

    if observer_result['success']:
        L = observer_result['observer_gains']

        result = ControllerResult(controller=L)
        result.add_step("Verificação de observabilidade realizada")
        result.add_step("Princípio da dualidade aplicado")
        result.add_step("Sistema dual criado (A^T, C^T)")
        result.add_step("Fórmula de Ackermann aplicada ao sistema dual")
        result.add_step(f"Ganhos do observador: L = {L}")
        result.add_step("Observador: ẋ̂ = (A - LC)x̂ + Bu + Ly")

        # Adicionar conteúdo educacional
        for note in observer_result['educational_content']:
            result.add_educational_note(note)

        result.stability_analysis = {
            'observer_matrix': observer_result['observer_matrix'],
            'desired_poles': desired_poles
        }

        return result

    else:
        result = ControllerResult(controller=None)
        result.add_step("❌ Falha: Sistema não é completamente observável")

        return result

class ObserverDesigner:
    """
    Classe para projeto sistemático de observadores

    Fornece interface unificada para projeto de observadores
    com demonstração da dualidade.
    """

    def __init__(self, system: SymbolicStateSpace, show_steps: bool = True):
        """
        Inicializa o designer de observadores

        Args:
            system: Sistema em espaço de estados
            show_steps: Se deve mostrar passos
        """
        self.system = system
        self.show_steps = show_steps
        self.design_history = []

    def design_observer(self,
                       desired_poles: List[Union[complex, sp.Symbol]],
                       method: str = 'ackermann') -> ControllerResult:
        """
        Projeta observador usando método especificado

        Args:
            desired_poles: Polos desejados para o observador
            method: Método de projeto ('ackermann', 'pole_placement')

        Returns:
            ControllerResult: Observador projetado
        """
        if method == 'ackermann':
            return design_luenberger_observer(self.system, desired_poles, self.show_steps)
        else:
            raise ValueError(f"Método '{method}' não implementado")

    def analyze_observability(self) -> Dict[str, Any]:
        """Analisa observabilidade do sistema"""
        return check_observability(self.system, self.show_steps)

    def demonstrate_duality(self, controller_poles: List, observer_poles: List) -> Dict[str, Any]:
        """
        Demonstra a dualidade entre controlador e observador

        Args:
            controller_poles: Polos do controlador
            observer_poles: Polos do observador

        Returns:
            Dict[str, Any]: Demonstração da dualidade
        """
        if self.show_steps:
            print("🎓 DEMONSTRAÇÃO DA DUALIDADE CONTROLADOR-OBSERVADOR")
            print("=" * 60)

        # Verificar controlabilidade
        from .pole_placement import check_controllability
        controllability = check_controllability(self.system, show_steps=False)

        # Verificar observabilidade
        observability = check_observability(self.system, show_steps=False)

        if self.show_steps:
            print(f"✅ Sistema controlável: {controllability['is_controllable']}")
            print(f"✅ Sistema observável: {observability['is_observable']}")

            print(f"\n🔄 SISTEMA ORIGINAL:")
            print(f"A = {self.system.A}")
            print(f"B = {self.system.B}")
            print(f"C = {self.system.C}")

            print(f"\n🔄 SISTEMA DUAL:")
            print(f"A_dual = A^T = {self.system.A.T}")
            print(f"B_dual = C^T = {self.system.C.T}")
            print(f"C_dual = B^T = {self.system.B.T}")

            print(f"\n📊 MATRIZES DE CONTROLABILIDADE E OBSERVABILIDADE:")
            print(f"Wc (controlabilidade) = {controllability['controllability_matrix']}")
            print(f"Wo (observabilidade) = {observability['observability_matrix']}")
            print(f"Relação: Wo = (Wc_dual)^T")

        return {
            'controllability': controllability,
            'observability': observability,
            'dual_A': self.system.A.T,
            'dual_B': self.system.C.T,
            'dual_C': self.system.B.T,
            'duality_verified': controllability['is_controllable'] == observability['is_observable']
        }
