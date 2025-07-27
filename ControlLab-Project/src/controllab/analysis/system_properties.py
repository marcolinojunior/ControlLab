# DENTRO DE: src/controllab/analysis/system_properties.py

import numpy as np
import sympy as sp
from typing import List, Tuple, Optional
from ..core.symbolic_tf import SymbolicTransferFunction

def verify_second_order_approximation(
    system: SymbolicTransferFunction,
    dominance_factor: float = 5.0
) -> Tuple[bool, str, Optional[List[sp.Expr]]]:
    """
    Verifica se um sistema pode ser confiavelmente aproximado por um modelo
    de segunda ordem dominante.

    A verificação se baseia em duas regras principais:
    1. Os polos não dominantes devem estar significativamente mais distantes
       do eixo imaginário do que os polos dominantes.
    2. Não deve haver zeros próximos aos polos dominantes, pois eles podem
       cancelar o efeito dos polos.

    Args:
        system: A função de transferência do sistema em malha fechada.
        dominance_factor: Fator que define "quão mais longe" os outros polos
                          devem estar (padrão: 5.0).

    Returns:
        Uma tupla contendo:
        - bool: True se a aproximação é válida, False caso contrário.
        - str: Uma mensagem explicando o resultado (um "Warning" se for inválida).
        - List[sp.Expr]: O par de polos dominantes, se encontrado.
    """
    poles = system.poles()
    zeros = system.zeros()

    if len(poles) <= 2:
        return True, "O sistema é de segunda ordem ou inferior. A aproximação é válida por definição.", poles

    # Ordena os polos pela parte real (mais próximos do eixo jw primeiro)
    # Usamos sorted com uma chave lambda para garantir a ordenação correta
    sorted_poles = sorted(poles, key=lambda p: abs(sp.re(p).evalf()))

    # O par de polos dominante são os dois primeiros da lista ordenada
    dominant_poles = sorted_poles[:2]
    non_dominant_poles = sorted_poles[2:]

    real_part_dominant = abs(sp.re(dominant_poles[0]).evalf())

    # 1. Verifica a regra da distância dos polos
    for pole in non_dominant_poles:
        real_part_other = abs(sp.re(pole).evalf())
        if real_part_other < dominance_factor * real_part_dominant:
            warning_msg = (
                f"Aproximação de 2ª ordem não confiável: O polo não dominante em s={pole.evalf(3)} "
                f"({real_part_other:.2f}) não é pelo menos {dominance_factor}x mais distante do eixo jω "
                f"que o polo dominante ({real_part_dominant:.2f})."
            )
            return False, warning_msg, dominant_poles

    # 2. Verifica a regra da proximidade dos zeros
    for zero in zeros:
        for pole in dominant_poles:
            # Calcula a distância entre o zero e o polo dominante
            distance = abs(pole.evalf() - zero.evalf())
            # Se um zero estiver "muito perto" de um polo dominante (ex: < 25% da distância do polo à origem),
            # ele pode cancelar o efeito do polo.
            if distance < 0.25 * abs(pole.evalf()):
                warning_msg = (
                    f"Aproximação de 2ª ordem não confiável: O zero em s={zero.evalf(3)} está muito "
                    f"próximo do polo dominante em s={pole.evalf(3)}, o que pode alterar significativamente a resposta."
                )
                return False, warning_msg, dominant_poles

    return True, "Aproximação de 2ª ordem é considerada válida.", dominant_poles
