# DENTRO DE: src/controllab/analysis/system_properties.py
import sympy as sp
import numpy as np
from typing import List, Tuple, Optional
from ..core.symbolic_tf import SymbolicTransferFunction

def verify_second_order_approximation(
    system: SymbolicTransferFunction,
    dominance_factor: float = 5.0
) -> Tuple[bool, str, Optional[List[sp.Expr]]]:
    """
    Verifica de forma robusta se um sistema pode ser aproximado por um modelo
    de segunda ordem dominante, usando as heurísticas padrão da engenharia de controle.
    """
    poles = system.poles()
    zeros = system.zeros()

    try:
        # É crucial avaliar as expressões para comparação numérica
        poles_eval = [complex(p.evalf()) for p in poles]
        zeros_eval = [complex(z.evalf()) for z in zeros]
    except Exception:
        # Se a avaliação falhar, o sistema ainda tem símbolos (como 'K')
        return False, "Análise de dominância não aplicável: O sistema contém polos/zeros simbólicos.", None

    # --- LÓGICA CORRIGIDA E REFINADA ---

    # REGRA 0: Tratamento de sistemas de 2ª ordem ou menos.
    if len(poles_eval) <= 2:
        if not zeros_eval:
            # Caso ideal: 1º ou 2º ordem, sem zeros. A aproximação para um sistema padrão é perfeita.
            return True, "O sistema é de segunda ordem ou inferior e não possui zeros. A aproximação é válida.", poles
        else:
            # Caso com zeros: A presença de zeros altera a resposta padrão de 2ª ordem.
            return False, "Aproximação não confiável: O sistema é de 2ª ordem, mas possui zeros que alteram a resposta transiente padrão.", poles

    # Para sistemas de ordem superior, a análise de dominância começa.
    # Ordena os polos pela magnitude de sua parte real (os mais próximos do eixo jw primeiro).
    sorted_poles = sorted(poles_eval, key=lambda p: abs(p.real))

    dominant_poles_eval = sorted_poles[:2]
    non_dominant_poles_eval = sorted_poles[2:]

    # A parte real do polo dominante (que não seja um integrador) dita a velocidade do sistema.
    # Usamos o segundo polo da lista ordenada, pois o primeiro pode ser s=0.
    real_part_dominant = abs(dominant_poles_eval[1].real)

    if np.isclose(real_part_dominant, 0):
        return False, "Aproximação não confiável: Os polos dominantes estão no eixo imaginário.", poles

    # REGRA 1: Os outros polos devem ser significativamente mais "rápidos".
    for pole in non_dominant_poles_eval:
        real_part_other = abs(pole.real)
        if real_part_other < dominance_factor * real_part_dominant:
            msg = (
                f"Aproximação não confiável: O polo em s={pole:.2f} (decaimento em {real_part_other:.2f}) "
                f"não é pelo menos {dominance_factor}x mais rápido "
                f"que o polo dominante (decaimento em {real_part_dominant:.2f})."
            )
            return False, msg, poles

    # REGRA 2: Não deve haver zeros "lentos" que interfiram.
    # Um zero é problemático se for mais lento (mais próximo da origem) que os polos dominantes.
    for zero in zeros_eval:
        if abs(zero.real) < real_part_dominant:
            msg = (
                f"Aproximação não confiável: O zero em s={zero:.2f} é mais lento que os polos dominantes e "
                f"terá um impacto significativo na resposta, invalidando a aproximação padrão."
            )
            return False, msg, poles

    return True, "Aproximação de 2ª ordem é considerada válida.", poles
