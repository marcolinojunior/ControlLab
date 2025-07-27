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

    # Se o sistema já é de 2ª ordem ou inferior, a aproximação é válida por definição.
    if len(poles) <= 2:
        if not zeros:
            return True, "O sistema é de segunda ordem sem zeros. A aproximação é válida.", poles
        else:
            # Se há zeros, a aproximação para um sistema *padrão* de 2ª ordem é inválida.
            return False, "Aproximação de 2ª ordem não confiável: O sistema é de 2ª ordem mas possui zeros que alteram a resposta padrão.", poles

    # É crucial converter os polos para valores numéricos para comparação.
    # Usamos try-except para lidar com polos que ainda podem ser simbólicos.
    try:
        poles_eval = [complex(p.evalf()) for p in poles]
        zeros_eval = [complex(z.evalf()) for z in zeros]
    except Exception:
        # Se .evalf() falhar, significa que o polo ainda é simbólico (ex: contém 'K')
        # Nesse caso, a análise de dominância numérica não é possível.
        return False, "Análise de dominância não aplicável: O sistema contém polos/zeros simbólicos.", None

    # Ordena os polos pela magnitude de sua parte real (os mais lentos primeiro)
    sorted_poles = sorted(poles_eval, key=lambda p: abs(p.real))

    dominant_poles = sorted_poles[:2]
    non_dominant_poles = sorted_poles[2:]

    # A parte real do polo dominante (quão lento ele é)
    # Usamos o segundo polo da lista, pois o primeiro pode ser um polo em s=0 (integrador)
    # que não tem uma dinâmica de decaimento típica.
    if len(dominant_poles) > 1:
        real_part_dominant = abs(dominant_poles[1].real)
    else: # Caso de apenas um polo dominante (e outros não dominantes)
        real_part_dominant = abs(dominant_poles[0].real)

    # Se o polo dominante está no eixo imaginário, qualquer outro polo o torna não-dominante
    if np.isclose(real_part_dominant, 0):
        return False, "Aproximação não confiável: Os polos dominantes estão no eixo imaginário.", poles

    # REGRA 1: Os outros polos devem ser pelo menos 'dominance_factor' vezes mais rápidos.
    for pole in non_dominant_poles:
        real_part_other = abs(pole.real)
        if real_part_other < dominance_factor * real_part_dominant:
            warning_msg = (
                f"Aproximação não confiável: O polo em s={pole:.2f} (decaimento em {real_part_other:.2f}) "
                f"não é pelo menos {dominance_factor}x mais rápido "
                f"que o polo dominante (decaimento em {real_part_dominant:.2f})."
            )
            return False, warning_msg, poles

    # REGRA 2: Não deve haver zeros mais lentos que os polos dominantes.
    for zero in zeros_eval:
        real_part_zero = abs(zero.real)
        if real_part_zero < dominance_factor * real_part_dominant:
            warning_msg = (
                f"Aproximação não confiável: O zero em s={zero:.2f} (parte real {real_part_zero:.2f}) "
                f"é mais lento que os polos dominantes e irá dominar a resposta."
            )
            return False, warning_msg, poles

    return True, "Aproximação de 2ª ordem é considerada válida.", poles
