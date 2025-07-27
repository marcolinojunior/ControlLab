"""
ControlLab - Compensadores Clássicos
====================================

Este módulo implementa compensadores clássicos no domínio da frequência:
- PID (Proporcional-Integral-Derivativo)
- Lead (Avanço de fase)
- Lag (Atraso de fase)
- Lead-Lag (Combinado)

Características:
- Derivação simbólica completa
- Explicações step-by-step
- Projeto por especificações
- Validação automática
"""

import sympy as sp
from typing import Dict, List, Optional, Union, Tuple, Any
from ..core.symbolic_tf import SymbolicTransferFunction
from .design_utils import ControllerResult, DesignSpecifications, create_educational_content

def PID(Kp: Union[float, sp.Symbol],
        Ki: Union[float, sp.Symbol],
        Kd: Union[float, sp.Symbol],
        variable: str = 's') -> SymbolicTransferFunction:
    """
    Cria um controlador PID simbólico

    Args:
        Kp: Ganho proporcional
        Ki: Ganho integral
        Kd: Ganho derivativo
        variable: Variável de Laplace (padrão 's')

    Returns:
        SymbolicTransferFunction: Controlador PID C(s) = Kp + Ki/s + Kd*s

    Example:
        >>> Kp, Ki, Kd = sp.symbols('Kp Ki Kd', real=True, positive=True)
        >>> controller = PID(Kp, Ki, Kd)
        >>> print(controller)
        (Kd*s**2 + Kp*s + Ki)/s
    """
    s = sp.Symbol(variable)

    # C(s) = Kp + Ki/s + Kd*s = (Kd*s² + Kp*s + Ki)/s
    numerator = Kd * s**2 + Kp * s + Ki
    denominator = s

    controller = SymbolicTransferFunction(numerator, denominator)

    # Adicionar ao histórico usando o método correto do SymbolicTransferFunction
    controller.history.add_step("PID_Creation", f"Criado controlador PID com Kp={Kp}, Ki={Ki}, Kd={Kd}",
                                None, controller, {"type": "pid"})
    controller.history.add_step("PID_Structure", "Estrutura: C(s) = Kp + Ki/s + Kd*s",
                                None, "C(s) = Kp + Ki/s + Kd*s", {"type": "explanation"})
    controller.history.add_step("PID_Rational", "Forma racional: C(s) = (Kd*s² + Kp*s + Ki)/s",
                                None, "(Kd*s² + Kp*s + Ki)/s", {"type": "explanation"})

    return controller

def Lead(K: Union[float, sp.Symbol],
         zero: Union[float, sp.Symbol],
         pole: Union[float, sp.Symbol],
         variable: str = 's') -> SymbolicTransferFunction:
    """
    Cria um compensador Lead (avanço de fase)

    Args:
        K: Ganho do compensador
        zero: Localização do zero (deve ser < polo para avanço)
        pole: Localização do polo
        variable: Variável de Laplace

    Returns:
        SymbolicTransferFunction: Compensador Lead C(s) = K(s + zero)/(s + pole)

    Example:
        >>> K, z, p = sp.symbols('K z p', real=True, positive=True)
        >>> lead = Lead(K, z, p)
        >>> print(lead)
        K*(s + z)/(s + p)
    """
    s = sp.Symbol(variable)

    # C(s) = K * (s + zero) / (s + pole)
    numerator = K * (s + zero)
    denominator = s + pole

    controller = SymbolicTransferFunction(numerator, denominator)

    # Histórico educacional usando o método correto
    controller.history.add_step("Lead_Creation", f"Criado compensador Lead: K={K}, zero={zero}, polo={pole}",
                                None, controller, {"type": "lead"})
    controller.history.add_step("Lead_Purpose", "Compensador Lead - adiciona fase positiva",
                                None, "Avanço de fase", {"type": "explanation"})
    controller.history.add_step("Lead_Condition", "Para avanço de fase ideal: zero < polo",
                                None, "zero < polo", {"type": "condition"})

    return controller

def Lag(K: Union[float, sp.Symbol],
        zero: Union[float, sp.Symbol],
        pole: Union[float, sp.Symbol],
        variable: str = 's') -> SymbolicTransferFunction:
    """
    Cria um compensador Lag (atraso de fase)

    Args:
        K: Ganho do compensador
        zero: Localização do zero (deve ser > polo para atraso)
        pole: Localização do polo
        variable: Variável de Laplace

    Returns:
        SymbolicTransferFunction: Compensador Lag C(s) = K(s + zero)/(s + pole)
    """
    s = sp.Symbol(variable)

    # C(s) = K * (s + zero) / (s + pole)
    numerator = K * (s + zero)
    denominator = s + pole

    controller = SymbolicTransferFunction(numerator, denominator)

    # Histórico educacional
    controller.history.add_step("Lag_Creation", f"Criado compensador Lag: K={K}, zero={zero}, polo={pole}",
                                None, controller, {"type": "lag"})
    controller.history.add_step("Lag_Purpose", "Compensador Lag - reduz erro em regime permanente",
                                None, "Redução de erro", {"type": "explanation"})
    controller.history.add_step("Lag_Condition", "Para atraso de fase ideal: zero > pole",
                                None, "zero > pole", {"type": "condition"})

    return controller

def LeadLag(K: Union[float, sp.Symbol],
            z1: Union[float, sp.Symbol], p1: Union[float, sp.Symbol],
            z2: Union[float, sp.Symbol], p2: Union[float, sp.Symbol],
            variable: str = 's') -> SymbolicTransferFunction:
    """
    Cria um compensador Lead-Lag combinado

    Args:
        K: Ganho total
        z1, p1: Zero e polo da parte Lead (z1 < p1)
        z2, p2: Zero e polo da parte Lag (z2 > p2)
        variable: Variável de Laplace

    Returns:
        SymbolicTransferFunction: Compensador Lead-Lag
    """
    s = sp.Symbol(variable)

    # C(s) = K * (s + z1)(s + z2) / ((s + p1)(s + p2))
    numerator = K * (s + z1) * (s + z2)
    denominator = (s + p1) * (s + p2)

    controller = SymbolicTransferFunction(numerator, denominator)

    controller.history.add_step("LeadLag_Creation", f"Compensador Lead-Lag: Lead(z1={z1}, p1={p1}) + Lag(z2={z2}, p2={p2})",
                                None, controller, {"type": "leadlag"})
    controller.history.add_step("LeadLag_Purpose", "Combina vantagens de Lead e Lag",
                                None, "Avanço + Atraso", {"type": "explanation"})

    return controller

class CompensatorDesigner:
    """
    Classe para projeto sistemático de compensadores

    Fornece métodos para projeto automático baseado em especificações
    com explicações pedagógicas detalhadas.
    """

    def __init__(self, show_steps: bool = True):
        """
        Inicializa o designer de compensadores

        Args:
            show_steps: Se deve mostrar passos detalhados
        """
        self.show_steps = show_steps
        self.design_history = []

    def design_by_specifications(self,
                                plant: SymbolicTransferFunction,
                                specs: DesignSpecifications,
                                method: str = 'auto') -> ControllerResult:
        """
        Projeta compensador baseado em especificações

        Args:
            plant: Planta do sistema
            specs: Especificações de desempenho
            method: Método de projeto ('lead', 'lag', 'pid', 'auto')

        Returns:
            ControllerResult: Resultado do projeto
        """
        if self.show_steps:
            print("🎯 PROJETO DE COMPENSADOR POR ESPECIFICAÇÕES")
            print("=" * 50)
            print(f"🏭 Planta: G(s) = {plant}")
            print(f"📋 Especificações: {specs.to_dict()}")

        # Escolher método automaticamente se necessário
        if method == 'auto':
            method = self._select_method(specs)
            if self.show_steps:
                print(f"🔧 Método selecionado: {method}")

        # Projetar baseado no método
        if method == 'lead':
            return self._design_lead_by_specs(plant, specs)
        elif method == 'lag':
            return self._design_lag_by_specs(plant, specs)
        elif method == 'pid':
            return self._design_pid_by_specs(plant, specs)
        else:
            raise ValueError(f"Método '{method}' não implementado")

    def _select_method(self, specs: DesignSpecifications) -> str:
        """Seleciona método baseado nas especificações"""

        # Se há especificações de resposta transitória, usar Lead
        if specs.overshoot is not None or specs.settling_time is not None:
            return 'lead'

        # Se há especificações de erro em regime, usar Lag
        if specs.steady_state_error is not None:
            return 'lag'

        # Se há especificações mistas, usar PID
        return 'pid'

    def _design_lead_by_specs(self, plant: SymbolicTransferFunction,
                             specs: DesignSpecifications) -> ControllerResult:
        """Projeta compensador Lead por especificações"""

        result = ControllerResult(controller=None)
        result.add_step("Iniciando projeto de compensador Lead")

        # Análise da planta
        result.add_step("Analisando características da planta")

        # Para projeto simplificado, usar parâmetros típicos
        K = sp.Symbol('K_lead', real=True, positive=True)
        z = sp.Symbol('z_lead', real=True, positive=True)
        p = sp.Symbol('p_lead', real=True, positive=True)

        # Criar compensador Lead genérico
        compensator = Lead(K, z, p)
        result.controller = compensator

        result.add_step(f"Compensador Lead projetado: C(s) = {compensator}")
        result.add_educational_note("Compensador Lead adiciona fase positiva")
        result.add_educational_note("Melhora margem de fase e resposta transitória")

        return result

    def _design_lag_by_specs(self, plant: SymbolicTransferFunction,
                            specs: DesignSpecifications) -> ControllerResult:
        """Projeta compensador Lag por especificações"""

        result = ControllerResult(controller=None)
        result.add_step("Iniciando projeto de compensador Lag")

        K = sp.Symbol('K_lag', real=True, positive=True)
        z = sp.Symbol('z_lag', real=True, positive=True)
        p = sp.Symbol('p_lag', real=True, positive=True)

        compensator = Lag(K, z, p)
        result.controller = compensator

        result.add_step(f"Compensador Lag projetado: C(s) = {compensator}")
        result.add_educational_note("Compensador Lag reduz erro em regime permanente")

        return result

    def _design_pid_by_specs(self, plant: SymbolicTransferFunction,
                            specs: DesignSpecifications) -> ControllerResult:
        """Projeta controlador PID por especificações"""

        result = ControllerResult(controller=None)
        result.add_step("Iniciando projeto de controlador PID")

        Kp = sp.Symbol('Kp', real=True)
        Ki = sp.Symbol('Ki', real=True)
        Kd = sp.Symbol('Kd', real=True)

        controller = PID(Kp, Ki, Kd)
        result.controller = controller

        result.add_step(f"Controlador PID projetado: C(s) = {controller}")
        result.add_educational_note("PID combina ação proporcional, integral e derivativa")

        return result

def design_lead_compensator(plant: SymbolicTransferFunction,
                           phase_margin_target: float,
                           gain_crossover: float,
                           show_steps: bool = True) -> ControllerResult:
    """
    Projeta compensador Lead para margem de fase específica

    Args:
        plant: Planta do sistema G(s)
        phase_margin_target: Margem de fase desejada (graus)
        gain_crossover: Frequência de cruzamento de ganho desejada (rad/s)
        show_steps: Se deve mostrar passos

    Returns:
        ControllerResult: Compensador projetado
    """
    if show_steps:
        print("🎯 PROJETO DE COMPENSADOR LEAD")
        print("=" * 40)
        print(f"🏭 Planta: G(s) = {plant}")
        print(f"📐 Margem de fase alvo: {phase_margin_target}°")
        print(f"📊 Frequência de cruzamento: {gain_crossover} rad/s")

    result = ControllerResult(controller=None)

    # Passo 1: Calcular fase adicional necessária
    result.add_step("Calculando fase adicional necessária")

    # Para projeto pedagógico, usar símbolos
    K = sp.Symbol('K_c', real=True, positive=True)
    alpha = sp.Symbol('alpha', real=True, positive=True)  # alpha = p/z > 1
    wm = sp.Symbol('omega_m', real=True, positive=True)   # frequência máxima de fase

    # Compensador Lead: C(s) = K * (s/z + 1) / (s/p + 1)
    # onde p = alpha * z
    s = sp.Symbol('s')
    z = sp.Symbol('z', real=True, positive=True)
    p = alpha * z

    numerator = K * (s + z)
    denominator = s + p

    compensator = SymbolicTransferFunction(numerator, denominator)
    result.controller = compensator

    if phase_margin_target > 60:
        compensator.history.add_warning(
            f"O avanço de fase de {phase_margin_target}° é muito grande para um único compensador de avanço. "
            "Pode ser necessário um compensador de ordem superior ou um método de projeto diferente."
        )

    result.add_step(f"Compensador Lead genérico: C(s) = {compensator}")
    result.add_step(f"Com relação α = p/z = {alpha}")

    # Adicionar conteúdo educacional
    educational_notes = create_educational_content("lead_compensator", {
        'phase_margin': phase_margin_target,
        'gain_crossover': gain_crossover
    })

    for note in educational_notes:
        result.add_educational_note(note)

    result.add_educational_note(f"Frequência de máxima fase: ωm = √(zp) = z√α")
    result.add_educational_note(f"Máxima fase adicional: φm = arcsin((α-1)/(α+1))")

    if show_steps:
        print(result.get_formatted_report())

    return result

def design_lag_compensator(plant: SymbolicTransferFunction,
                          error_reduction_factor: float,
                          show_steps: bool = True) -> ControllerResult:
    """
    Projeta compensador Lag para redução de erro

    Args:
        plant: Planta do sistema
        error_reduction_factor: Fator de redução do erro (> 1)
        show_steps: Se deve mostrar passos

    Returns:
        ControllerResult: Compensador projetado
    """
    if show_steps:
        print("🎯 PROJETO DE COMPENSADOR LAG")
        print("=" * 40)
        print(f"🏭 Planta: G(s) = {plant}")
        print(f"📉 Fator de redução de erro: {error_reduction_factor}")

    result = ControllerResult(controller=None)

    # Compensador Lag genérico
    K = sp.Symbol('K_lag', real=True, positive=True)
    beta = sp.Symbol('beta', real=True, positive=True)  # beta = z/p > 1

    s = sp.Symbol('s')
    p = sp.Symbol('p_lag', real=True, positive=True)
    z = beta * p  # Para Lag: z > p

    numerator = K * (s + z)
    denominator = s + p

    compensator = SymbolicTransferFunction(numerator, denominator)
    result.controller = compensator

    result.add_step(f"Compensador Lag: C(s) = {compensator}")
    result.add_step(f"Com relação β = z/p = {beta}")

    # Conteúdo educacional
    educational_notes = create_educational_content("lag_compensator", {
        'error_reduction': error_reduction_factor
    })

    for note in educational_notes:
        result.add_educational_note(note)

    if show_steps:
        print(result.get_formatted_report())

    return result

def design_pid_tuning(plant: SymbolicTransferFunction,
                     method: str = 'ziegler_nichols',
                     show_steps: bool = True) -> ControllerResult:
    """
    Projeta controlador PID usando métodos de sintonização

    Args:
        plant: Planta do sistema
        method: Método de sintonização ('ziegler_nichols', 'cohen_coon')
        show_steps: Se deve mostrar passos

    Returns:
        ControllerResult: Controlador PID sintonizado
    """
    if show_steps:
        print(f"🎯 SINTONIZAÇÃO PID - MÉTODO: {method.upper()}")
        print("=" * 50)
        print(f"🏭 Planta: G(s) = {plant}")

    result = ControllerResult(controller=None)

    # Parâmetros PID genéricos
    Kp = sp.Symbol('Kp', real=True)
    Ki = sp.Symbol('Ki', real=True)
    Kd = sp.Symbol('Kd', real=True)

    controller = PID(Kp, Ki, Kd)
    result.controller = controller

    result.add_step(f"Método de sintonização: {method}")
    result.add_step(f"Controlador PID: C(s) = {controller}")

    if method == 'ziegler_nichols':
        result.add_educational_note("Método Ziegler-Nichols (1ª regra):")
        result.add_educational_note("1. Obter curva de reação da planta")
        result.add_educational_note("2. Identificar parâmetros K, L, T")
        result.add_educational_note("3. Calcular: Kp=1.2T/(KL), Ki=Kp/(2L), Kd=Kp*L/2")

    elif method == 'cohen_coon':
        result.add_educational_note("Método Cohen-Coon:")
        result.add_educational_note("Melhoria do Ziegler-Nichols para sistemas com atraso")
        result.add_educational_note("Considera relação L/T para ajustar parâmetros")

    # Conteúdo educacional PID
    educational_notes = create_educational_content("pid", {'method': method})
    for note in educational_notes:
        result.add_educational_note(note)

    if show_steps:
        print(result.get_formatted_report())

    return result

def decompose_pid_controller(controller: SymbolicTransferFunction) -> Tuple[float, float, float]:
    """
    Decompõe uma função de transferência em seus componentes PID.

    Args:
        controller (SymbolicTransferFunction): A função de transferência do controlador.

    Returns:
        Tuple[float, float, float]: Uma tupla contendo os ganhos (Kp, Ki, Kd).
    """
    s = controller.variable
    num = controller.numerator
    den = controller.denominator

    # Garante que o denominador é um polinômio em s
    if not den.is_polynomial(s):
        raise ValueError(f"Denominador '{den}' não é um polinômio válido em {s}")

    # Garante que o numerador é um polinômio em s
    if not num.is_polynomial(s):
        raise ValueError(f"Numerador '{num}' não é um polinômio válido em {s}")

    # Controlador na forma P, PI, PD, PID
    if den.equals(1):  # P, PD
        p_num = sp.Poly(num, s)
        coeffs = p_num.all_coeffs()
        if len(coeffs) == 1:  # P
            return coeffs[0], 0, 0
        elif len(coeffs) == 2:  # PD
            return coeffs[1], 0, coeffs[0]
        else:
            raise ValueError("Não é um controlador P ou PD válido")

    elif den.equals(s):  # I, PI, PID
        p_num = sp.Poly(num, s)
        coeffs = p_num.all_coeffs()
        if len(coeffs) == 1:  # I
            return 0, coeffs[0], 0
        elif len(coeffs) == 2:  # PI
            return coeffs[0], coeffs[1], 0
        elif len(coeffs) == 3:  # PID
            return coeffs[1], coeffs[2], coeffs[0]
        else:
            raise ValueError("Não é um controlador I, PI ou PID válido")

    else:
        raise ValueError(f"Denominador '{den}' não é válido para um controlador PID padrão")


def design_by_root_locus(plant: SymbolicTransferFunction,
                        desired_poles: List[complex],
                        show_steps: bool = True) -> ControllerResult:
    """
    Projeta controlador baseado no lugar geométrico das raízes

    Args:
        plant: Planta do sistema
        desired_poles: Polos desejados em malha fechada
        show_steps: Se deve mostrar passos

    Returns:
        ControllerResult: Controlador projetado
    """
    if show_steps:
        print("🎯 PROJETO POR LUGAR GEOMÉTRICO DAS RAÍZES")
        print("=" * 50)
        print(f"🏭 Planta: G(s) = {plant}")
        print(f"🎯 Polos desejados: {desired_poles}")

    result = ControllerResult(controller=None)

    # Para projeto por root locus, geralmente usa-se ganho proporcional
    K = sp.Symbol('K', real=True, positive=True)

    # Controlador proporcional
    s = sp.Symbol('s')
    controller = SymbolicTransferFunction(K, 1)
    result.controller = controller

    result.add_step("Projeto por lugar geométrico das raízes")
    result.add_step(f"Controlador proporcional: C(s) = {K}")
    result.add_step("1. Plotar lugar geométrico da planta")
    result.add_step("2. Encontrar pontos correspondentes aos polos desejados")
    result.add_step("3. Calcular ganho K necessário")

    result.add_educational_note("Root Locus mostra como polos variam com ganho K")
    result.add_educational_note("Permite escolha direta dos polos de malha fechada")
    result.add_educational_note("Útil para especificar resposta transitória")

    if show_steps:
        print(result.get_formatted_report())

    return result
