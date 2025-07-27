"""
ControlLab - Utilitários para Projeto de Controladores
======================================================

Este módulo contém classes e funções utilitárias para o projeto de controladores,
incluindo estruturas de dados para resultados e especificações de projeto.

Características:
- Classes para organizar resultados de projeto
- Validação de estabilidade e desempenho
- Métricas de performance
- Histórico de passos de projeto
"""

import sympy as sp
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from ..core.symbolic_tf import SymbolicTransferFunction
from ..core.symbolic_ss import SymbolicStateSpace
from ..analysis.system_properties import verify_second_order_approximation

@dataclass
class DesignSpecifications:
    """
    Classe para especificações de projeto de controladores

    Atributos:
        overshoot: Sobressinal máximo permitido (%)
        settling_time: Tempo de acomodação (s)
        rise_time: Tempo de subida (s)
        steady_state_error: Erro em regime permanente
        phase_margin: Margem de fase (graus)
        gain_margin: Margem de ganho (dB)
        bandwidth: Largura de banda (rad/s)
        damping_ratio: Coeficiente de amortecimento
        natural_frequency: Frequência natural (rad/s)
    """
    overshoot: Optional[float] = None
    settling_time: Optional[float] = None
    rise_time: Optional[float] = None
    steady_state_error: Optional[float] = None
    phase_margin: Optional[float] = None
    gain_margin: Optional[float] = None
    bandwidth: Optional[float] = None
    damping_ratio: Optional[float] = None
    natural_frequency: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Converte especificações para dicionário"""
        return {k: v for k, v in self.__dict__.items() if v is not None}

    def validate(self) -> Tuple[bool, List[str]]:
        """
        Valida se as especificações são consistentes

        Returns:
            Tuple[bool, List[str]]: (válido, lista_de_erros)
        """
        errors = []

        if self.overshoot is not None and (self.overshoot < 0 or self.overshoot > 100):
            errors.append("Sobressinal deve estar entre 0% e 100%")

        if self.settling_time is not None and self.settling_time <= 0:
            errors.append("Tempo de acomodação deve ser positivo")

        if self.damping_ratio is not None and self.damping_ratio < 0:
            errors.append("Coeficiente de amortecimento deve ser não-negativo")

        if self.phase_margin is not None and (self.phase_margin < 0 or self.phase_margin >= 180):
            errors.append("Margem de fase deve estar entre 0° e 180°")

        return len(errors) == 0, errors

@dataclass
class ControllerResult:
    """
    Classe para armazenar resultados de projeto de controladores

    Atributos:
        controller: Função de transferência do controlador
        closed_loop: Sistema em malha fechada
        specifications_met: Especificações atendidas
        design_steps: Passos do projeto
        performance_metrics: Métricas de desempenho
        stability_analysis: Análise de estabilidade
        educational_content: Conteúdo educacional
    """
    controller: Union[SymbolicTransferFunction, SymbolicStateSpace, sp.Matrix]
    closed_loop: Optional[SymbolicTransferFunction] = None
    specifications_met: Dict[str, bool] = field(default_factory=dict)
    design_steps: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    stability_analysis: Dict[str, Any] = field(default_factory=dict)
    educational_content: List[str] = field(default_factory=list)

    def add_step(self, step: str):
        """Adiciona um passo ao histórico de projeto"""
        self.design_steps.append(step)

    def add_educational_note(self, note: str):
        """Adiciona uma nota educacional"""
        self.educational_content.append(note)

    def get_formatted_report(self) -> str:
        """
        Gera relatório formatado do projeto

        Returns:
            str: Relatório completo formatado
        """
        report = []
        report.append("=" * 60)
        report.append("📋 RELATÓRIO DE PROJETO DE CONTROLADOR")
        report.append("=" * 60)

        # Controlador projetado
        report.append(f"\n🎯 CONTROLADOR PROJETADO:")
        report.append(f"📐 C(s) = {self.controller}")

        # Sistema em malha fechada
        if self.closed_loop:
            report.append(f"\n🔄 SISTEMA EM MALHA FECHADA:")
            report.append(f"📐 T(s) = {self.closed_loop}")

        # Especificações atendidas
        if self.specifications_met:
            report.append(f"\n✅ ESPECIFICAÇÕES ATENDIDAS:")
            for spec, met in self.specifications_met.items():
                status = "✅" if met else "❌"
                report.append(f"   {status} {spec}")

        # Métricas de desempenho
        if self.performance_metrics:
            report.append(f"\n📊 MÉTRICAS DE DESEMPENHO:")
            for metric, value in self.performance_metrics.items():
                report.append(f"   📈 {metric}: {value}")

        # Passos do projeto
        if self.design_steps:
            report.append(f"\n📋 PASSOS DO PROJETO:")
            for i, step in enumerate(self.design_steps, 1):
                report.append(f"   {i}. {step}")

        # Conteúdo educacional
        if self.educational_content:
            report.append(f"\n🎓 NOTAS EDUCACIONAIS:")
            for note in self.educational_content:
                report.append(f"   📚 {note}")

        return "\n".join(report)

def validate_closed_loop_stability(
    plant: SymbolicTransferFunction,
    controller: SymbolicTransferFunction,
    show_steps: bool = True
) -> Tuple[bool, Dict[str, Any]]:
    """
    Valida se o sistema em malha fechada é estável

    Args:
        plant: Planta do sistema
        controller: Controlador projetado
        show_steps: Se deve mostrar passos detalhados

    Returns:
        Tuple[bool, Dict]: (estável, análise_detalhada)
    """
    from ..analysis.stability_analysis import StabilityAnalysisEngine

    # Calcular sistema em malha fechada
    # T(s) = C(s)G(s) / (1 + C(s)G(s))
    cg = controller * plant
    closed_loop = cg / (1 + cg)
    closed_loop = closed_loop.simplify()

    if show_steps:
        print("📋 VALIDAÇÃO DE ESTABILIDADE EM MALHA FECHADA")
        print("=" * 50)
        print(f"🔧 Controlador: C(s) = {controller}")
        print(f"🏭 Planta: G(s) = {plant}")
        print(f"🔄 Malha fechada: T(s) = {closed_loop}")

    # Analisar estabilidade
    engine = StabilityAnalysisEngine()
    stability_result = engine.comprehensive_analysis(closed_loop, show_all_steps=show_steps)

    if stability_result:
        is_stable = stability_result.is_stable
        analysis = {
            'closed_loop': closed_loop,
            'stability_result': stability_result,
            'is_stable': is_stable
        }
    else:
        is_stable = False
        analysis = {
            'closed_loop': closed_loop,
            'error': 'Falha na análise de estabilidade'
        }

    return is_stable, analysis

def calculate_performance_metrics(
    closed_loop: SymbolicTransferFunction,
    specifications: Optional[DesignSpecifications] = None
) -> Tuple[Dict[str, float], SymbolicTransferFunction]:
    """
    Calcula métricas de desempenho do sistema

    Args:
        closed_loop: Sistema em malha fechada
        specifications: Especificações de projeto

    Returns:
        Tuple[Dict[str, float], SymbolicTransferFunction]: Métricas calculadas e o objeto closed_loop modificado
    """
    metrics = {}

    # --- INÍCIO DA MODIFICAÇÃO ---

    # 1. Chama a nova função de verificação
    is_valid, warning_message, dominant_poles = verify_second_order_approximation(closed_loop)

    # 2. Se a aproximação NÃO for válida, adiciona o aviso ao histórico do objeto
    if not is_valid:
        closed_loop.history.add_step(
            operation="Aviso de Análise",
            description="Validação da aproximação de segunda ordem.",
            before=closed_loop,
            after=closed_loop, # O objeto não muda, apenas recebe um aviso
            explanation=warning_message
        )

    # 3. Continua o cálculo das métricas...
    # A lógica para calcular overshoot, etc., pode agora usar os 'dominant_poles'
    # retornados pela função de verificação, sabendo que são os corretos.

    # ... (resto da lógica para calcular overshoot, settling time, etc.)

    # --- FIM DA MODIFICAÇÃO ---

    try:
        # Obter polos do sistema
        poles = closed_loop.get_poles()

        if poles:
            if dominant_poles:
                # Encontrar polo dominante (menor |parte_real|)
                dominant_pole = min(dominant_poles, key=lambda p: abs(p.real))

                # Calcular métricas baseadas no polo dominante
                if dominant_pole.imag != 0:  # Polos complexos
                    wn = abs(dominant_pole)  # Frequência natural
                    zeta = -dominant_pole.real / wn  # Coeficiente de amortecimento

                    metrics['natural_frequency'] = wn
                    metrics['damping_ratio'] = zeta

                    if zeta < 1:  # Sistema subamortecido
                        # Tempo de acomodação (2% do valor final)
                        metrics['settling_time_2pct'] = 4 / abs(dominant_pole.real)

                        # Sobressinal
                        overshoot = 100 * np.exp(-np.pi * zeta / np.sqrt(1 - zeta**2))
                        metrics['overshoot_percent'] = overshoot

                        # Tempo de pico
                        if zeta < 1:
                            wd = wn * np.sqrt(1 - zeta**2)
                            metrics['peak_time'] = np.pi / wd

                else:  # Polo real
                    # Tempo de acomodação para sistema de primeira ordem
                    metrics['time_constant'] = 1 / abs(dominant_pole.real)
                    metrics['settling_time_2pct'] = 4 * metrics['time_constant']

        # Erro em regime permanente (teorema do valor final)
        s = sp.Symbol('s')
        try:
            # Para entrada degrau unitário
            steady_state_error = 1 - sp.limit(closed_loop.expression, s, 0)
            if steady_state_error.is_real:
                metrics['steady_state_error_step'] = float(steady_state_error)
        except:
            pass

    except Exception as e:
        metrics['calculation_error'] = str(e)

    return metrics, closed_loop

def create_educational_content(method: str, parameters: Dict[str, Any]) -> List[str]:
    """
    Cria conteúdo educacional baseado no método de projeto

    Args:
        method: Método de projeto utilizado
        parameters: Parâmetros do projeto

    Returns:
        List[str]: Lista de explicações educacionais
    """
    content = []

    if method == "lead_compensator":
        content.extend([
            "🎓 COMPENSADOR LEAD (AVANÇO DE FASE):",
            "• Adiciona fase positiva para melhorar margem de fase",
            "• Fórmula: C(s) = K(s + z)/(s + p), onde p > z",
            "• Aumenta largura de banda e velocidade de resposta",
            "• Pode amplificar ruído em altas frequências"
        ])

    elif method == "lag_compensator":
        content.extend([
            "🎓 COMPENSADOR LAG (ATRASO DE FASE):",
            "• Reduz erro em regime permanente",
            "• Fórmula: C(s) = K(s + z)/(s + p), onde z > p",
            "• Não afeta significativamente resposta transitória",
            "• Diminui largura de banda"
        ])

    elif method == "pid":
        content.extend([
            "🎓 CONTROLADOR PID:",
            "• Proporcional: Kp - reduz erro, pode causar instabilidade",
            "• Integral: Ki - elimina erro em regime, pode causar overshoot",
            "• Derivativo: Kd - melhora estabilidade, sensível a ruído",
            "• Fórmula: C(s) = Kp + Ki/s + Kd*s"
        ])

    elif method == "pole_placement":
        content.extend([
            "🎓 ALOCAÇÃO DE POLOS:",
            "• Posiciona polos em locais desejados do plano s",
            "• Requer sistema completamente controlável",
            "• Fórmula de Ackermann: K = [0...0 1] * Wc^(-1) * αc(A)",
            "• Permite controle direto da resposta transitória"
        ])

    elif method == "lqr":
        content.extend([
            "🎓 REGULADOR LINEAR QUADRÁTICO (LQR):",
            "• Minimiza função custo quadrática J = ∫(x'Qx + u'Ru)dt",
            "• Solução: K = R^(-1) * B' * P",
            "• P é solução da Equação Algébrica de Riccati",
            "• Garante margem de fase de pelo menos 60°"
        ])

    return content

def format_design_steps(steps: List[str]) -> str:
    """
    Formata os passos de projeto de forma pedagógica

    Args:
        steps: Lista de passos do projeto

    Returns:
        str: Passos formatados
    """
    if not steps:
        return ""

    formatted = []
    formatted.append("📋 PASSOS DO PROJETO:")
    formatted.append("=" * 40)

    for i, step in enumerate(steps, 1):
        formatted.append(f"📝 Passo {i}: {step}")
        formatted.append("-" * 30)

    return "\n".join(formatted)
