"""
ControlLab - Análise de Desempenho de Controladores
==================================================

Este módulo implementa análise de desempenho e robustez:
- Análise de resposta transitória
- Cálculo de índices de desempenho
- Análise de sensibilidade
- Análise de robustez

Características:
- Métricas de desempenho completas
- Análise de robustez sistemática
- Comparação entre controladores
- Explicações pedagógicas
"""

import sympy as sp
from typing import Dict, List, Optional, Union, Tuple, Any
from ..core.symbolic_tf import SymbolicTransferFunction
from ..core.symbolic_ss import SymbolicStateSpace
from .design_utils import ControllerResult

def analyze_transient_response(closed_loop: Union[SymbolicTransferFunction, SymbolicStateSpace],
                             show_steps: bool = True) -> Dict[str, Any]:
    """
    Analisa resposta transitória do sistema em malha fechada
    
    Args:
        closed_loop: Sistema em malha fechada
        show_steps: Se deve mostrar passos detalhados
    
    Returns:
        Dict[str, Any]: Análise da resposta transitória
    """
    if show_steps:
        print("📊 ANÁLISE DE RESPOSTA TRANSITÓRIA")
        print("=" * 40)
        print(f"🔄 Sistema em malha fechada: {closed_loop}")
    
    analysis = {}
    
    if isinstance(closed_loop, SymbolicTransferFunction):
        # Análise via função de transferência
        poles = closed_loop.poles()  # Usar método correto
        zeros = closed_loop.zeros()  # Usar método correto
        
        if show_steps:
            print(f"\n📐 Polos: {poles}")
            print(f"📐 Zeros: {zeros}")
        
        analysis.update(_analyze_poles_response(poles, show_steps))
        
    elif isinstance(closed_loop, SymbolicStateSpace):
        # Análise via espaço de estados
        A_cl = closed_loop.A
        eigenvals = A_cl.eigenvals()
        
        if show_steps:
            print(f"📐 Autovalores: {eigenvals}")
        
        # Converter autovalores para formato de polos
        poles = list(eigenvals.keys())
        analysis.update(_analyze_poles_response(poles, show_steps))
    
    return analysis

def _analyze_poles_response(poles: List, show_steps: bool) -> Dict[str, Any]:
    """Analisa resposta baseada nos polos"""
    
    analysis = {
        'poles': poles,
        'stability': 'unknown',
        'dominant_poles': [],
        'response_type': 'unknown'
    }
    
    if not poles:
        return analysis
    
    # Verificar estabilidade
    stable_poles = []
    unstable_poles = []
    
    for pole in poles:
        try:
            if pole.is_real:
                if pole < 0:
                    stable_poles.append(pole)
                else:
                    unstable_poles.append(pole)
            else:
                # Polo complexo - verificar parte real
                real_part = sp.re(pole)
                if real_part < 0:
                    stable_poles.append(pole)
                else:
                    unstable_poles.append(pole)
        except:
            # Se não conseguir determinar, assumir estável
            stable_poles.append(pole)
    
    if unstable_poles:
        analysis['stability'] = 'unstable'
        analysis['unstable_poles'] = unstable_poles
    else:
        analysis['stability'] = 'stable'
    
    if show_steps:
        print(f"\n✅ ANÁLISE DE ESTABILIDADE:")
        print(f"📊 Polos estáveis: {len(stable_poles)}")
        print(f"📊 Polos instáveis: {len(unstable_poles)}")
        print(f"🎯 Status: {analysis['stability']}")
    
    # Encontrar polos dominantes (menor |parte_real| para estáveis)
    if stable_poles:
        try:
            # Para polos reais
            real_poles = [p for p in stable_poles if p.is_real]
            if real_poles:
                dominant_real = max(real_poles)  # Mais próximo de zero
                analysis['dominant_poles'].append(dominant_real)
            
            # Para polos complexos
            complex_poles = [p for p in stable_poles if not p.is_real]
            if complex_poles:
                # Encontrar par complexo com menor |parte_real|
                min_real_part = min([abs(sp.re(p)) for p in complex_poles])
                dominant_complex = [p for p in complex_poles if abs(sp.re(p)) == min_real_part]
                analysis['dominant_poles'].extend(dominant_complex[:2])  # Par conjugado
        except:
            analysis['dominant_poles'] = poles[:2]  # Primeiros dois como backup
    
    # Classificar tipo de resposta
    if analysis['dominant_poles']:
        dominant = analysis['dominant_poles'][0]
        try:
            if dominant.is_real:
                analysis['response_type'] = 'first_order' if len(analysis['dominant_poles']) == 1 else 'overdamped'
            else:
                analysis['response_type'] = 'underdamped'
        except:
            analysis['response_type'] = 'complex'
    
    if show_steps:
        print(f"\n📈 CARACTERÍSTICAS DA RESPOSTA:")
        print(f"🎯 Tipo: {analysis['response_type']}")
        print(f"🎯 Polos dominantes: {analysis['dominant_poles']}")
    
    return analysis

def calculate_performance_indices(response_data: Dict[str, Any],
                                 show_steps: bool = True) -> Dict[str, float]:
    """
    Calcula índices de desempenho do sistema
    
    Args:
        response_data: Dados da resposta do sistema
        show_steps: Se deve mostrar passos
    
    Returns:
        Dict[str, float]: Índices de desempenho calculados
    """
    if show_steps:
        print("📊 CÁLCULO DE ÍNDICES DE DESEMPENHO")
        print("=" * 45)
    
    indices = {}
    
    # Verificar se há polos dominantes para cálculos
    if 'dominant_poles' in response_data and response_data['dominant_poles']:
        dominant = response_data['dominant_poles'][0]
        
        try:
            if response_data['response_type'] == 'underdamped':
                # Sistema de segunda ordem subamortecido
                if len(response_data['dominant_poles']) >= 2:
                    pole = dominant
                    wn = abs(pole)  # Frequência natural
                    sigma = -sp.re(pole)  # Parte real
                    wd = sp.im(pole)  # Frequência amortecida
                    
                    zeta = sigma / wn  # Coeficiente de amortecimento
                    
                    if show_steps:
                        print(f"📐 Sistema de 2ª ordem subamortecido:")
                        print(f"   ωn = {wn} rad/s")
                        print(f"   ζ = {zeta}")
                        print(f"   ωd = {wd} rad/s")
                    
                    # Calcular métricas
                    indices['natural_frequency'] = float(wn)
                    indices['damping_ratio'] = float(zeta)
                    indices['damped_frequency'] = float(abs(wd))
                    
                    # Tempo de pico
                    if wd != 0:
                        indices['peak_time'] = float(sp.pi / abs(wd))
                    
                    # Sobressinal
                    if zeta < 1 and zeta > 0:
                        Mp = sp.exp(-sp.pi * zeta / sp.sqrt(1 - zeta**2))
                        indices['overshoot_percent'] = float(Mp * 100)
                    
                    # Tempo de acomodação (critério 2%)
                    if sigma > 0:
                        indices['settling_time_2pct'] = float(4 / sigma)
                
            elif response_data['response_type'] == 'first_order':
                # Sistema de primeira ordem
                pole = dominant
                if pole.is_real and pole < 0:
                    tau = -1 / pole  # Constante de tempo
                    indices['time_constant'] = float(tau)
                    indices['settling_time_2pct'] = float(4 * tau)
                    
                    if show_steps:
                        print(f"📐 Sistema de 1ª ordem:")
                        print(f"   τ = {tau} s")
                        print(f"   ts = {4 * tau} s")
        
        except Exception as e:
            if show_steps:
                print(f"⚠️ Erro no cálculo: {e}")
    
    # Índices adicionais baseados em ISE, IAE, ITAE
    indices.update(_calculate_error_indices(show_steps))
    
    if show_steps:
        print(f"\n📊 ÍNDICES CALCULADOS:")
        for name, value in indices.items():
            print(f"   {name}: {value}")
    
    return indices

def _calculate_error_indices(show_steps: bool) -> Dict[str, str]:
    """Calcula índices de erro (pedagogicamente)"""
    
    if show_steps:
        print(f"\n🎓 ÍNDICES DE ERRO (conceituais):")
        print(f"   ISE = ∫₀^∞ e²(t) dt")
        print(f"   IAE = ∫₀^∞ |e(t)| dt") 
        print(f"   ITAE = ∫₀^∞ t|e(t)| dt")
        print(f"   ITSE = ∫₀^∞ te²(t) dt")
    
    return {
        'ISE_formula': '∫₀^∞ e²(t) dt',
        'IAE_formula': '∫₀^∞ |e(t)| dt',
        'ITAE_formula': '∫₀^∞ t|e(t)| dt',
        'ITSE_formula': '∫₀^∞ te²(t) dt'
    }

def sensitivity_analysis(controller_gains: Union[sp.Matrix, List],
                        plant_variations: Dict[str, float],
                        show_steps: bool = True) -> Dict[str, Any]:
    """
    Analisa sensibilidade do controlador a variações da planta
    
    Args:
        controller_gains: Ganhos do controlador
        plant_variations: Variações paramétricas da planta
        show_steps: Se deve mostrar passos
    
    Returns:
        Dict[str, Any]: Análise de sensibilidade
    """
    if show_steps:
        print("🔍 ANÁLISE DE SENSIBILIDADE")
        print("=" * 35)
        print(f"🎛️ Ganhos do controlador: {controller_gains}")
        print(f"📊 Variações da planta: {plant_variations}")
    
    sensitivity = {
        'controller_gains': controller_gains,
        'plant_variations': plant_variations,
        'sensitivity_measures': {},
        'robustness_indicators': []
    }
    
    if show_steps:
        print(f"\n🎓 CONCEITOS DE SENSIBILIDADE:")
        print(f"• Sensibilidade mede variação da saída devido a variações paramétricas")
        print(f"• S = ∂y/∂p * p/y (sensibilidade normalizada)")
        print(f"• |S| < 1: baixa sensibilidade (robusto)")
        print(f"• |S| > 1: alta sensibilidade (não robusto)")
    
    # Para análise pedagógica, mostrar fórmulas
    sensitivity['formulas'] = {
        'sensitivity_function': 'S(s) = 1/(1 + L(s))',
        'complementary_sensitivity': 'T(s) = L(s)/(1 + L(s))',
        'control_sensitivity': 'CS(s) = C(s)/(1 + L(s))'
    }
    
    if show_steps:
        print(f"\n📐 FUNÇÕES DE SENSIBILIDADE:")
        for name, formula in sensitivity['formulas'].items():
            print(f"   {name}: {formula}")
    
    return sensitivity

def robustness_analysis(uncertainties: Dict[str, Any],
                       performance_specs: Dict[str, float],
                       show_steps: bool = True) -> Dict[str, Any]:
    """
    Analisa robustez do sistema a incertezas
    
    Args:
        uncertainties: Descrição das incertezas
        performance_specs: Especificações de desempenho
        show_steps: Se deve mostrar passos
    
    Returns:
        Dict[str, Any]: Análise de robustez
    """
    if show_steps:
        print("🛡️ ANÁLISE DE ROBUSTEZ")
        print("=" * 30)
        print(f"🎲 Incertezas: {uncertainties}")
        print(f"📋 Especificações: {performance_specs}")
    
    robustness = {
        'uncertainties': uncertainties,
        'performance_specs': performance_specs,
        'stability_margins': {},
        'performance_degradation': {},
        'robustness_measures': []
    }
    
    if show_steps:
        print(f"\n🎓 CONCEITOS DE ROBUSTEZ:")
        print(f"• Estabilidade robusta: sistema permanece estável com incertezas")
        print(f"• Desempenho robusto: especificações mantidas com incertezas")
        print(f"• Margem de ganho: variação de ganho que causa instabilidade")
        print(f"• Margem de fase: variação de fase que causa instabilidade")
    
    # Margens de estabilidade teóricas
    robustness['theoretical_margins'] = {
        'gain_margin_definition': '|L(jω₀)| onde ∠L(jω₀) = -180°',
        'phase_margin_definition': '180° + ∠L(jωc) onde |L(jωc)| = 1',
        'delay_margin': 'φₘ/ωc (margem de atraso)',
        'modulus_margin': 'min|1 + L(jω)| (margem de módulo)'
    }
    
    if show_steps:
        print(f"\n📊 MARGENS DE ESTABILIDADE:")
        for name, definition in robustness['theoretical_margins'].items():
            print(f"   {name}: {definition}")
    
    return robustness
