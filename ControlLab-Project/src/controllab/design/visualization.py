"""
ControlLab - Visualização do Processo de Projeto
===============================================

Este módulo implementa visualizações educacionais para o processo de projeto
de controladores, incluindo análise de trade-offs e animações.

Características:
- Visualização do efeito de compensadores
- Plotagem de trade-offs de projeto
- Animação de alocação de polos
- Comparação antes/depois do projeto
"""

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union, Tuple, Any
from ..core.symbolic_tf import SymbolicTransferFunction
from ..core.symbolic_ss import SymbolicStateSpace
from .design_utils import ControllerResult, DesignSpecifications
from .specifications import PerformanceSpec

def show_compensator_effect(plant: SymbolicTransferFunction,
                           compensator: SymbolicTransferFunction,
                           frequency_range: Optional[Tuple[float, float]] = None,
                           show_plots: bool = True) -> Dict[str, Any]:
    """
    Mostra o efeito de um compensador na planta
    
    Args:
        plant: Planta original G(s)
        compensator: Compensador C(s)
        frequency_range: Faixa de frequências (min, max) em rad/s
        show_plots: Se deve mostrar gráficos
    
    Returns:
        Dict: Análise comparativa do efeito
    """
    print("🎯 ANÁLISE DO EFEITO DO COMPENSADOR")
    print("=" * 50)
    print(f"🏭 Planta original: G(s) = {plant}")
    print(f"🔧 Compensador: C(s) = {compensator}")
    
    # Sistema compensado
    compensated = compensator * plant
    print(f"🔄 Sistema compensado: C(s)G(s) = {compensated}")
    
    # Sistema em malha fechada
    closed_loop = compensated / (1 + compensated)
    closed_loop = closed_loop.simplify()
    print(f"🎛️ Malha fechada: T(s) = {closed_loop}")
    
    analysis = {
        'original_plant': plant,
        'compensator': compensator,
        'compensated_system': compensated,
        'closed_loop': closed_loop,
        'improvements': []
    }
    
    try:
        # Análise de polos
        original_poles = plant.get_poles()
        cl_poles = closed_loop.get_poles()
        
        print(f"\n📊 ANÁLISE DE POLOS:")
        print(f"   🔴 Polos da planta: {original_poles}")
        print(f"   🔵 Polos em malha fechada: {cl_poles}")
        
        # Verificar melhorias
        if cl_poles and original_poles:
            # Verificar se polos ficaram mais estáveis (mais negativos)
            original_real_parts = [float(sp.re(p)) for p in original_poles if sp.re(p).is_real]
            cl_real_parts = [float(sp.re(p)) for p in cl_poles if sp.re(p).is_real]
            
            if cl_real_parts and original_real_parts:
                if min(cl_real_parts) < min(original_real_parts):
                    analysis['improvements'].append("✅ Polos mais estáveis (mais negativos)")
                else:
                    analysis['improvements'].append("⚠️ Polos menos estáveis")
        
        # Análise de ganho DC
        original_dc = plant.evaluate_at(0)
        cl_dc = closed_loop.evaluate_at(0)
        
        print(f"\n📈 ANÁLISE DE GANHO:")
        print(f"   📊 Ganho DC original: {original_dc}")
        print(f"   📊 Ganho DC malha fechada: {cl_dc}")
        
        if abs(cl_dc - 1) < abs(original_dc - 1):
            analysis['improvements'].append("✅ Melhor rastreamento (ganho DC mais próximo de 1)")
        
    except Exception as e:
        analysis['error'] = f"Erro na análise: {str(e)}"
        print(f"⚠️ Erro na análise numérica: {e}")
    
    # Análise simbólica do tipo de compensador
    s = sp.Symbol('s')
    comp_expr = compensator.expression
    
    print(f"\n🔍 ANÁLISE DO COMPENSADOR:")
    if comp_expr.has(s**2):
        print("   📝 Contém termo derivativo (ação D)")
        analysis['improvements'].append("📝 Ação derivativa adicionada")
    
    if comp_expr.has(1/s):
        print("   📝 Contém termo integral (ação I)")
        analysis['improvements'].append("📝 Ação integral adicionada")
    
    # Verificar se é lead ou lag
    try:
        # Tentar identificar zeros e polos
        zeros = compensator.get_zeros()
        poles = compensator.get_poles()
        
        if zeros and poles:
            print(f"   🎯 Zeros: {zeros}")
            print(f"   📍 Polos: {poles}")
            
            # Comparar magnitudes (aproximação para casos simples)
            try:
                zero_mag = abs(float(zeros[0])) if zeros[0].is_real else abs(complex(zeros[0]))
                pole_mag = abs(float(poles[0])) if poles[0].is_real else abs(complex(poles[0]))
                
                if zero_mag < pole_mag:
                    print("   📐 Compensador Lead (avanço de fase)")
                    analysis['improvements'].append("📐 Avanço de fase adicionado")
                elif zero_mag > pole_mag:
                    print("   📐 Compensador Lag (atraso de fase)")
                    analysis['improvements'].append("📐 Redução de erro em regime")
            except:
                pass
    except:
        pass
    
    print(f"\n✅ MELHORIAS IDENTIFICADAS:")
    for improvement in analysis['improvements']:
        print(f"   {improvement}")
    
    return analysis

def plot_design_tradeoffs(specifications: List[PerformanceSpec],
                         solutions: List[ControllerResult],
                         criteria: List[str] = None) -> Dict[str, Any]:
    """
    Plota trade-offs entre diferentes soluções de projeto
    
    Args:
        specifications: Lista de especificações testadas
        solutions: Lista de soluções correspondentes
        criteria: Critérios para comparação
    
    Returns:
        Dict: Análise de trade-offs
    """
    if criteria is None:
        criteria = ['overshoot', 'settling_time', 'steady_state_error', 'control_effort']
    
    print("📊 ANÁLISE DE TRADE-OFFS DE PROJETO")
    print("=" * 50)
    
    tradeoff_data = {
        'solutions': [],
        'scores': {},
        'recommendations': []
    }
    
    for i, (spec, solution) in enumerate(zip(specifications, solutions)):
        solution_name = f"Solução {i+1}"
        print(f"\n🎯 {solution_name}:")
        print(f"   🔧 Controlador: {solution.controller}")
        
        # Calcular pontuações para cada critério
        scores = {}
        
        # Sobressinal (menor é melhor)
        if spec.overshoot is not None:
            scores['overshoot'] = max(0, 100 - spec.overshoot) / 100
            print(f"   📈 Sobressinal: {spec.overshoot:.1f}% (score: {scores['overshoot']:.2f})")
        
        # Tempo de acomodação (menor é melhor)
        if spec.settling_time is not None:
            scores['settling_time'] = max(0, 1 / (1 + spec.settling_time))
            print(f"   ⏱️ Tempo acomodação: {spec.settling_time:.2f}s (score: {scores['settling_time']:.2f})")
        
        # Erro regime (menor é melhor)
        if spec.steady_state_error is not None:
            scores['steady_state_error'] = max(0, 1 - spec.steady_state_error)
            print(f"   🎯 Erro regime: {spec.steady_state_error:.2f}% (score: {scores['steady_state_error']:.2f})")
        
        # Score geral (média dos scores disponíveis)
        if scores:
            overall_score = sum(scores.values()) / len(scores)
            print(f"   🏆 Score geral: {overall_score:.2f}")
        else:
            overall_score = 0
        
        tradeoff_data['solutions'].append({
            'name': solution_name,
            'controller': solution.controller,
            'scores': scores,
            'overall_score': overall_score
        })
    
    # Determinar melhor solução
    if tradeoff_data['solutions']:
        best_solution = max(tradeoff_data['solutions'], key=lambda x: x['overall_score'])
        print(f"\n🏆 MELHOR SOLUÇÃO: {best_solution['name']}")
        print(f"   🎯 Score: {best_solution['overall_score']:.2f}")
        print(f"   🔧 Controlador: {best_solution['controller']}")
        
        tradeoff_data['recommendations'].append(f"Recomendação: {best_solution['name']}")
    
    # Análise de conflitos
    print(f"\n⚖️ ANÁLISE DE CONFLITOS:")
    tradeoff_data['recommendations'].append("Conflitos típicos:")
    tradeoff_data['recommendations'].append("• Velocidade vs Estabilidade")
    tradeoff_data['recommendations'].append("• Precisão vs Robustez")
    tradeoff_data['recommendations'].append("• Performance vs Esforço de controle")
    
    for rec in tradeoff_data['recommendations']:
        print(f"   {rec}")
    
    return tradeoff_data

def animate_pole_placement(system: SymbolicStateSpace,
                          pole_trajectory: List[List[complex]],
                          show_animation: bool = True) -> Dict[str, Any]:
    """
    Anima o processo de alocação de polos
    
    Args:
        system: Sistema em espaço de estados
        pole_trajectory: Trajetória dos polos durante o projeto
        show_animation: Se deve mostrar animação
    
    Returns:
        Dict: Dados da animação
    """
    print("🎬 ANIMAÇÃO DE ALOCAÇÃO DE POLOS")
    print("=" * 40)
    print(f"🎛️ Sistema: {system}")
    
    animation_data = {
        'initial_poles': [],
        'final_poles': [],
        'trajectory': pole_trajectory,
        'stability_evolution': []
    }
    
    if pole_trajectory:
        initial_poles = pole_trajectory[0]
        final_poles = pole_trajectory[-1]
        
        print(f"🔴 Polos iniciais: {initial_poles}")
        print(f"🔵 Polos finais: {final_poles}")
        
        animation_data['initial_poles'] = initial_poles
        animation_data['final_poles'] = final_poles
        
        # Analisar evolução da estabilidade
        for i, poles in enumerate(pole_trajectory):
            stable = all(p.real < 0 for p in poles)
            animation_data['stability_evolution'].append(stable)
            
            if i == 0:
                status = "INICIAL"
            elif i == len(pole_trajectory) - 1:
                status = "FINAL"
            else:
                status = f"PASSO {i}"
            
            stability_status = "ESTÁVEL" if stable else "INSTÁVEL"
            print(f"   📍 {status}: {poles} - {stability_status}")
        
        # Calcular movimento dos polos
        total_movement = 0
        for i in range(len(initial_poles)):
            movement = abs(final_poles[i] - initial_poles[i])
            total_movement += movement
            print(f"   📏 Polo {i+1} moveu: {movement:.3f}")
        
        print(f"   📊 Movimento total: {total_movement:.3f}")
        animation_data['total_movement'] = total_movement
        
        # Verificar se melhorou estabilidade
        initial_stable = animation_data['stability_evolution'][0]
        final_stable = animation_data['stability_evolution'][-1]
        
        if not initial_stable and final_stable:
            print("   ✅ Sistema estabilizado!")
            animation_data['improvement'] = "Estabilizado"
        elif initial_stable and final_stable:
            # Calcular melhoria na margem de estabilidade
            initial_margin = min(-p.real for p in initial_poles)
            final_margin = min(-p.real for p in final_poles)
            
            if final_margin > initial_margin:
                print(f"   ✅ Margem de estabilidade melhorada: {initial_margin:.3f} → {final_margin:.3f}")
                animation_data['improvement'] = f"Margem +{final_margin-initial_margin:.3f}"
            else:
                print(f"   ⚠️ Margem de estabilidade reduzida: {initial_margin:.3f} → {final_margin:.3f}")
                animation_data['improvement'] = f"Margem {final_margin-initial_margin:.3f}"
        else:
            print("   ❌ Sistema desestabilizado!")
            animation_data['improvement'] = "Desestabilizado"
    
    return animation_data

def visualize_controller_design_process(plant: SymbolicTransferFunction,
                                      specifications: PerformanceSpec,
                                      controller: SymbolicTransferFunction,
                                      show_steps: bool = True) -> Dict[str, Any]:
    """
    Visualiza todo o processo de projeto de controlador
    
    Args:
        plant: Planta do sistema
        specifications: Especificações de projeto
        controller: Controlador projetado
        show_steps: Se deve mostrar passos detalhados
    
    Returns:
        Dict: Dados completos da visualização
    """
    if show_steps:
        print("🎨 VISUALIZAÇÃO COMPLETA DO PROCESSO DE PROJETO")
        print("=" * 60)
    
    visualization_data = {
        'plant_analysis': {},
        'specification_analysis': {},
        'design_process': {},
        'final_analysis': {}
    }
    
    # 1. Análise da planta
    if show_steps:
        print("\n1️⃣ ANÁLISE DA PLANTA:")
        print(f"   🏭 G(s) = {plant}")
    
    try:
        plant_poles = plant.get_poles()
        plant_zeros = plant.get_zeros()
        
        if show_steps:
            print(f"   📍 Polos: {plant_poles}")
            print(f"   🎯 Zeros: {plant_zeros}")
        
        # Verificar estabilidade da planta
        plant_stable = all(sp.re(p) < 0 for p in plant_poles if p.is_finite)
        stability_status = "ESTÁVEL" if plant_stable else "INSTÁVEL"
        
        if show_steps:
            print(f"   🔍 Estabilidade: {stability_status}")
        
        visualization_data['plant_analysis'] = {
            'poles': plant_poles,
            'zeros': plant_zeros,
            'stable': plant_stable
        }
        
    except Exception as e:
        if show_steps:
            print(f"   ⚠️ Erro na análise: {e}")
        visualization_data['plant_analysis']['error'] = str(e)
    
    # 2. Análise das especificações
    if show_steps:
        print("\n2️⃣ ANÁLISE DAS ESPECIFICAÇÕES:")
    
    spec_valid, spec_errors = specifications.validate_consistency()
    if show_steps:
        if spec_valid:
            print("   ✅ Especificações consistentes")
        else:
            print("   ❌ Especificações inconsistentes:")
            for error in spec_errors:
                print(f"      • {error}")
    
    # Converter especificações
    zeta, wn = specifications.to_second_order_params()
    freq_specs = specifications.to_frequency_specs()
    
    if show_steps and zeta is not None and wn is not None:
        print(f"   📊 Parâmetros: ζ={zeta:.3f}, ωn={wn:.3f} rad/s")
    
    if show_steps and freq_specs:
        print("   🔄 Equiv. frequência:")
        for spec, value in freq_specs.items():
            print(f"      📐 {spec}: {value:.2f}")
    
    visualization_data['specification_analysis'] = {
        'valid': spec_valid,
        'errors': spec_errors,
        'damping_ratio': zeta,
        'natural_frequency': wn,
        'frequency_specs': freq_specs
    }
    
    # 3. Processo de projeto
    if show_steps:
        print("\n3️⃣ PROCESSO DE PROJETO:")
        print(f"   🔧 Controlador: C(s) = {controller}")
    
    # Análise do efeito do controlador
    effect_analysis = show_compensator_effect(plant, controller, show_plots=False)
    visualization_data['design_process'] = effect_analysis
    
    # 4. Análise final
    if show_steps:
        print("\n4️⃣ ANÁLISE FINAL:")
    
    try:
        # Sistema em malha fechada
        compensated = controller * plant
        closed_loop = compensated / (1 + compensated)
        closed_loop = closed_loop.simplify()
        
        if show_steps:
            print(f"   🎛️ Malha fechada: T(s) = {closed_loop}")
        
        # Verificar especificações
        from .specifications import verify_specifications
        verification = verify_specifications(closed_loop, specifications)
        
        if show_steps:
            print("   📋 Verificação de especificações:")
            for spec, met in verification.items():
                status = "✅" if met else "❌"
                print(f"      {status} {spec}")
        
        visualization_data['final_analysis'] = {
            'closed_loop': closed_loop,
            'verification': verification
        }
        
    except Exception as e:
        if show_steps:
            print(f"   ⚠️ Erro na análise final: {e}")
        visualization_data['final_analysis']['error'] = str(e)
    
    return visualization_data

def compare_before_after(plant: SymbolicTransferFunction,
                        controller: SymbolicTransferFunction,
                        show_detailed: bool = True) -> Dict[str, Any]:
    """
    Compara sistema antes e depois do controlador
    
    Args:
        plant: Planta original
        controller: Controlador projetado
        show_detailed: Se deve mostrar análise detalhada
    
    Returns:
        Dict: Comparação completa
    """
    print("🔄 COMPARAÇÃO ANTES vs DEPOIS")
    print("=" * 40)
    
    comparison = {
        'before': {},
        'after': {},
        'improvements': [],
        'degradations': []
    }
    
    try:
        # Sistema original (malha aberta)
        original_poles = plant.get_poles()
        original_zeros = plant.get_zeros()
        
        # Sistema com controlador (malha fechada)
        compensated = controller * plant
        closed_loop = compensated / (1 + compensated)
        closed_loop = closed_loop.simplify()
        
        cl_poles = closed_loop.get_poles()
        cl_zeros = closed_loop.get_zeros()
        
        print("📊 ANTES (Malha Aberta):")
        print(f"   🏭 Sistema: G(s) = {plant}")
        print(f"   📍 Polos: {original_poles}")
        print(f"   🎯 Zeros: {original_zeros}")
        
        print("\n📊 DEPOIS (Malha Fechada):")
        print(f"   🎛️ Sistema: T(s) = {closed_loop}")
        print(f"   📍 Polos: {cl_poles}")
        print(f"   🎯 Zeros: {cl_zeros}")
        
        comparison['before'] = {
            'system': plant,
            'poles': original_poles,
            'zeros': original_zeros
        }
        
        comparison['after'] = {
            'system': closed_loop,
            'poles': cl_poles,
            'zeros': cl_zeros
        }
        
        # Análise de melhorias
        print("\n✅ MELHORIAS:")
        
        # Verificar estabilidade
        original_stable = all(sp.re(p) < 0 for p in original_poles if p.is_finite)
        cl_stable = all(sp.re(p) < 0 for p in cl_poles if p.is_finite)
        
        if not original_stable and cl_stable:
            improvement = "Sistema estabilizado"
            comparison['improvements'].append(improvement)
            print(f"   🎯 {improvement}")
        
        # Verificar velocidade de resposta
        try:
            original_fastest = max(-float(sp.re(p)) for p in original_poles if sp.re(p).is_real)
            cl_fastest = max(-float(sp.re(p)) for p in cl_poles if sp.re(p).is_real)
            
            if cl_fastest > original_fastest:
                improvement = f"Resposta mais rápida: {original_fastest:.2f} → {cl_fastest:.2f}"
                comparison['improvements'].append(improvement)
                print(f"   ⚡ {improvement}")
        except:
            pass
        
        # Verificar rastreamento
        try:
            original_dc = plant.evaluate_at(0)
            cl_dc = closed_loop.evaluate_at(0)
            
            if abs(cl_dc - 1) < abs(original_dc):
                improvement = f"Melhor rastreamento: erro DC {abs(original_dc):.3f} → {abs(cl_dc-1):.3f}"
                comparison['improvements'].append(improvement)
                print(f"   🎯 {improvement}")
        except:
            pass
        
        if not comparison['improvements']:
            print("   📝 Nenhuma melhoria quantitativa detectada")
        
        # Possíveis degradações
        print("\n⚠️ POSSÍVEIS TRADE-OFFS:")
        
        # Verificar se aumentou ordem do sistema
        original_order = len(original_poles)
        cl_order = len(cl_poles)
        
        if cl_order > original_order:
            degradation = f"Ordem aumentada: {original_order} → {cl_order}"
            comparison['degradations'].append(degradation)
            print(f"   📈 {degradation}")
        
        # Verificar complexidade
        controller_order = len(controller.get_poles())
        if controller_order > 1:
            degradation = f"Controlador de ordem {controller_order}"
            comparison['degradations'].append(degradation)
            print(f"   🔧 {degradation}")
        
        if not comparison['degradations']:
            print("   ✅ Nenhuma degradação significativa detectada")
        
    except Exception as e:
        error_msg = f"Erro na comparação: {str(e)}"
        comparison['error'] = error_msg
        print(f"❌ {error_msg}")
    
    return comparison
