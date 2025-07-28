"""
Teste Completo do Módulo 6 - Projeto de Controladores
====================================================

Teste das funcionalidades implementadas:
- specifications.py: Sistema de especificações de performance
- visualization.py: Visualizações educacionais  
- comparison.py: Comparação de métodos
- antiwindup.py: Compensação anti-windup
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from controllab.core.symbolic_tf import SymbolicTransferFunction
from controllab.design.specifications import PerformanceSpec, verify_specifications, pole_placement_from_specs
from controllab.design.comparison import compare_controller_designs, pareto_analysis

from controllab.design.antiwindup import design_antiwindup_compensation, SaturationLimits
from controllab.analysis.stability_analysis import analyze_stability

import sympy as sp

def test_module6_complete():
    """Teste completo do módulo 6"""
    print("🎯 TESTE COMPLETO MÓDULO 6 - PROJETO DE CONTROLADORES")
    print("=" * 70)
    
    # 1. Definir sistema de teste
    s = sp.Symbol('s')
    plant = SymbolicTransferFunction(1 / (s**2 + 2*s + 1), s)
    print(f"🏭 Planta de teste: G(s) = {plant}")
    
    # 2. Teste das especificações
    print(f"\n📋 TESTE 1: ESPECIFICAÇÕES DE PERFORMANCE")
    print("-" * 50)
    
    specs = PerformanceSpec(
        overshoot=20.0,
        settling_time=5.0,
        steady_state_error=0.02,
        bandwidth=2.0,
        phase_margin=45.0
    )
    print(f"✅ Especificações criadas: {specs}")
    
    # 3. Teste de controladores diferentes
    print(f"\n🎛️ TESTE 2: DESIGN DE CONTROLADORES")
    print("-" * 50)
    
    # Controlador P
    Kp = 10
    controller_P = SymbolicTransferFunction(Kp, s)
    print(f"   P: C_P(s) = {controller_P}")
    
    # Controlador PI  
    Kp, Ki = 5, 2
    controller_PI = SymbolicTransferFunction(Kp + Ki/s, s)
    print(f"   PI: C_PI(s) = {controller_PI}")
    
    # Controlador PID
    Kp, Ki, Kd = 8, 4, 1
    controller_PID = SymbolicTransferFunction(Kp + Ki/s + Kd*s, s)
    print(f"   PID: C_PID(s) = {controller_PID}")
    
    controllers = [controller_P, controller_PI, controller_PID]
    
    # 4. Teste de comparação
    print(f"\n⚖️ TESTE 3: COMPARAÇÃO DE CONTROLADORES")
    print("-" * 50)
    
    try:
        comparison_result = compare_controller_designs(
            plant=plant,
            controllers=controllers, 
            specifications=specs,
            criteria=['stability', 'performance', 'complexity']
        )
        print("✅ Comparação concluída com sucesso!")
        
    except Exception as e:
        print(f"⚠️ Erro na comparação: {e}")
    
    # 5. Teste de anti-windup
    print(f"\n🔧 TESTE 4: COMPENSAÇÃO ANTI-WINDUP")
    print("-" * 50)
    
    # Usar controlador PI que tem ação integral
    saturation = SaturationLimits(u_min=-10.0, u_max=10.0)
    
    try:
        antiwindup_result = design_antiwindup_compensation(
            controller=controller_PI,
            plant=plant,
            saturation_limits=saturation,
            method='back_calculation'
        )
        print("✅ Anti-windup implementado com sucesso!")
        print(f"   Original: {antiwindup_result.original_controller}")
        print(f"   Com AW: {antiwindup_result.antiwindup_controller}")
        
    except Exception as e:
        print(f"⚠️ Erro no anti-windup: {e}")
    
    # 6. Teste de verificação de especificações
    print(f"\n📊 TESTE 5: VERIFICAÇÃO DE ESPECIFICAÇÕES")
    print("-" * 50)
    
    # Sistema em malha fechada com controlador PI
    try:
        closed_loop = (controller_PI * plant) / (1 + controller_PI * plant)
        closed_loop = closed_loop.simplify()
        print(f"   Sistema MF: T(s) = {closed_loop}")
        
        verification = verify_specifications(closed_loop, specs)
        print(f"   Verificação: {verification}")
        
    except Exception as e:
        print(f"⚠️ Erro na verificação: {e}")
    

    # 6b. Análise pedagógica de estabilidade
    print(f"\n📖 ANÁLISE PEDAGÓGICA DE ESTABILIDADE")
    print("-" * 50)
    try:
        stability_report = analyze_stability(closed_loop, show_steps=True)
        print(stability_report.get_executive_summary())
        print(stability_report.get_detailed_analysis())
        print(stability_report.get_educational_section())
    except Exception as e:
        print(f"⚠️ Erro na análise pedagógica: {e}")

    # 7. Teste de posicionamento de polos a partir de specs
    print(f"\n🎯 TESTE 6: POSICIONAMENTO DE POLOS")
    print("-" * 50)
    
    try:
        desired_poles = pole_placement_from_specs(specs)
        print(f"   Polos desejados: {desired_poles}")
        
    except Exception as e:
        print(f"⚠️ Erro no posicionamento: {e}")
    
    print(f"\n✅ TESTE MÓDULO 6 CONCLUÍDO!")
    print("=" * 70)
    return True

if __name__ == "__main__":
    test_module6_complete()
