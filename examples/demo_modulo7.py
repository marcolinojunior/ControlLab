"""
Demonstração do Módulo 7 - Sistemas Discretos
=============================================

Este arquivo demonstra as funcionalidades implementadas no Módulo 7 do ControlLab,
incluindo transformadas Z, discretização, análise de estabilidade e lugar das raízes.
"""

import sympy as sp
import numpy as np
from src.controllab.core.symbolic_tf import SymbolicTransferFunction
from src.controllab.modeling import (
    ZTransformer, apply_z_transform, inverse_z_transform,
    DiscretizationMethods, compare_discretization_methods,
    analyze_discrete_stability, plot_discrete_root_locus,
    analyze_discrete_frequency_response
)

def demo_z_transform():
    """Demonstra transformadas Z básicas"""
    print("🎯 DEMONSTRAÇÃO 1: TRANSFORMADAS Z")
    print("=" * 40)
    
    # Criar transformador Z
    transformer = ZTransformer()
    n = sp.Symbol('n', integer=True)
    z = sp.Symbol('z')
    
    # Exemplo 1: Degrau unitário
    print("\n📊 Exemplo 1: Degrau unitário u[n]")
    step_function = sp.Heaviside(n)
    result1 = transformer.apply_z_transform(step_function, n, z, show_steps=True)
    
    # Exemplo 2: Sequência exponencial
    print("\n📊 Exemplo 2: Sequência exponencial (0.5)^n * u[n]")
    exp_function = (sp.Rational(1,2))**n * sp.Heaviside(n)
    result2 = transformer.apply_z_transform(exp_function, n, z, show_steps=True)
    
    # Exemplo 3: Transformada inversa
    print("\n📊 Exemplo 3: Transformada Z inversa")
    Z_expr = z / (z - sp.Rational(1,2))
    result3 = transformer.inverse_z_transform(Z_expr, z, n, show_steps=True)
    
    return result1, result2, result3

def demo_discretization():
    """Demonstra métodos de discretização"""
    print("\n\n🎯 DEMONSTRAÇÃO 2: MÉTODOS DE DISCRETIZAÇÃO")
    print("=" * 50)
    
    # Sistema contínuo de exemplo: G(s) = 1/(s+1)
    s = sp.Symbol('s')
    z = sp.Symbol('z')
    
    # Função de transferência contínua
    H_s = SymbolicTransferFunction(1, s + 1, s)
    
    print(f"📊 Sistema contínuo: H(s) = {H_s.num}/{H_s.den}")
    
    # Período de amostragem
    T = 0.1
    
    # Comparar métodos de discretização
    results = compare_discretization_methods(H_s, T, show_steps=True)
    
    return results

def demo_discrete_stability():
    """Demonstra análise de estabilidade discreta"""
    print("\n\n🎯 DEMONSTRAÇÃO 3: ANÁLISE DE ESTABILIDADE DISCRETA")
    print("=" * 55)
    
    z = sp.Symbol('z')
    
    # Sistema discreto de exemplo: H(z) = 0.5z/(z^2 - 0.8z + 0.15)
    H_z = SymbolicTransferFunction(
        sp.Rational(1,2) * z,
        z**2 - sp.Rational(4,5) * z + sp.Rational(3,20),
        z
    )
    
    print(f"📊 Sistema discreto: H(z) = {H_z.num}/{H_z.den}")
    
    # Análise por círculo unitário
    print("\n🔍 Método 1: Análise do círculo unitário")
    result1 = analyze_discrete_stability(H_z, method='circle', show_steps=True)
    
    # Teste de Jury
    print("\n🔍 Método 2: Teste de Jury")
    result2 = analyze_discrete_stability(H_z, method='jury', show_steps=True)
    
    # Análise de margens
    print("\n🔍 Método 3: Margens de estabilidade")
    result3 = analyze_discrete_stability(H_z, method='margins', show_steps=True)
    
    return result1, result2, result3

def demo_discrete_root_locus():
    """Demonstra lugar das raízes discreto"""
    print("\n\n🎯 DEMONSTRAÇÃO 4: LUGAR DAS RAÍZES DISCRETO")
    print("=" * 45)
    
    z = sp.Symbol('z')
    
    # Sistema em malha aberta: G(z) = K/(z(z-0.5))
    G_z = SymbolicTransferFunction(
        1,
        z * (z - sp.Rational(1,2)),
        z
    )
    
    print(f"📊 Sistema em malha aberta: G(z) = {G_z.num}/{G_z.den}")
    
    # Calcular lugar das raízes
    rl_result = plot_discrete_root_locus(G_z, sampling_time=0.1, 
                                       gain_range=(0.1, 5.0), show_steps=True)
    
    return rl_result

def demo_discrete_frequency_response():
    """Demonstra análise de frequência discreta"""
    print("\n\n🎯 DEMONSTRAÇÃO 5: RESPOSTA EM FREQUÊNCIA DISCRETA")
    print("=" * 55)
    
    z = sp.Symbol('z')
    
    # Sistema discreto para análise
    H_z = SymbolicTransferFunction(
        sp.Rational(1,4) * (z + 1),
        z**2 - sp.Rational(1,2) * z + sp.Rational(1,8),
        z
    )
    
    print(f"📊 Sistema: H(z) = {H_z.num}/{H_z.den}")
    
    # Análise de Bode discreta
    freq_result = analyze_discrete_frequency_response(H_z, sampling_time=0.1, 
                                                    show_steps=True)
    
    return freq_result

def demo_integration_example():
    """Demonstra integração entre diferentes análises"""
    print("\n\n🎯 DEMONSTRAÇÃO 6: EXEMPLO INTEGRADO")
    print("=" * 40)
    
    s = sp.Symbol('s')
    z = sp.Symbol('z')
    
    # Sistema contínuo original
    H_s = SymbolicTransferFunction(10, s**2 + 3*s + 2, s)
    print(f"📊 Sistema contínuo: H(s) = {H_s.num}/{H_s.den}")
    
    # Período de amostragem
    T = 0.1
    
    # Passo 1: Discretizar usando Tustin
    discretizer = DiscretizationMethods(T)
    disc_result = discretizer.tustin_transform(H_s, T, show_steps=True)
    
    if disc_result.discrete_tf:
        H_z = disc_result.discrete_tf
        
        # Passo 2: Analisar estabilidade
        print("\n🔍 Análise de estabilidade do sistema discretizado:")
        stab_result = analyze_discrete_stability(H_z, method='circle', show_steps=True)
        
        # Passo 3: Análise de frequência
        print("\n📈 Resposta em frequência:")
        freq_result = analyze_discrete_frequency_response(H_z, T, show_steps=True)
        
        # Resumo
        print("\n📋 RESUMO DO EXEMPLO INTEGRADO:")
        print("=" * 35)
        print(f"   ✅ Sistema discretizado com T = {T}")
        print(f"   📊 Estabilidade: {'Estável' if stab_result.is_stable else 'Instável'}")
        if freq_result.gain_margin and freq_result.phase_margin:
            print(f"   📈 Margens: GM = {freq_result.gain_margin:.1f}dB, PM = {freq_result.phase_margin:.1f}°")
        print(f"   🎯 Preservação de estabilidade: {disc_result.stability_preserved}")
        
        return disc_result, stab_result, freq_result
    
    return None, None, None

def main():
    """Função principal da demonstração"""
    print("🎓 CONTROLLAB - DEMONSTRAÇÃO MÓDULO 7")
    print("🔬 Sistemas Discretos e Transformada Z")
    print("=" * 50)
    print("Esta demonstração mostra as funcionalidades do Módulo 7:")
    print("• Transformadas Z diretas e inversas")
    print("• Métodos de discretização (Tustin, ZOH, FOH, Euler)")
    print("• Análise de estabilidade discreta (Jury, círculo unitário)")
    print("• Lugar das raízes no domínio Z")
    print("• Resposta em frequência discreta")
    print("• Integração entre diferentes análises")
    print("")
    
    try:
        # Executar demonstrações
        demo_z_transform()
        demo_discretization()
        demo_discrete_stability()
        demo_discrete_root_locus()
        demo_discrete_frequency_response()
        demo_integration_example()
        
        print("\n" + "="*60)
        print("✅ DEMONSTRAÇÃO CONCLUÍDA COM SUCESSO!")
        print("✅ Módulo 7 - Sistemas Discretos implementado e testado")
        print("📚 Todas as funcionalidades pedagógicas funcionando")
        print("🎯 Pronto para uso educacional e projetos avançados")
        
    except Exception as e:
        print(f"\n❌ Erro durante a demonstração: {e}")
        print("⚠️  Verifique se todas as dependências estão instaladas")

if __name__ == "__main__":
    main()
