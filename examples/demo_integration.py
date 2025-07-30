#!/usr/bin/env python3
"""
Demo de Integração Simbólico-Numérico - ControlLab
Demonstração prática da ponte entre análise simbólica e computação numérica
"""

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from pathlib import Path
import sys

# Adicionar caminho dos módulos
sys.path.append(str(Path(__file__).parent.parent / "src"))

from controllab.core.symbolic_tf import SymbolicTransferFunction
from controllab.numerical.interface import NumericalInterface
from controllab.numerical.factory import NumericalSystemFactory
from controllab.numerical.validation import NumericalValidator

def demo_basic_conversion():
    """Demonstra conversão básica simbólico → numérico"""
    print("="*60)
    print("DEMO 1: Conversão Básica Simbólico → Numérico")
    print("="*60)
    
    # 1. Criar sistema simbólico
    s = sp.Symbol('s')
    tf_symbolic = SymbolicTransferFunction(1, s + 1, s)
    
    print(f"📊 Sistema Simbólico: {tf_symbolic}")
    
    # 2. Converter para numérico
    interface = NumericalInterface()
    tf_numeric = interface.symbolic_to_control_tf(tf_symbolic)
    
    print(f"🔢 Sistema Numérico: {tf_numeric}")
    print(f"   Polos: {tf_numeric.poles()}")
    print(f"   Zeros: {tf_numeric.zeros()}")
    
    return tf_symbolic, tf_numeric

def demo_parametric_analysis():
    """Demonstra análise paramétrica com substituições"""
    print("\n" + "="*60)
    print("DEMO 2: Análise Paramétrica")
    print("="*60)
    
    # Sistema paramétrico: G(s) = K / (s^2 + 2ζωn*s + ωn^2)
    s = sp.Symbol('s')
    K, zeta, wn = sp.symbols('K zeta wn', real=True, positive=True)
    
    # Sistema de segunda ordem padrão
    numerator = K * wn**2
    denominator = s**2 + 2*zeta*wn*s + wn**2
    tf_param = SymbolicTransferFunction(numerator, denominator, s)
    
    print(f"📊 Sistema Paramétrico: {tf_param}")
    
    # Diferentes configurações
    configs = [
        {"K": 1, "zeta": 0.1, "wn": 10, "nome": "Subamortecido"},
        {"K": 1, "zeta": 0.7, "wn": 10, "nome": "Amortecido"},
        {"K": 1, "zeta": 1.0, "wn": 10, "nome": "Criticamente Amortecido"},
        {"K": 2, "zeta": 0.3, "wn": 5, "nome": "Alta Frequência"}
    ]
    
    interface = NumericalInterface()
    
    plt.figure(figsize=(12, 8))
    
    for i, config in enumerate(configs):
        # Aplicar substituições
        substitutions = {K: config["K"], zeta: config["zeta"], wn: config["wn"]}
        
        # Converter e analisar
        tf_numeric = interface.symbolic_to_control_tf(tf_param, substitutions)
        time, response = interface.compute_step_response(tf_numeric)
        
        # Plotar
        plt.subplot(2, 2, i+1)
        plt.plot(time, response)
        plt.title(f'{config["nome"]}\nK={config["K"]}, ζ={config["zeta"]}, ωn={config["wn"]}')
        plt.xlabel('Tempo (s)')
        plt.ylabel('Amplitude')
        plt.grid(True)
        
        print(f"✓ {config['nome']}: Polos = {tf_numeric.poles()}")
    
    plt.tight_layout()
    plt.savefig('demo_parametric_analysis.png', dpi=300, bbox_inches='tight')
    print(f"📈 Gráfico salvo: demo_parametric_analysis.png")
    
    return configs

def demo_frequency_analysis():
    """Demonstra análise de resposta em frequência"""
    print("\n" + "="*60)
    print("DEMO 3: Análise de Frequência")
    print("="*60)
    
    # Sistema com diferentes características
    s = sp.Symbol('s')
    systems = [
        {
            "tf": SymbolicTransferFunction(100, s**2 + 10*s + 100, s),
            "nome": "Passa-baixa",
            "cor": "blue"
        },
        {
            "tf": SymbolicTransferFunction(s**2, s**2 + 10*s + 100, s),
            "nome": "Passa-alta",
            "cor": "red"
        },
        {
            "tf": SymbolicTransferFunction(10*s, s**2 + 10*s + 100, s),
            "nome": "Passa-banda",
            "cor": "green"
        }
    ]
    
    interface = NumericalInterface()
    omega = np.logspace(-1, 3, 1000)  # 0.1 to 1000 rad/s
    
    plt.figure(figsize=(12, 6))
    
    for sys in systems:
        # Converter para numérico
        tf_numeric = interface.symbolic_to_control_tf(sys["tf"])
        
        # Calcular resposta em frequência
        freq, mag, phase = interface.compute_frequency_response(tf_numeric, omega)
        
        # Plotar magnitude
        plt.subplot(1, 2, 1)
        plt.semilogx(freq, 20*np.log10(np.abs(mag)), 
                    label=sys["nome"], color=sys["cor"], linewidth=2)
        
        # Plotar fase
        plt.subplot(1, 2, 2)
        plt.semilogx(freq, np.angle(mag)*180/np.pi, 
                    label=sys["nome"], color=sys["cor"], linewidth=2)
        
        print(f"✓ {sys['nome']}: Sistema analisado")
    
    # Configurar gráficos
    plt.subplot(1, 2, 1)
    plt.title('Diagrama de Bode - Magnitude')
    plt.xlabel('Frequência (rad/s)')
    plt.ylabel('Magnitude (dB)')
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.title('Diagrama de Bode - Fase')
    plt.xlabel('Frequência (rad/s)')
    plt.ylabel('Fase (graus)')
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('demo_frequency_analysis.png', dpi=300, bbox_inches='tight')
    print(f"📈 Gráfico salvo: demo_frequency_analysis.png")
    
    return systems

def demo_stability_analysis():
    """Demonstra análise de estabilidade"""
    print("\n" + "="*60)
    print("DEMO 4: Análise de Estabilidade")
    print("="*60)
    
    # Sistema de controle com ganho variável
    s = sp.Symbol('s')
    K = sp.Symbol('K', real=True, positive=True)
    
    # Sistema em malha fechada: G(s) = K / (s^3 + 6s^2 + 11s + 6 + K)
    tf_closed = SymbolicTransferFunction(K, s**3 + 6*s**2 + 11*s + 6 + K, s)
    
    print(f"📊 Sistema em Malha Fechada: {tf_closed}")
    
    interface = NumericalInterface()
    validator = NumericalValidator()
    
    # Testar diferentes ganhos
    gains = [0.1, 1, 5, 10, 20, 50]
    
    print(f"\n🔍 Análise de Estabilidade:")
    print(f"{'Ganho K':<10} {'Estável':<8} {'Polos':<50}")
    print("-"*70)
    
    stable_gains = []
    unstable_gains = []
    
    for gain in gains:
        # Substituir ganho
        substitutions = {K: gain}
        tf_numeric = interface.symbolic_to_control_tf(tf_closed, substitutions)
        
        # Obter polos
        poles = tf_numeric.poles()
        
        # Verificar estabilidade
        stability = validator.check_stability_numerical(poles)
        
        # Formatação dos polos
        poles_str = ", ".join([f"{p:.2f}" for p in poles[:3]])  # Primeiros 3 polos
        if len(poles) > 3:
            poles_str += "..."
        
        is_stable = stability['is_stable']
        status = "✓ Sim" if is_stable else "✗ Não"
        
        print(f"{gain:<10.1f} {status:<8} {poles_str:<50}")
        
        if is_stable:
            stable_gains.append(gain)
        else:
            unstable_gains.append(gain)
    
    print(f"\n📊 Resumo:")
    print(f"   Ganhos Estáveis: {stable_gains}")
    print(f"   Ganhos Instáveis: {unstable_gains}")
    
    if stable_gains and unstable_gains:
        critical_gain = min(unstable_gains)
        print(f"   Ganho Crítico: ~{critical_gain}")
    
    return gains, stable_gains, unstable_gains

def demo_system_factory():
    """Demonstra uso da factory com cache"""
    print("\n" + "="*60)
    print("DEMO 5: Factory e Cache")
    print("="*60)
    
    factory = NumericalSystemFactory()
    s = sp.Symbol('s')
    
    # Criar vários sistemas
    systems = [
        SymbolicTransferFunction(1, s + 1, s),
        SymbolicTransferFunction(1, s**2 + 2*s + 1, s),
        SymbolicTransferFunction(10, s**2 + 10*s + 100, s),
        SymbolicTransferFunction(1, s + 1, s)  # Repetido para testar cache
    ]
    
    print("🏭 Criando sistemas via factory...")
    
    for i, tf in enumerate(systems):
        # Criar sistema (primeira vez será computado, repetições vêm do cache)
        numeric_tf = factory.create_tf_system(tf)
        print(f"   Sistema {i+1}: {numeric_tf}")
    
    # Estatísticas do cache
    stats = factory.get_cache_stats()
    print(f"\n📊 Estatísticas do Cache:")
    print(f"   Sistemas em cache: {stats['cached_systems']}")
    print(f"   Uso do cache: {stats['cache_usage']:.1%}")
    print(f"   Chaves de cache: {len(stats['cache_keys'])}")
    
    return factory, stats

def main():
    """Executa todas as demonstrações"""
    print("🚀 ControlLab - Demo de Integração Simbólico-Numérico")
    print("=" * 80)
    
    try:
        # Executar demos
        demo_basic_conversion()
        demo_parametric_analysis()
        demo_frequency_analysis()
        demo_stability_analysis()
        demo_system_factory()
        
        print("\n" + "="*60)
        print("✅ TODAS AS DEMONSTRAÇÕES CONCLUÍDAS COM SUCESSO!")
        print("="*60)
        print("\n📁 Arquivos gerados:")
        print("   • demo_parametric_analysis.png")
        print("   • demo_frequency_analysis.png")
        print("\n🎯 Recursos demonstrados:")
        print("   ✓ Conversão simbólico → numérico")
        print("   ✓ Análise paramétrica")
        print("   ✓ Resposta em frequência")
        print("   ✓ Análise de estabilidade")
        print("   ✓ Cache e factory pattern")
        print("   ✓ Integração completa dos módulos")
        
    except Exception as e:
        print(f"\n❌ Erro durante a demonstração: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
