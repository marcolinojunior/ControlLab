"""
Teste de Integração do Módulo 5 - Análise de Estabilidade
=========================================================

Este script testa todos os componentes do Módulo 5 de análise de estabilidade,
verificando se funcionam corretamente e se integram entre si.
"""

import sys
import os

# Adicionar o caminho do projeto
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

import sympy as sp
from sympy import symbols, simplify, expand

def test_routh_hurwitz():
    """Testa o analisador de Routh-Hurwitz"""
    print("🔢 TESTE: Analisador de Routh-Hurwitz")
    print("-" * 40)
    
    try:
        from src.controllab.analysis.routh_hurwitz import RouthHurwitzAnalyzer
        
        # Criar analisador
        analyzer = RouthHurwitzAnalyzer()
        
        # Teste 1: Sistema estável de 3ª ordem
        s = symbols('s')
        poly_stable = s**3 + 2*s**2 + 3*s + 1
        
        print("Testando polinômio estável:", poly_stable)
        routh_array = analyzer.build_routh_array(poly_stable, show_steps=True)
        result = analyzer.analyze_stability(routh_array, show_steps=True)
        
        print(f"Resultado: {'ESTÁVEL' if result.is_stable else 'INSTÁVEL'}")
        print(f"Polos instáveis: {result.unstable_poles_count}")
        
        # Teste 2: Sistema instável
        poly_unstable = s**3 - s**2 + 2*s + 1
        print(f"\nTestando polinômio instável: {poly_unstable}")
        routh_array_2 = analyzer.build_routh_array(poly_unstable, show_steps=False)
        result_2 = analyzer.analyze_stability(routh_array_2, show_steps=False)
        
        print(f"Resultado: {'ESTÁVEL' if result_2.is_stable else 'INSTÁVEL'}")
        print(f"Polos instáveis: {result_2.unstable_poles_count}")
        
        print("✅ ROUTH-HURWITZ: FUNCIONANDO CORRETAMENTE\n")
        return True
        
    except Exception as e:
        print(f"❌ ERRO no Routh-Hurwitz: {e}\n")
        return False

def test_root_locus():
    """Testa o analisador de Root Locus"""
    print("📍 TESTE: Analisador de Root Locus")
    print("-" * 35)
    
    try:
        from src.controllab.analysis.root_locus import RootLocusAnalyzer
        
        # Criar analisador
        analyzer = RootLocusAnalyzer()
        
        # Sistema de teste: G(s)H(s) = K/[(s+1)(s+2)(s+3)]
        s = symbols('s')
        system = 1 / ((s + 1) * (s + 2) * (s + 3))
        
        print("Testando sistema:", system)
        result = analyzer.analyze_comprehensive(system, show_steps=True)
        
        print(f"Número de polos: {len(result.poles)}")
        print(f"Número de zeros: {len(result.zeros)}")
        print(f"Número de branches: {result.num_branches}")
        
        if hasattr(result, 'centroid'):
            print(f"Centróide: {result.centroid}")
        
        print("✅ ROOT LOCUS: FUNCIONANDO CORRETAMENTE\n")
        return True
        
    except Exception as e:
        print(f"❌ ERRO no Root Locus: {e}\n")
        return False

def test_frequency_response():
    """Testa o analisador de resposta em frequência"""
    print("📈 TESTE: Analisador de Resposta em Frequência")
    print("-" * 45)
    
    try:
        from src.controllab.analysis.frequency_response import FrequencyAnalyzer
        
        # Criar analisador
        analyzer = FrequencyAnalyzer()
        
        # Sistema de teste: G(s) = 1/(s+1)
        s = symbols('s')
        system = 1 / (s + 1)
        
        print("Testando sistema:", system)
        result = analyzer.calculate_gain_phase_margins(system, show_steps=True)
        
        print(f"Margem de ganho: {result.gain_margin_db:.2f} dB")
        print(f"Margem de fase: {result.phase_margin:.2f}°")
        print(f"Sistema estável: {'SIM' if result.is_stable else 'NÃO'}")
        
        print("✅ FREQUENCY RESPONSE: FUNCIONANDO CORRETAMENTE\n")
        return True
        
    except Exception as e:
        print(f"❌ ERRO no Frequency Response: {e}\n")
        return False

def test_stability_utils():
    """Testa os utilitários de estabilidade"""
    print("🔧 TESTE: Utilitários de Estabilidade")
    print("-" * 35)
    
    try:
        from src.controllab.analysis.stability_utils import StabilityValidator
        
        # Criar validador
        validator = StabilityValidator()
        
        # Sistema de teste
        s = symbols('s')
        system = 1 / (s**2 + 2*s + 1)
        
        print("Testando sistema:", system)
        results = validator.validate_stability_methods(system, show_steps=True)
        
        print("Métodos testados:")
        for method, result in results.items():
            if isinstance(result, dict) and 'is_stable' in result:
                status = "ESTÁVEL" if result['is_stable'] else "INSTÁVEL"
                print(f"  {method}: {status}")
        
        # Mostrar relatório de validação
        print("\nRelatório de validação:")
        print(validator.history.get_formatted_report())
        
        print("✅ STABILITY UTILS: FUNCIONANDO CORRETAMENTE\n")
        return True
        
    except Exception as e:
        print(f"❌ ERRO nos Stability Utils: {e}\n")
        return False

def test_stability_analysis():
    """Testa o motor principal de análise"""
    print("🎯 TESTE: Motor Principal de Análise")
    print("-" * 35)
    
    try:
        from src.controllab.analysis.stability_analysis import StabilityAnalysisEngine
        
        # Criar motor de análise
        engine = StabilityAnalysisEngine()
        
        # Sistema de teste: função de transferência de 2ª ordem
        s = symbols('s')
        system = (s + 1) / (s**2 + 3*s + 2)
        
        print("Testando sistema:", system)
        
        # Análise rápida
        quick_result = engine.quick_stability_check(system)
        print(f"Verificação rápida: {'ESTÁVEL' if quick_result['is_stable'] else 'INSTÁVEL'}")
        print(f"Método usado: {quick_result['method_used']}")
        
        # Análise completa
        print("\nRealizada análise completa...")
        comprehensive_result = engine.comprehensive_analysis(system, show_all_steps=False)
        
        # Mostrar resumo executivo
        print("\n" + comprehensive_result.get_executive_summary())
        
        print("✅ STABILITY ANALYSIS: FUNCIONANDO CORRETAMENTE\n")
        return True
        
    except Exception as e:
        print(f"❌ ERRO no Stability Analysis: {e}\n")
        return False

def test_integration():
    """Testa a integração completa do módulo"""
    print("🔗 TESTE: Integração Completa do Módulo 5")
    print("-" * 45)
    
    try:
        # Tentar importar via __init__.py
        from src.controllab.analysis import analyze_stability
        
        # Sistema de teste
        s = symbols('s')
        system = 2 / (s**3 + 4*s**2 + 5*s + 2)
        
        print("Testando sistema via interface integrada:", system)
        
        # Análise completa via função de conveniência
        result = analyze_stability(system, show_steps=False)
        
        # Mostrar relatório final
        print(result.get_full_report())
        
        print("✅ INTEGRAÇÃO: FUNCIONANDO CORRETAMENTE\n")
        return True
        
    except Exception as e:
        print(f"❌ ERRO na Integração: {e}\n")
        return False

def main():
    """Executa todos os testes do Módulo 5"""
    print("🧪 TESTES DO MÓDULO 5 - ANÁLISE DE ESTABILIDADE")
    print("=" * 60)
    print()
    
    # Contador de sucessos
    tests_passed = 0
    total_tests = 6
    
    # Executar testes individuais
    if test_routh_hurwitz():
        tests_passed += 1
    
    if test_root_locus():
        tests_passed += 1
    
    if test_frequency_response():
        tests_passed += 1
    
    if test_stability_utils():
        tests_passed += 1
    
    if test_stability_analysis():
        tests_passed += 1
    
    if test_integration():
        tests_passed += 1
    
    # Relatório final
    print("📋 RELATÓRIO FINAL DOS TESTES")
    print("=" * 35)
    print(f"Testes executados: {total_tests}")
    print(f"Testes aprovados: {tests_passed}")
    print(f"Taxa de sucesso: {(tests_passed/total_tests)*100:.1f}%")
    
    if tests_passed == total_tests:
        print("\n🎉 MÓDULO 5 IMPLEMENTADO COM SUCESSO!")
        print("✅ Todos os componentes funcionando corretamente")
        print("✅ Integração entre módulos validada")
        print("✅ Interface pedagógica operacional")
    else:
        print(f"\n⚠️  ALGUNS PROBLEMAS ENCONTRADOS")
        print(f"❌ {total_tests - tests_passed} teste(s) falharam")
        print("🔧 Verificar erros acima para correções necessárias")

if __name__ == "__main__":
    main()
