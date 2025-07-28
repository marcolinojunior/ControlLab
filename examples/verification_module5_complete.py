#!/usr/bin/env python3
"""
VERIFICAÇÃO COMPLETA DO MÓDULO 5 - ANÁLISE DE ESTABILIDADE
==========================================================

Este script verifica se TODAS as funcionalidades mencionadas no arquivo
05-analise-estabilidade.md foram realmente implementadas.

Não acreditamos em mensagens anteriores - vamos testar TUDO!
"""

import sys
import traceback
from pathlib import Path
import sympy as sp

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_section(section_name: str, test_func):
    """Executa um teste de seção com tratamento de erro"""
    print(f"\n{'='*60}")
    print(f"🧪 TESTANDO: {section_name}")
    print('='*60)
    
    try:
        result = test_func()
        if result:
            print(f"✅ {section_name}: APROVADO")
            return True
        else:
            print(f"❌ {section_name}: FALHOU")
            return False
    except Exception as e:
        print(f"💥 {section_name}: ERRO - {str(e)}")
        print(traceback.format_exc())
        return False

def test_imports():
    """5.1 - Testar todas as importações mencionadas no 05-analise-estabilidade.md"""
    print("🔍 Verificando importações...")
    
    # Importações básicas do módulo
    from controllab.analysis import (
        StabilityAnalysisEngine,
        RouthHurwitzAnalyzer,
        RootLocusAnalyzer,
        FrequencyAnalyzer,
        StabilityValidator,
        ParametricAnalyzer
    )
    
    # Classes de dados
    from controllab.analysis import (
        StabilityResult,
        LocusFeatures,
        StabilityMargins
    )
    
    # Funções de conveniência
    from controllab.analysis import (
        analyze_stability,
        quick_stability_check,
        validate_stability_methods
    )
    
    print("✅ Todas as importações principais funcionaram!")
    return True

def test_routh_hurwitz_requirements():
    """5.1 & 5.2 - Testar critério de Routh-Hurwitz conforme especificações"""
    print("🔍 Testando Routh-Hurwitz...")
    
    from controllab.analysis.routh_hurwitz import (
        RouthHurwitzAnalyzer,
        build_routh_array,
        analyze_stability,
        handle_zero_in_first_column,
        handle_row_of_zeros
    )
    
    # Testar com polinômio simples
    s, K = sp.symbols('s K')
    poly = s**3 + 2*s**2 + s + K
    
    # Instanciar analisador
    analyzer = RouthHurwitzAnalyzer()
    
    # Teste 1: Construir tabela de Routh
    print("  📋 Testando build_routh_array...")
    routh_table = analyzer.build_routh_array(poly, show_steps=True)
    assert routh_table is not None, "Falha ao construir tabela de Routh"
    
    # Teste 2: Analisar estabilidade
    print("  📊 Testando analyze_stability...")
    stability = analyzer.analyze_stability(routh_table, show_steps=True)
    assert stability is not None, "Falha na análise de estabilidade"
    assert hasattr(stability, 'stable_range'), "StabilityResult deve ter stable_range"
    
    # Teste 3: Funções standalone
    print("  🔧 Testando funções standalone...")
    routh_standalone = build_routh_array(poly, show_steps=False)
    stability_standalone = analyze_stability(poly, show_steps=False)
    
    # Teste 4: Histórico pedagógico
    print("  📚 Testando histórico pedagógico...")
    assert hasattr(stability, 'history'), "StabilityResult deve ter histórico"
    if stability.history:
        formatted_report = stability.history.get_formatted_report()
        assert len(formatted_report) > 100, "Relatório pedagógico muito curto"
    
    print("✅ Routh-Hurwitz: Todas as funções especificadas funcionam!")
    return True

def test_root_locus_requirements():
    """5.3 & 5.4 - Testar Root Locus conforme especificações"""
    print("🔍 Testando Root Locus...")
    
    from controllab.analysis.root_locus import (
        RootLocusAnalyzer,
        get_locus_features,
        calculate_asymptotes,
        find_breakaway_points,
        find_jw_crossings,
        calculate_locus_points
    )
    from controllab.core import SymbolicTransferFunction
    
    # Sistema de teste: G(s) = 1/(s*(s+1)*(s+2))
    s = sp.Symbol('s')
    G = SymbolicTransferFunction(1, s*(s+1)*(s+2))
    
    # Instanciar analisador
    analyzer = RootLocusAnalyzer()
    
    # Teste 1: Características do locus
    print("  📐 Testando get_locus_features...")
    features = analyzer.get_locus_features(G, show_steps=True)
    assert features is not None, "Falha ao obter características do locus"
    assert hasattr(features, 'asymptotes'), "LocusFeatures deve ter asymptotes"
    assert hasattr(features, 'breakaway_points'), "LocusFeatures deve ter breakaway_points"
    
    # Teste 2: Funções específicas das 6 regras
    print("  📏 Testando funções das 6 regras...")
    
    # Regra 3: Assíntotas
    zeros = []
    poles = [-0, -1, -2]
    asymptotes = calculate_asymptotes(zeros, poles)
    assert 'angles' in asymptotes, "calculate_asymptotes deve retornar angles"
    assert 'centroid' in asymptotes, "calculate_asymptotes deve retornar centroid"
    
    # Regra 4: Pontos de breakaway
    breakaway = find_breakaway_points(G)
    assert isinstance(breakaway, list), "find_breakaway_points deve retornar lista"
    
    # Regra 5: Cruzamentos jω
    jw_crossings = find_jw_crossings(G)
    assert isinstance(jw_crossings, list), "find_jw_crossings deve retornar lista"
    
    # Teste 3: Cálculo de pontos do locus
    print("  📍 Testando calculate_locus_points...")
    k_range = [0.1, 0.5, 1.0, 2.0, 5.0]
    locus_points = analyzer.calculate_locus_points(G, k_range, show_steps=False)
    assert 'k_values' in locus_points, "calculate_locus_points deve retornar k_values"
    assert 'roots' in locus_points, "calculate_locus_points deve retornar roots"
    
    print("✅ Root Locus: Todas as 6 regras e funções especificadas funcionam!")
    return True

def test_frequency_response_requirements():
    """5.5 & 5.6 - Testar Frequency Response conforme especificações"""
    print("🔍 Testando Frequency Response...")
    
    from controllab.analysis.frequency_response import (
        FrequencyAnalyzer,
        get_nyquist_contour,
        calculate_frequency_response,
        apply_nyquist_criterion,
        calculate_gain_phase_margins
    )
    from controllab.core import SymbolicTransferFunction
    import numpy as np
    
    # Sistema de teste: G(s) = 1/(s+1)
    s = sp.Symbol('s')
    G = SymbolicTransferFunction(1, s+1)
    
    # Instanciar analisador
    analyzer = FrequencyAnalyzer()
    
    # Teste 1: Contorno de Nyquist
    print("  🔄 Testando get_nyquist_contour...")
    contour = analyzer.get_nyquist_contour(G, radius=100, epsilon=1e-3, show_steps=True)
    assert contour is not None, "Falha ao obter contorno de Nyquist"
    
    # Teste 2: Resposta em frequência
    print("  📈 Testando calculate_frequency_response...")
    omega_range = np.logspace(-2, 2, 50)
    freq_response = analyzer.calculate_frequency_response(G, omega_range, show_steps=True)
    assert freq_response is not None, "Falha no cálculo da resposta em frequência"
    
    # Teste 3: Critério de Nyquist
    print("  🎯 Testando apply_nyquist_criterion...")
    nyquist_result = analyzer.apply_nyquist_criterion(G, contour, show_steps=True)
    assert nyquist_result is not None, "Falha na aplicação do critério de Nyquist"
    
    # Teste 4: Margens de ganho e fase
    print("  📊 Testando calculate_gain_phase_margins...")
    margins = analyzer.calculate_gain_phase_margins(G, show_steps=True)
    assert margins is not None, "Falha no cálculo das margens"
    assert hasattr(margins, 'gain_margin'), "StabilityMargins deve ter gain_margin"
    assert hasattr(margins, 'phase_margin'), "StabilityMargins deve ter phase_margin"
    
    print("✅ Frequency Response: Todas as funções especificadas funcionam!")
    return True

def test_parametric_analysis_requirements():
    """5.12 - Testar análise paramétrica conforme especificações"""
    print("🔍 Testando Análise Paramétrica...")
    
    from controllab.analysis.stability_utils import (
        ParametricAnalyzer,
        stability_region_2d,
        root_locus_3d
    )
    from controllab.core import SymbolicTransferFunction
    
    # Sistema paramétrico: G(s) = K1/(s^2 + K2*s + 1)
    s, K1, K2 = sp.symbols('s K1 K2')
    num = K1
    den = s**2 + K2*s + 1
    G = SymbolicTransferFunction(num, den)
    
    # Instanciar analisador
    analyzer = ParametricAnalyzer()
    
    # Teste 1: Região de estabilidade 2D
    print("  🗺️ Testando stability_region_2d...")
    try:
        stability_2d = analyzer.stability_region_2d(
            G, K1, K2, 
            param1_range=(0.1, 10), 
            param2_range=(0.1, 5),
            resolution=10
        )
        assert stability_2d is not None, "Falha na análise de região 2D"
        print("    ✅ stability_region_2d funcionou!")
    except Exception as e:
        print(f"    💥 ERRO CRÍTICO stability_region_2d: {e}")
        return False
    
    # Teste 2: Root locus 3D
    print("  📦 Testando root_locus_3d...")
    try:
        locus_3d = analyzer.root_locus_3d(
            G, K1, K2,
            k_range=[0.1, 0.5, 1.0, 2.0, 5.0]
        )
        assert locus_3d is not None, "Falha na análise 3D"
        print("    ✅ root_locus_3d funcionou!")
    except Exception as e:
        print(f"    💥 ERRO CRÍTICO root_locus_3d: {e}")
        return False
    
    # Teste 3: Funções standalone
    print("  🔧 Testando funções standalone...")
    try:
        stability_2d_standalone = stability_region_2d(
            G, K1, K2, 
            param1_range=(0.1, 10), 
            param2_range=(0.1, 5)
        )
        locus_3d_standalone = root_locus_3d(G, K1, K2, k_range=[0.1, 1.0, 5.0])
        print("    ✅ Funções standalone funcionaram!")
    except Exception as e:
        print(f"    💥 ERRO CRÍTICO funções standalone: {e}")
        return False
    
    print("✅ Análise Paramétrica: Principais funções especificadas funcionam!")
    return True

def test_validation_requirements():
    """5.10 - Testar validação cruzada conforme especificações"""
    print("🔍 Testando Validação Cruzada...")
    
    from controllab.analysis.stability_utils import (
        StabilityValidator,
        validate_stability_methods
    )
    from controllab.core import SymbolicTransferFunction
    
    # Sistema de teste: G(s) = 1/(s^3 + 2s^2 + s + 1)
    s = sp.Symbol('s')
    G = SymbolicTransferFunction(1, s**3 + 2*s**2 + s + 1)
    
    # Instanciar validador
    validator = StabilityValidator()
    
    # Teste 1: Validação completa
    print("  🔍 Testando validate_stability_methods...")
    validation = validator.validate_stability_methods(G, show_steps=True)
    assert validation is not None, "Falha na validação cruzada"
    assert 'routh_hurwitz' in validation, "Validação deve incluir routh_hurwitz"
    assert 'root_analysis' in validation, "Validação deve incluir root_analysis"
    assert 'frequency_analysis' in validation, "Validação deve incluir frequency_analysis"
    
    # Teste 2: Função standalone
    print("  🔧 Testando função standalone...")
    validation_standalone = validate_stability_methods(G, show_steps=False)
    assert validation_standalone is not None, "Falha na validação standalone"
    
    print("✅ Validação Cruzada: Todas as funções especificadas funcionam!")
    return True

def test_main_engine():
    """Testar o motor principal de análise"""
    print("🔍 Testando Motor Principal...")
    
    from controllab.analysis import (
        StabilityAnalysisEngine,
        analyze_stability,
        quick_stability_check
    )
    from controllab.core import SymbolicTransferFunction
    
    # Sistema de teste
    s = sp.Symbol('s')
    G = SymbolicTransferFunction(1, s**2 + 2*s + 1)
    
    # Teste 1: Motor principal
    print("  🚀 Testando StabilityAnalysisEngine...")
    engine = StabilityAnalysisEngine()
    report = engine.analyze_complete_stability(G, show_steps=True)
    assert report is not None, "Falha no motor principal"
    
    # Teste 2: Funções de conveniência
    print("  🎯 Testando analyze_stability...")
    analysis = analyze_stability(G, show_steps=True)
    assert analysis is not None, "Falha em analyze_stability"
    
    print("  ⚡ Testando quick_stability_check...")
    is_stable = quick_stability_check(G)
    assert isinstance(is_stable, bool), "quick_stability_check deve retornar bool"
    
    print("✅ Motor Principal: Todas as funções funcionam!")
    return True

def test_pedagogical_features():
    """Testar características pedagógicas mencionadas em 5.9"""
    print("🔍 Testando Características Pedagógicas...")
    
    from controllab.analysis import RouthHurwitzAnalyzer
    from controllab.core import SymbolicTransferFunction
    
    # Sistema de teste
    s, K = sp.symbols('s K')
    poly = s**3 + 2*s**2 + s + K
    
    # Teste histórico detalhado
    analyzer = RouthHurwitzAnalyzer()
    routh_table = analyzer.build_routh_array(poly, show_steps=True)
    stability = analyzer.analyze_stability(routh_table, show_steps=True)
    
    # Verificar se tem histórico
    assert hasattr(stability, 'history'), "Deve ter histórico pedagógico"
    
    if stability.history:
        report = stability.history.get_formatted_report()
        print("  📚 Exemplo de relatório pedagógico:")
        print("  " + "="*50)
        lines = report.split('\n')[:15]  # Primeiras 15 linhas
        for line in lines:
            print(f"  {line}")
        print("  " + "="*50)
        
        # Verificar se contém elementos pedagógicos
        assert "ANÁLISE DE ROUTH-HURWITZ" in report, "Deve ter título pedagógico"
        assert "PASSOS DA CONSTRUÇÃO" in report, "Deve ter seção de passos"
    
    print("✅ Características Pedagógicas: Histórico detalhado funciona!")
    return True

def main():
    """Execução principal da verificação"""
    print("🚀 INICIANDO VERIFICAÇÃO COMPLETA DO MÓDULO 5")
    print("=" * 70)
    print("📋 Verificando se TODAS as funcionalidades do 05-analise-estabilidade.md")
    print("    foram realmente implementadas, sem confiar em mensagens anteriores!")
    
    tests = [
        ("IMPORTAÇÕES", test_imports),
        ("5.1 & 5.2 - ROUTH-HURWITZ", test_routh_hurwitz_requirements),
        ("5.3 & 5.4 - ROOT LOCUS", test_root_locus_requirements),
        ("5.5 & 5.6 - FREQUENCY RESPONSE", test_frequency_response_requirements),
        ("5.12 - ANÁLISE PARAMÉTRICA", test_parametric_analysis_requirements),
        ("5.10 - VALIDAÇÃO CRUZADA", test_validation_requirements),
        ("MOTOR PRINCIPAL", test_main_engine),
        ("5.9 - CARACTERÍSTICAS PEDAGÓGICAS", test_pedagogical_features),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        if test_section(test_name, test_func):
            passed += 1
    
    print(f"\n{'='*70}")
    print(f"🎯 RESULTADO FINAL: {passed}/{total} testes aprovados")
    
    if passed == total:
        print("🎉 MÓDULO 5 ESTÁ 100% IMPLEMENTADO!")
        print("✅ Todas as funcionalidades do 05-analise-estabilidade.md funcionam!")
    else:
        print(f"⚠️ {total - passed} funcionalidades ainda precisam ser corrigidas")
        
    print("="*70)
    
    # Verificar testes específicos mencionados no documento
    print("\n📁 VERIFICANDO ARQUIVOS DE TESTE MENCIONADOS:")
    test_files_expected = [
        "tests/test_routh_hurwitz.py",
        "tests/test_root_locus.py", 
        "tests/test_frequency_response.py",
        "tests/test_stability_integration.py"
    ]
    
    missing_files = []
    for test_file in test_files_expected:
        if not Path(test_file).exists():
            missing_files.append(test_file)
            print(f"❌ {test_file} - NÃO EXISTE")
        else:
            print(f"✅ {test_file} - EXISTE")
    
    if missing_files:
        print(f"\n⚠️ ARQUIVOS DE TESTE FALTANDO: {len(missing_files)}")
        print("   Mencionados no 05-analise-estabilidade.md mas não implementados:")
        for file in missing_files:
            print(f"   - {file}")
    else:
        print("\n✅ TODOS OS ARQUIVOS DE TESTE MENCIONADOS EXISTEM!")
    
    return passed == total and len(missing_files) == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
