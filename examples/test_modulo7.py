"""
Teste de Importação do Módulo 7 - Sistemas Discretos
===================================================

Testa se todos os módulos do Módulo 7 podem ser importados corretamente.
"""

def test_module7_imports():
    """Testa importações do módulo 7"""
    print("🔄 TESTANDO IMPORTAÇÕES DO MÓDULO 7")
    print("=" * 40)
    
    try:
        # Teste 1: Transformada Z
        print("📦 Testando z_transform...")
        from src.controllab.modeling.z_transform import ZTransformer, ZTransformResult
        print("   ✅ z_transform importado com sucesso")
        
        # Teste 2: Discretização
        print("📦 Testando discretization...")
        from src.controllab.modeling.discretization import DiscretizationMethods, DiscretizationResult
        print("   ✅ discretization importado com sucesso")
        
        # Teste 3: Estabilidade discreta
        print("📦 Testando discrete_stability...")
        from src.controllab.modeling.discrete_stability import DiscreteStabilityAnalyzer, StabilityResult
        print("   ✅ discrete_stability importado com sucesso")
        
        # Teste 4: Lugar das raízes discreto
        print("📦 Testando discrete_root_locus...")
        from src.controllab.modeling.discrete_root_locus import DiscreteRootLocus, DiscreteRootLocusResult
        print("   ✅ discrete_root_locus importado com sucesso")
        
        # Teste 5: Resposta em frequência discreta
        print("📦 Testando discrete_frequency_response...")
        from src.controllab.modeling.discrete_frequency_response import DiscreteFrequencyAnalyzer, DiscreteFrequencyResult
        print("   ✅ discrete_frequency_response importado com sucesso")
        
        # Teste 6: Importação pelo __init__
        print("📦 Testando importação via modeling.__init__...")
        from src.controllab.modeling import (
            ZTransformer, apply_z_transform,
            DiscretizationMethods, compare_discretization_methods,
            analyze_discrete_stability, plot_discrete_root_locus,
            analyze_discrete_frequency_response
        )
        print("   ✅ Importação via __init__ bem-sucedida")
        
        print("\n✅ TODOS OS TESTES DE IMPORTAÇÃO PASSARAM!")
        print("🎉 Módulo 7 pronto para uso!")
        return True
        
    except ImportError as e:
        print(f"\n❌ Erro de importação: {e}")
        print("⚠️  Algumas dependências podem estar faltando")
        return False
    except Exception as e:
        print(f"\n❌ Erro inesperado: {e}")
        return False

def test_basic_functionality():
    """Testa funcionalidade básica sem SymPy"""
    print("\n🔄 TESTANDO FUNCIONALIDADE BÁSICA")
    print("=" * 35)
    
    try:
        # Teste de criação de objetos básicos
        from src.controllab.modeling.z_transform import ZTransformer
        transformer = ZTransformer()
        print("   ✅ ZTransformer criado")
        
        from src.controllab.modeling.discretization import DiscretizationMethods
        discretizer = DiscretizationMethods(0.1)
        print("   ✅ DiscretizationMethods criado")
        
        from src.controllab.modeling.discrete_stability import DiscreteStabilityAnalyzer
        analyzer = DiscreteStabilityAnalyzer()
        print("   ✅ DiscreteStabilityAnalyzer criado")
        
        print("\n✅ FUNCIONALIDADE BÁSICA OK!")
        return True
        
    except Exception as e:
        print(f"\n❌ Erro na funcionalidade básica: {e}")
        return False

def main():
    """Executa todos os testes"""
    print("🧪 TESTE DO MÓDULO 7 - SISTEMAS DISCRETOS")
    print("=" * 45)
    print("Verificando se o módulo foi implementado corretamente...\n")
    
    # Executar testes
    test1_passed = test_module7_imports()
    test2_passed = test_basic_functionality()
    
    # Resumo
    print(f"\n{'='*50}")
    print("📊 RESUMO DOS TESTES:")
    print(f"   Importações: {'✅ PASSOU' if test1_passed else '❌ FALHOU'}")
    print(f"   Funcionalidade: {'✅ PASSOU' if test2_passed else '❌ FALHOU'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 MÓDULO 7 IMPLEMENTADO COM SUCESSO!")
        print("📚 Pronto para demonstrações e uso pedagógico")
    else:
        print("\n⚠️  ALGUNS PROBLEMAS DETECTADOS")
        print("🔧 Verificar dependências e implementação")

if __name__ == "__main__":
    main()
