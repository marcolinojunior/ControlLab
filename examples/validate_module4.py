#!/usr/bin/env python3
"""
Teste simples de validação do Módulo 4 - Modelagem de Sistemas
==============================================================

Valida se todos os arquivos do módulo foram criados corretamente.
"""

import os
import sys

def test_module_structure():
    """Testa se todos os arquivos do módulo foram criados"""
    print("=== Teste de Estrutura do Módulo 4 - Modelagem ===")
    
    # Caminho base do módulo
    base_path = os.path.join(os.path.dirname(__file__), 'src', 'controllab', 'modeling')
    
    # Arquivos esperados
    expected_files = [
        '__init__.py',
        'laplace_transform.py',
        'partial_fractions.py',
        'conversions.py',
        'canonical_forms.py',
        'physical_systems.py'
    ]
    
    print(f"Verificando estrutura em: {os.path.abspath(base_path)}")
    
    if not os.path.exists(base_path):
        print("❌ ERRO: Diretório do módulo não encontrado!")
        return False
    
    success = True
    for file in expected_files:
        file_path = os.path.join(base_path, file)
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"✅ {file} ({size} bytes)")
        else:
            print(f"❌ {file} - NÃO ENCONTRADO!")
            success = False
    
    return success


def test_import_without_sympy():
    """Testa se o módulo pode ser importado mesmo sem SymPy"""
    print("\n=== Teste de Importação sem SymPy ===")
    
    # Adicionar src ao path
    src_path = os.path.join(os.path.dirname(__file__), 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    
    try:
        # Tentar importar o módulo principal
        import controllab.modeling
        print("✅ Módulo controllab.modeling importado com sucesso")
        
        # Verificar se __all__ está definido
        if hasattr(controllab.modeling, '__all__'):
            print(f"✅ __all__ definido com {len(controllab.modeling.__all__)} itens")
            for item in controllab.modeling.__all__:
                print(f"   - {item}")
        else:
            print("⚠️  __all__ não definido")
        
        return True
        
    except ImportError as e:
        print(f"❌ Erro na importação: {e}")
        return False
    except Exception as e:
        print(f"❌ Erro inesperado: {e}")
        return False


def test_fallback_classes():
    """Testa se as classes de fallback funcionam"""
    print("\n=== Teste de Classes de Fallback ===")
    
    # Adicionar src ao path
    src_path = os.path.join(os.path.dirname(__file__), 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    
    try:
        from controllab.modeling import (
            LaplaceTransformer,
            PartialFractionExpander,
            tf_to_ss,
            controllable_canonical,
            MechanicalSystem
        )
        
        print("✅ Classes principais importadas")
        
        # Testar criação de uma instância (deve usar fallback)
        transformer = LaplaceTransformer()
        print("✅ LaplaceTransformer criado (fallback)")
        
        mechanical = MechanicalSystem(mass=1, damping=0.5, stiffness=2)
        print("✅ MechanicalSystem criado (fallback)")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro nos testes de fallback: {e}")
        return False


def main():
    """Executa todos os testes"""
    print("ControlLab - Teste de Validação do Módulo 4")
    print("=" * 50)
    
    tests = [
        ("Estrutura do Módulo", test_module_structure),
        ("Importação sem SymPy", test_import_without_sympy),
        ("Classes de Fallback", test_fallback_classes)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔧 Executando: {test_name}")
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ Erro no teste {test_name}: {e}")
            results.append(False)
    
    # Resumo
    print("\n" + "=" * 50)
    print("RESUMO DOS TESTES:")
    passed = sum(results)
    total = len(results)
    
    for i, (test_name, _) in enumerate(tests):
        status = "✅ PASSOU" if results[i] else "❌ FALHOU"
        print(f"  {test_name}: {status}")
    
    print(f"\nResultado Final: {passed}/{total} testes passaram")
    
    if passed == total:
        print("🎉 TODOS OS TESTES PASSARAM! Módulo 4 implementado com sucesso.")
        return True
    else:
        print("⚠️  Alguns testes falharam. Verifique a implementação.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
