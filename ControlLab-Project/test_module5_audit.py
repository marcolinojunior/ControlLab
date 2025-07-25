#!/usr/bin/env python3
"""
Auditoria segura do Módulo 5 - Análise de Estabilidade
Testa apenas importações e estrutura, sem executar análises que podem entrar em loop
"""

import sys
sys.path.insert(0, 'src')

def audit_module5():
    """Auditoria básica sem executar código que pode entrar em loop"""
    
    print("🔧 AUDITORIA MÓDULO 5 - ANÁLISE DE ESTABILIDADE")
    print("=" * 60)
    
    errors = []
    successes = []
    
    # 1. Teste de importações básicas
    print("1. Testando importações básicas...")
    try:
        import src.controllab.analysis
        successes.append("✅ Módulo analysis importado")
    except Exception as e:
        errors.append(f"❌ Erro na importação do módulo: {e}")
    
    # 2. Teste de classes principais
    print("2. Testando classes principais...")
    try:
        from src.controllab.analysis.stability_analysis import StabilityAnalysisEngine
        from src.controllab.analysis.routh_hurwitz import RouthHurwitzAnalyzer
        from src.controllab.analysis.root_locus import RootLocusAnalyzer  
        successes.append("✅ Classes principais importadas")
    except Exception as e:
        errors.append(f"❌ Erro na importação das classes: {e}")
    
    # 3. Teste de funções de conveniência
    print("3. Testando funções de conveniência...")
    try:
        from src.controllab.analysis import analyze_stability, quick_stability_check
        successes.append("✅ Funções de conveniência importadas")
    except Exception as e:
        errors.append(f"❌ Erro na importação das funções: {e}")
    
    # 4. Verificação de métodos sem execução
    print("4. Verificando métodos disponíveis...")
    try:
        from src.controllab.analysis.routh_hurwitz import RouthHurwitzAnalyzer
        analyzer = RouthHurwitzAnalyzer()
        
        # Verificar se métodos existem (sem executar)
        methods = ['build_routh_table', 'analyze_polynomial', 'get_stability_info']
        existing_methods = [method for method in methods if hasattr(analyzer, method)]
        
        successes.append(f"✅ Métodos encontrados: {existing_methods}")
    except Exception as e:
        errors.append(f"❌ Erro na verificação dos métodos: {e}")
    
    # 5. Verificação de dependências
    print("5. Verificando dependências...")
    try:
        import sympy as sp
        import numpy as np
        successes.append("✅ Dependências críticas (SymPy, NumPy) disponíveis")
    except Exception as e:
        errors.append(f"❌ Erro nas dependências: {e}")
    
    # Resumo da auditoria
    print("\n" + "=" * 60)
    print("📊 RESUMO DA AUDITORIA:")
    print(f"✅ Sucessos: {len(successes)}")
    print(f"❌ Erros: {len(errors)}")
    
    for success in successes:
        print(f"  {success}")
    
    if errors:
        print("\n🔴 PROBLEMAS ENCONTRADOS:")
        for error in errors:
            print(f"  {error}")
        print("\n🛠️  AÇÃO NECESSÁRIA: Corrigir problemas antes de usar o módulo")
        return False
    else:
        print("\n🎉 MÓDULO 5 PASSOU NA AUDITORIA!")
        print("✅ Status: Pronto para testes funcionais")
        return True

if __name__ == "__main__":
    audit_module5()
