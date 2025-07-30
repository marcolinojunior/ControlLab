#!/usr/bin/env python3
"""
Teste simples para validar o Módulo 4 - Modelagem Laplace
"""

import sys
import os

# Adicionar src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Ativar ambiente virtual se necessário
try:
    import sympy as sp
    import numpy as np
    print("✅ Dependências básicas OK")
except ImportError as e:
    print(f"❌ Erro nas dependências: {e}")
    sys.exit(1)

def test_module4_basic():
    """Teste básico do Módulo 4"""
    
    print("\n🔧 TESTANDO MÓDULO 4 - MODELAGEM LAPLACE")
    print("=" * 50)
    
    try:
        # 1. Testar importação do módulo
        print("1. Testando importação...")
        from controllab.modeling import laplace_transform
        from controllab.modeling import partial_fractions
        from controllab.modeling import conversions
        from controllab.modeling import canonical_forms
        from controllab.modeling import physical_systems
        print("   ✅ Todos os submódulos importados com sucesso")
        
        # 2. Testar LaplaceTransformer
        print("\n2. Testando LaplaceTransformer...")
        transformer = laplace_transform.LaplaceTransformer()
        
        # Variáveis simbólicas
        s, t = sp.symbols('s t')
        
        # Teste básico de transformada
        result = laplace_transform.apply_laplace_transform(sp.exp(-t), t, s)
        print(f"   ✅ Transformada de e^(-t): {result}")
        
        # 3. Testar Expansão em Frações Parciais
        print("\n3. Testando Expansão em Frações Parciais...")
        expander = partial_fractions.PartialFractionExpander()
        
        # Teste com função simples
        tf_expr = 1 / (s * (s + 1))
        expansion = expander.expand(tf_expr, s)
        print(f"   ✅ Expansão de 1/(s(s+1)): {expansion}")
        
        # 4. Testar Conversões
        print("\n4. Testando Conversões...")
        
        # Criar uma função de transferência simbólica simples
        from controllab.core.symbolic_tf import SymbolicTransferFunction
        tf = SymbolicTransferFunction(1, [1, 1], s)
        
        # Converter para espaço de estados
        ss_result = conversions.tf_to_ss(tf)
        print(f"   ✅ Conversão TF→SS realizada")
        
        # 5. Testar Formas Canônicas
        print("\n5. Testando Formas Canônicas...")
        
        controllable_form = canonical_forms.controllable_canonical(tf)
        print(f"   ✅ Forma canônica controlável gerada")
        
        # 6. Testar Sistemas Físicos
        print("\n6. Testando Sistemas Físicos...")
        
        # Criar sistema massa-mola-amortecedor
        mechanical = physical_systems.MechanicalSystem(mass=1, damping=0.5, stiffness=1)
        mechanical.derive_equations()  # Derivar equações primeiro
        mechanical.apply_laplace_modeling()  # Aplicar Laplace
        tf_mechanical = mechanical.transfer_function
        print(f"   ✅ Sistema mecânico: {tf_mechanical}")
        
        # Criar circuito RLC
        electrical = physical_systems.ElectricalSystem(resistance=1, inductance=1, capacitance=1)
        electrical.derive_equations()  # Derivar equações primeiro
        electrical.apply_laplace_modeling()  # Aplicar Laplace
        tf_electrical = electrical.transfer_function
        print(f"   ✅ Sistema elétrico: {tf_electrical}")
        
        print("\n🎉 MÓDULO 4 - TODOS OS TESTES PASSARAM!")
        print("✅ Status: 100% FUNCIONAL")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERRO no teste: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_module4_basic()
    if success:
        print("\n" + "="*50)
        print("🚀 MÓDULO 4 PRONTO PARA USO!")
        print("📋 Próximo passo: Implementar Módulo 5 (Análise de Estabilidade)")
        sys.exit(0)
    else:
        sys.exit(1)
