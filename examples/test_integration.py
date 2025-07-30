#!/usr/bin/env python3
"""
Teste de integração dos módulos de modelagem
"""

import sys
sys.path.append('src')

def test_laplace_integration():
    """Testa integração do módulo de Laplace"""
    print("="*60)
    print("TESTE: Integração Módulo de Laplace")
    print("="*60)
    
    from controllab.modeling.laplace_transform import LaplaceTransformer, from_ode
    import sympy as sp
    
    # Teste 1: Transformada básica
    transformer = LaplaceTransformer()
    result, conv, expr = transformer.transform_function(sp.exp(-2*transformer.t), show_steps=True)
    print("Teste 1 - Transformada exp(-2t):")
    print(f"Resultado: {result}")
    print(f"Condição: Re(s) > {conv}")
    print()
    
    # Teste 2: EDO para TF
    print("Teste 2 - EDO para Função de Transferência:")
    t, s = sp.symbols('t s')
    x = sp.Function('x')(t)
    F = sp.Symbol('F')
    
    # Exemplo: massa-mola-amortecedor
    m, b, k = sp.symbols('m b k', positive=True)
    ode = m*x.diff(t, 2) + b*x.diff(t) + k*x - F
    
    try:
        tf_result = from_ode(ode, x, t, initial_conditions={'x(0)': 0, "x'(0)": 0})
        print(f"EDO: {ode} = 0")
        print(f"TF: {tf_result}")
        print()
    except Exception as e:
        print(f"Erro na conversão EDO->TF: {e}")
        print()

def test_conversion_integration():
    """Testa integração com conversões TF<->SS"""
    print("="*60)
    print("TESTE: Integração Conversões TF<->SS")
    print("="*60)
    
    try:
        from controllab.modeling.conversions import tf_to_ss, ss_to_tf
        from controllab.core.symbolic_tf import SymbolicTransferFunction
        import sympy as sp
        
        # Teste de conversão TF -> SS
        s = sp.Symbol('s')
        G = SymbolicTransferFunction(1, s**2 + 3*s + 2, s)
        
        print("Teste 1 - TF para SS:")
        print(f"G(s) = {G}")
        
        ss_result = tf_to_ss(G, form='controllable')
        print("Forma controlável:")
        print(f"A = {ss_result['A']}")
        print(f"B = {ss_result['B']}")
        print(f"C = {ss_result['C']}")
        print(f"D = {ss_result['D']}")
        print()
        
    except Exception as e:
        print(f"Erro na conversão: {e}")
        print()

def test_physical_systems():
    """Testa sistemas físicos"""
    print("="*60)
    print("TESTE: Sistemas Físicos")
    print("="*60)
    
    try:
        from controllab.modeling.physical_systems import MechanicalSystem, ElectricalSystem
        import sympy as sp
        
        # Teste sistema mecânico
        print("Teste 1 - Sistema Mecânico:")
        m, b, k = sp.symbols('m b k', positive=True)
        mech_sys = MechanicalSystem(mass=m, damping=b, stiffness=k)
        
        equations = mech_sys.derive_equations()
        print(f"Equações: {equations}")
        
        tf_result = mech_sys.apply_laplace_modeling()
        print(f"Função de Transferência: {tf_result}")
        print()
        
    except Exception as e:
        print(f"Erro nos sistemas físicos: {e}")
        print()

def test_partial_fractions():
    """Testa expansão em frações parciais"""
    print("="*60)
    print("TESTE: Frações Parciais")
    print("="*60)
    
    try:
        from controllab.modeling.partial_fractions import explain_partial_fractions
        import sympy as sp
        
        s = sp.Symbol('s')
        tf_expr = 1 / (s*(s+1)*(s+2))
        
        print("Teste - Expansão em frações parciais:")
        print(f"Expressão: {tf_expr}")
        
        expansion = explain_partial_fractions(tf_expr)
        print(f"Expansão: {expansion}")
        print()
        
    except Exception as e:
        print(f"Erro nas frações parciais: {e}")
        print()

if __name__ == "__main__":
    print("TESTE COMPLETO DE INTEGRAÇÃO DO MÓDULO 04-MODELAGEM-LAPLACE")
    print("="*80)
    
    test_laplace_integration()
    test_conversion_integration() 
    test_physical_systems()
    test_partial_fractions()
    
    print("="*80)
    print("TESTE CONCLUÍDO")
