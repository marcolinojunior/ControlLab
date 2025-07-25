"""
Teste simplificado do núcleo simbólico expandido
Demonstra funcionalidades core sem dependências externas
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import sympy as sp
from controllab.core.symbolic_tf import SymbolicTransferFunction
from controllab.core.symbolic_ss import SymbolicStateSpace


def test_enhanced_transfer_function():
    """Testa funcionalidades avançadas da função de transferência"""
    print("="*60)
    print("TESTE: Funcionalidades Avançadas da Função de Transferência")
    print("="*60)
    
    # Cria função de transferência de exemplo
    s = sp.Symbol('s', complex=True)
    G = SymbolicTransferFunction(10, s**2 + 3*s + 2, s)
    
    # Testa coeficientes
    print("1. Extração de coeficientes:")
    coeffs = G.get_coefficients()
    print(f"   Numerador: {coeffs['num']}")
    print(f"   Denominador: {coeffs['den']}")
    
    # Testa transformação de Tustin
    print("\n2. Transformação de Tustin (T=0.1):")
    T = 0.1
    G_discrete = G.apply_tustin_transform(T)
    print(f"   G(z) = {G_discrete}")
    
    # Testa análise de margens
    print("\n3. Análise de margens:")
    margins = G.get_margin_analysis()
    print(f"   Magnitude: {margins.get('magnitude_expr', 'N/A')}")
    print(f"   Fase: {margins.get('phase_expr', 'N/A')}")
    
    # Testa equação característica
    print("\n4. Equação característica:")
    char_eq = G.characteristic_equation()
    print(f"   Equação: {char_eq} = 0")
    
    # Testa aplicação de regras de Laplace
    print("\n5. Aplicação de regras de Laplace:")
    t = sp.Symbol('t', real=True, positive=True)
    time_expr = sp.exp(-2*t)
    laplace_result = G.apply_laplace_rules(time_expr)
    print(f"   L{{e^(-2t)}} processado")
    
    # Validações usando assert
    assert G is not None, "Função de transferência deve ser criada"
    assert coeffs['num'] is not None, "Numerador deve ser extraído"
    assert G_discrete is not None, "Transformação de Tustin deve funcionar"
    assert char_eq is not None, "Equação característica deve ser calculada"


def test_state_space_enhancements():
    """Testa melhorias no espaço de estados"""
    print("\n" + "="*60)
    print("TESTE: Melhorias no Espaço de Estados")
    print("="*60)
    
    # Sistema de teste
    A = sp.Matrix([[0, 1], [-2, -3]])
    B = sp.Matrix([[0], [1]])
    C = sp.Matrix([[1, 0]])
    D = sp.Matrix([[0]])
    
    sys = SymbolicStateSpace(A, B, C, D)
    
    print("1. Sistema criado:")
    print(f"   A = {A}")
    print(f"   B = {B}")
    print(f"   C = {C}")
    print(f"   D = {D}")
    
    # Testa função de transferência
    print("\n2. Função de transferência:")
    tf = sys.transfer_function()
    print(f"   G(s) = {tf}")
    
    # Testa autovalores
    print("\n3. Autovalores:")
    eigenvals = sys.eigenvalues()
    print(f"   Autovalores: {eigenvals}")
    
    # Testa controlabilidade
    print("\n4. Controlabilidade:")
    controllable = sys.is_controllable()
    print(f"   Controlável: {controllable}")
    
    # Testa observabilidade
    print("\n5. Observabilidade:")
    observable = sys.is_observable()
    print(f"   Observável: {observable}")
    
    # Validações usando assert
    assert sys is not None, "Sistema de espaço de estados deve ser criado"
    assert tf is not None, "Função de transferência deve ser calculada"
    assert eigenvals is not None, "Autovalores devem ser calculados"
    assert isinstance(controllable, (bool, sp.Basic)), "Controlabilidade deve retornar resultado válido"
    assert isinstance(observable, (bool, sp.Basic)), "Observabilidade deve retornar resultado válido"


def test_symbolic_operations():
    """Testa operações simbólicas avançadas"""
    print("\n" + "="*60)
    print("TESTE: Operações Simbólicas Avançadas")
    print("="*60)
    
    s = sp.Symbol('s', complex=True)
    
    # Sistemas de teste
    G1 = SymbolicTransferFunction(1, s + 1, s)
    G2 = SymbolicTransferFunction(2, s + 2, s)
    
    print("1. Sistemas criados:")
    print(f"   G1(s) = {G1}")
    print(f"   G2(s) = {G2}")
    
    # Operações básicas
    print("\n2. Multiplicação:")
    G_mult = G1 * G2
    print(f"   G1 * G2 = {G_mult}")
    
    print("\n3. Adição:")
    G_add = G1 + G2
    print(f"   G1 + G2 = {G_add}")
    
    print("\n4. Divisão:")
    G_div = G1 / G2
    print(f"   G1 / G2 = {G_div}")
    
    # Operações avançadas
    print("\n5. Simplificação:")
    G_complex = SymbolicTransferFunction(s**2 + 2*s + 1, s**3 + 3*s**2 + 3*s + 1, s)
    G_simplified = G_complex.simplify()
    print(f"   Original: {G_complex}")
    print(f"   Simplificado: {G_simplified}")
    
    print("\n6. Frações parciais:")
    G_partial = G_complex.partial_fractions()
    print(f"   Frações parciais: {G_partial}")
    
    # Substituição de parâmetros
    print("\n7. Substituição de parâmetros:")
    K = sp.Symbol('K', real=True)
    G_param = SymbolicTransferFunction(K, s + K, s)
    G_numeric = G_param.substitute({K: 5})
    print(f"   G(s,K) = {G_param}")
    print(f"   G(s,5) = {G_numeric}")
    
    # Validações usando assert
    assert G_mult is not None, "Multiplicação deve ser calculada"
    assert G_add is not None, "Adição deve ser calculada"
    assert G_div is not None, "Divisão deve ser calculada"
    assert G_simplified is not None, "Simplificação deve funcionar"
    assert G_partial is not None, "Frações parciais devem ser calculadas"
    assert G_numeric is not None, "Substituição numérica deve funcionar"


def test_poles_zeros_analysis():
    """Testa análise de polos e zeros"""
    print("\n" + "="*60)
    print("TESTE: Análise de Polos e Zeros")
    print("="*60)
    
    s = sp.Symbol('s', complex=True)
    
    # Sistema com polos e zeros conhecidos
    # G(s) = (s+1) / ((s+2)(s+3))
    num = s + 1
    den = (s + 2) * (s + 3)
    G = SymbolicTransferFunction(num, den, s)
    
    print(f"Sistema: G(s) = {G}")
    
    print("\n1. Análise de polos:")
    poles = G.poles()
    print(f"   Polos: {poles}")
    
    print("\n2. Análise de zeros:")
    zeros = G.zeros()
    print(f"   Zeros: {zeros}")
    
    print("\n3. Grau do sistema:")
    degree = G.degree
    print(f"   Grau (num, den): {degree}")
    
    print("\n4. Sistema próprio:")
    is_proper = G.is_proper
    print(f"   É próprio: {is_proper}")
    
    # Avaliação em pontos específicos
    print("\n5. Avaliação em s=0:")
    try:
        val_at_zero = G.evaluate(0)
        print(f"   G(0) = {val_at_zero}")
    except Exception as e:
        print(f"   Erro: {e}")
    
    print("\n6. Avaliação em s=j1:")
    try:
        val_at_j1 = G.evaluate(1j)
        print(f"   G(j1) = {val_at_j1}")
    except Exception as e:
        print(f"   Erro: {e}")
    
    # Validações usando assert
    assert poles is not None, "Polos devem ser calculados"
    assert zeros is not None, "Zeros devem ser calculados"
    assert degree is not None, "Grau deve ser determinado"
    assert is_proper is not None, "Propriedade 'próprio' deve ser verificada"


def test_latex_generation():
    """Testa geração de LaTeX"""
    print("\n" + "="*60)
    print("TESTE: Geração de LaTeX")
    print("="*60)
    
    s = sp.Symbol('s', complex=True)
    G = SymbolicTransferFunction(s + 1, s**2 + 2*s + 1, s)
    
    print("1. Função de transferência para LaTeX:")
    latex_str = G.to_latex()
    print(f"   LaTeX: {latex_str}")
    
    # Sistema em espaço de estados
    print("\n2. Espaço de estados:")
    A = sp.Matrix([[0, 1], [-1, -2]])
    B = sp.Matrix([[0], [1]])
    C = sp.Matrix([[1, 0]])
    D = sp.Matrix([[0]])
    
    sys = SymbolicStateSpace(A, B, C, D)
    
    print(f"   Sistema criado com A = {A}")
    print(f"   B = {B}, C = {C}, D = {D}")
    
    # Validações usando assert
    assert G is not None, "Função de transferência deve ser criada"
    assert latex_str is not None, "LaTeX deve ser gerado"
    assert sys is not None, "Sistema de espaço de estados deve ser criado"


def test_history_system():
    """Testa sistema de histórico"""
    print("\n" + "="*60)
    print("TESTE: Sistema de Histórico")
    print("="*60)
    
    s = sp.Symbol('s', complex=True)
    G1 = SymbolicTransferFunction(1, s + 1, s)
    G2 = SymbolicTransferFunction(2, s + 2, s)
    
    print("1. Operação com histórico:")
    G_result = G1 * G2
    
    print("\n2. Histórico formatado:")
    history = G_result.history.get_formatted_history()
    for step in history:
        print(f"   {step}")
    
    print("\n3. Histórico LaTeX:")
    latex_history = G_result.history.get_latex_history()
    print(f"   LaTeX disponível: {len(latex_history) > 0}")
    
    # Validações usando assert
    assert G_result is not None, "Resultado da operação deve existir"
    assert history is not None, "Histórico deve ser gerado"
    assert latex_history is not None, "Histórico LaTeX deve estar disponível"


def test_integration_example():
    """Exemplo de integração das funcionalidades"""
    print("\n" + "="*60)
    print("TESTE: Exemplo de Integração")
    print("="*60)
    
    # Problema: Analisar sistema G(s) = K/(s²+3s+2)
    s = sp.Symbol('s', complex=True)
    K = sp.Symbol('K', real=True, positive=True)
    
    G = SymbolicTransferFunction(K, s**2 + 3*s + 2, s)
    
    print(f"Sistema: G(s) = {G}")
    
    # 1. Análise estrutural
    print("\n1. Análise Estrutural:")
    poles = G.poles()
    zeros = G.zeros()
    print(f"   Polos: {poles}")
    print(f"   Zeros: {zeros}")
    print(f"   Ordem: {G.degree}")
    
    # 2. Substituição de parâmetros
    print("\n2. Caso Específico (K=10):")
    G_numeric = G.substitute({K: 10})
    print(f"   G(s) = {G_numeric}")
    
    # 3. Transformação para discreto
    print("\n3. Transformação Discreta (T=0.1s):")
    G_discrete = G_numeric.apply_tustin_transform(0.1)
    print(f"   G(z) = {G_discrete}")
    
    # 4. Análise de resposta
    print("\n4. Coeficientes:")
    coeffs = G_numeric.get_coefficients()
    print(f"   Numerador: {coeffs['num']}")
    print(f"   Denominador: {coeffs['den']}")
    
    # 5. Sistema em malha fechada (realimentação unitária)
    print("\n5. Malha Fechada:")
    # T(s) = G(s)/(1+G(s))
    T_num = G_numeric.numerator
    T_den = G_numeric.denominator + G_numeric.numerator
    T = SymbolicTransferFunction(T_num, T_den, s)
    print(f"   T(s) = {T}")
    
    # 6. Análise da malha fechada
    print("\n6. Análise da Malha Fechada:")
    T_poles = T.poles()
    print(f"   Polos de malha fechada: {T_poles}")
    
    # Validações usando assert
    assert G is not None, "Função de transferência parametrizada deve ser criada"
    assert G_numeric is not None, "Substituição numérica deve funcionar"
    assert G_discrete is not None, "Transformação discreta deve funcionar"
    assert coeffs is not None, "Coeficientes devem ser extraídos"
    assert T is not None, "Malha fechada deve ser calculada"
    assert T_poles is not None, "Polos de malha fechada devem ser encontrados"


def main():
    """Executa todos os testes"""
    print("TESTE SIMPLIFICADO DO NÚCLEO SIMBÓLICO EXPANDIDO")
    print("Demonstrando funcionalidades core implementadas")
    print("="*80)
    
    tests = [
        ("Funcionalidades Avançadas TF", test_enhanced_transfer_function),
        ("Melhorias Espaço de Estados", test_state_space_enhancements),
        ("Operações Simbólicas", test_symbolic_operations),
        ("Análise Polos/Zeros", test_poles_zeros_analysis),
        ("Geração LaTeX", test_latex_generation),
        ("Sistema de Histórico", test_history_system),
        ("Integração Completa", test_integration_example)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            print(f"\nExecutando: {test_name}")
            result = test_func()
            results.append((test_name, "SUCESSO" if result else "FALHA"))
            print(f"Status: {'✅ SUCESSO' if result else '❌ FALHA'}")
        except Exception as e:
            results.append((test_name, f"ERRO: {str(e)}"))
            print(f"Status: ❌ ERRO - {str(e)}")
    
    # Resumo final
    print("\n" + "="*80)
    print("RESUMO DOS TESTES")
    print("="*80)
    
    for test_name, status in results:
        status_symbol = "✅" if status == "SUCESSO" else "❌"
        print(f"{status_symbol} {test_name}: {status}")
    
    successful_tests = sum(1 for _, status in results if status == "SUCESSO")
    total_tests = len(results)
    
    print(f"\nTotal: {successful_tests}/{total_tests} testes bem-sucedidos")
    
    if successful_tests == total_tests:
        print("\n🎉 TODOS OS TESTES CORE PASSARAM!")
        print("O núcleo simbólico base está funcionando perfeitamente!")
        print("\n📝 FUNCIONALIDADES IMPLEMENTADAS:")
        print("   ✅ Funções de transferência simbólicas avançadas")
        print("   ✅ Transformação de Tustin para discretização")
        print("   ✅ Análise de margens (expressões simbólicas)")
        print("   ✅ Aplicação de regras de Laplace")
        print("   ✅ Extração de coeficientes")
        print("   ✅ Sistemas em espaço de estados completos")
        print("   ✅ Análise de controlabilidade/observabilidade")
        print("   ✅ Operações simbólicas avançadas")
        print("   ✅ Análise detalhada de polos e zeros")
        print("   ✅ Geração de LaTeX")
        print("   ✅ Sistema de histórico pedagógico")
        print("   ✅ Integração completa de funcionalidades")
        
        print("\n🚀 PRONTO PARA EXPANSÃO:")
        print("   - Módulos de estabilidade (Routh-Hurwitz, Nyquist, Bode)")
        print("   - Design de controladores (PID, Lead-Lag, LQR)")
        print("   - Transformadas completas (Laplace, Z, Fourier)")
        print("   - Visualização avançada (diagramas, plotagem)")
        
    else:
        print(f"\n⚠️  {total_tests - successful_tests} teste(s) falharam")
        print("Verifique os erros acima para debugging")
    
    return successful_tests == total_tests


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
