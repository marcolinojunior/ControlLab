#!/usr/bin/env python3
"""
Testes para Fase 3: Interface Numérica
Validação da ponte simbólico ↔ numérico
"""

import pytest
import numpy as np
import sympy as sp
import sys
import os

# Adicionar src ao path para importar módulos
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from controllab.core.symbolic_tf import SymbolicTransferFunction
from controllab.core.symbolic_ss import SymbolicStateSpace
from controllab.numerical.interface import NumericalInterface
from controllab.numerical.compiler import ExpressionCompiler, compile_expression
from controllab.numerical.validation import NumericalValidator

# Verificar dependências numéricas
try:
    import control
    CONTROL_AVAILABLE = True
except ImportError:
    CONTROL_AVAILABLE = False

@pytest.fixture
def s():
    """Símbolo de Laplace"""
    return sp.Symbol('s')

@pytest.fixture
def numerical_interface():
    """Interface numérica para testes"""
    return NumericalInterface()

@pytest.fixture
def sample_tf(s):
    """Função de transferência simbólica de exemplo"""
    return SymbolicTransferFunction(1, s**2 + 2*s + 1, s)

@pytest.fixture
def sample_ss(s):
    """Sistema em espaço de estados de exemplo"""
    A = sp.Matrix([[-1, 1], [0, -2]])
    B = sp.Matrix([[0], [1]])
    C = sp.Matrix([[1, 0]])
    D = sp.Matrix([[0]])
    return SymbolicStateSpace(A, B, C, D)

class TestNumericalInterface:
    """Testes da interface numérica principal"""
    
    def test_import_numerical_interface(self):
        """Teste de importação do módulo numerical"""
        from controllab.numerical.interface import NumericalInterface
        assert NumericalInterface is not None
        
    def test_numerical_interface_instantiation(self, numerical_interface):
        """Teste de instanciação da interface numérica"""
        assert numerical_interface is not None
        assert hasattr(numerical_interface, 'symbolic_to_control_tf')
        assert hasattr(numerical_interface, 'symbolic_to_control_ss')
    
    @pytest.mark.skipif(not CONTROL_AVAILABLE, reason="python-control não disponível")
    def test_symbolic_to_control_tf_conversion(self, numerical_interface, sample_tf):
        """Teste de conversão simbólico → control.TransferFunction"""
        # Conversão básica sem substituições
        result = numerical_interface.symbolic_to_control_tf(sample_tf)
        
        assert result is not None
        # Verificar se é um objeto control.TransferFunction
        assert hasattr(result, 'num')
        assert hasattr(result, 'den')
        
        # Verificar coeficientes (sistema 1/(s^2 + 2s + 1))
        expected_num = [1]
        expected_den = [1, 2, 1]
        
        np.testing.assert_array_almost_equal(result.num[0][0], expected_num)
        np.testing.assert_array_almost_equal(result.den[0][0], expected_den)
    
    @pytest.mark.skipif(not CONTROL_AVAILABLE, reason="python-control não disponível")
    def test_symbolic_to_control_tf_with_substitutions(self, numerical_interface, s):
        """Teste de conversão com substituições de parâmetros"""
        K = sp.Symbol('K')
        wn = sp.Symbol('wn')
        
        # Sistema parametrizado: K*wn^2 / (s^2 + 2*wn*s + wn^2)
        tf_parametric = SymbolicTransferFunction(K * wn**2, s**2 + 2*wn*s + wn**2, s)
        
        substitutions = {K: 5.0, wn: 2.0}
        result = numerical_interface.symbolic_to_control_tf(tf_parametric, substitutions)
        
        # Verificar coeficientes com substituições aplicadas
        expected_num = [20.0]  # K * wn^2 = 5 * 4 = 20
        expected_den = [1.0, 4.0, 4.0]  # s^2 + 2*2*s + 2^2
        
        np.testing.assert_array_almost_equal(result.num[0][0], expected_num)
        np.testing.assert_array_almost_equal(result.den[0][0], expected_den)
    
    @pytest.mark.skipif(not CONTROL_AVAILABLE, reason="python-control não disponível")
    def test_symbolic_to_control_ss_conversion(self, numerical_interface, sample_ss):
        """Teste de conversão simbólico → control.StateSpace"""
        result = numerical_interface.symbolic_to_control_ss(sample_ss)
        
        assert result is not None
        assert hasattr(result, 'A')
        assert hasattr(result, 'B') 
        assert hasattr(result, 'C')
        assert hasattr(result, 'D')
        
        # Verificar matrizes convertidas
        expected_A = np.array([[-1, 1], [0, -2]], dtype=float)
        expected_B = np.array([[0], [1]], dtype=float)
        expected_C = np.array([[1, 0]], dtype=float)
        expected_D = np.array([[0]], dtype=float)
        
        np.testing.assert_array_almost_equal(result.A, expected_A)
        np.testing.assert_array_almost_equal(result.B, expected_B)
        np.testing.assert_array_almost_equal(result.C, expected_C)
        np.testing.assert_array_almost_equal(result.D, expected_D)

class TestExpressionCompiler:
    """Testes do compilador de expressões"""
    
    def test_expression_compiler_instantiation(self):
        """Teste de instanciação do compilador"""
        compiler = ExpressionCompiler()
        assert compiler is not None
        
    def test_compile_simple_expression(self, s):
        """Teste de compilação de expressão simples"""
        expr = s**2 + 2*s + 1
        variables = [s]
        
        compiled_func = compile_expression(expr, variables, backend='numpy')
        
        assert compiled_func is not None
        assert callable(compiled_func)
        
        # Testar avaliação da função compilada
        result = compiled_func(2.0)
        expected = 2**2 + 2*2 + 1  # = 9
        assert abs(result - expected) < 1e-12
        
    def test_compile_multivariate_expression(self):
        """Teste de compilação com múltiplas variáveis"""
        x, y = sp.symbols('x y')
        expr = x**2 + y**2 + 2*x*y
        variables = [x, y]
        
        compiled_func = compile_expression(expr, variables, backend='numpy')
        
        # Testar com valores específicos
        result = compiled_func(3.0, 4.0)
        expected = 3**2 + 4**2 + 2*3*4  # = 9 + 16 + 24 = 49
        assert abs(result - expected) < 1e-12
        
    def test_compiled_function_properties(self, s):
        """Teste das propriedades da função compilada"""
        expr = s**3 + s**2 + s + 1
        variables = [s]
        
        compiled_func = compile_expression(expr, variables, backend='numpy')
        
        # Verificar propriedades da CompiledFunction
        assert hasattr(compiled_func, 'expression')
        assert hasattr(compiled_func, 'variables')
        assert hasattr(compiled_func, 'backend')
        assert hasattr(compiled_func, 'call_count')
        
        # Testar contador de chamadas
        initial_count = compiled_func.call_count
        compiled_func(1.0)
        assert compiled_func.call_count == initial_count + 1

class TestNumericalValidator:
    """Testes do validador numérico"""
    
    def test_validator_instantiation(self):
        """Teste de instanciação do validador"""
        validator = NumericalValidator()
        assert validator is not None
        
    def test_validate_complete_substitutions(self, s):
        """Teste de validação com substituições completas"""
        expr = s**2 + 2*s + 1
        substitutions = {s: 3.0}
        
        validator = NumericalValidator()
        result = validator.validate_numerical_substitutions(expr, substitutions)
        
        assert result['valid'] == True
        assert len(result['missing_symbols']) == 0
        
    def test_validate_incomplete_substitutions(self):
        """Teste de validação com substituições incompletas"""
        s, K = sp.symbols('s K')
        expr = K * s**2 + 2*s + 1
        substitutions = {s: 3.0}  # K não substituído
        
        validator = NumericalValidator()
        result = validator.validate_numerical_substitutions(expr, substitutions)
        
        assert result['valid'] == False
        assert K in result['missing_symbols']

class TestIntegrationSymbolicNumerical:
    """Testes de integração simbólico ↔ numérico"""
    
    @pytest.mark.skipif(not CONTROL_AVAILABLE, reason="python-control não disponível")
    def test_step_response_equivalence(self, numerical_interface, sample_tf):
        """Teste de equivalência na resposta ao degrau"""
        # Converter para control
        control_tf = numerical_interface.symbolic_to_control_tf(sample_tf)
        
        # Calcular resposta ao degrau
        time_vector = np.linspace(0, 5, 100)
        t_out, y_out = control.step_response(control_tf, time_vector)
        
        assert len(t_out) > 0
        assert len(y_out) > 0
        assert len(t_out) == len(y_out)
        
        # Verificar propriedades básicas da resposta
        assert y_out[0] == 0.0  # Condição inicial
        assert y_out[-1] > 0.9  # Valor final próximo de 1 (estável)
        
    @pytest.mark.skipif(not CONTROL_AVAILABLE, reason="python-control não disponível")
    def test_poles_equivalence(self, numerical_interface, sample_tf):
        """Teste de equivalência dos polos"""
        # Polos simbólicos
        symbolic_poles = sample_tf.poles()
        
        # Converter para control e obter polos numéricos
        control_tf = numerical_interface.symbolic_to_control_tf(sample_tf)
        numerical_poles = control.poles(control_tf)
        
        # Converter polos simbólicos para numérico
        symbolic_poles_float = [complex(p) for p in symbolic_poles]
        
        # Comparar (ordenar pois ordem pode diferir)
        symbolic_poles_sorted = sorted(symbolic_poles_float, key=lambda x: (x.real, x.imag))
        numerical_poles_sorted = sorted(numerical_poles, key=lambda x: (x.real, x.imag))
        
        for sp_pole, num_pole in zip(symbolic_poles_sorted, numerical_poles_sorted):
            assert abs(sp_pole - num_pole) < 1e-12

class TestPerformanceBenchmarks:
    """Testes de performance e benchmarks"""
    
    def test_compilation_performance(self, s):
        """Teste de performance da compilação"""
        import time
        
        # Expressão complexa para testar performance
        expr = sum(s**i for i in range(10))
        variables = [s]
        
        start_time = time.time()
        compiled_func = compile_expression(expr, variables, backend='numpy')
        compilation_time = time.time() - start_time
        
        # Compilação deve ser razoavelmente rápida (< 1 segundo)
        assert compilation_time < 1.0
        
        # Testar performance de execução
        start_time = time.time()
        for _ in range(1000):
            compiled_func(2.0)
        execution_time = time.time() - start_time
        
        # Execução deve ser muito rápida
        assert execution_time < 0.1

def test_module_integration():
    """Teste de integração completa dos módulos"""
    # Verificar se todos os módulos podem ser importados juntos
    from controllab.numerical import (
        NumericalInterface, 
        ExpressionCompiler, 
        NumericalValidator,
        check_numerical_dependencies,
        get_available_backends
    )
    
    # Verificar dependências
    dependencies = check_numerical_dependencies()
    assert 'numpy' in dependencies
    
    backends = get_available_backends()
    assert 'numpy' in backends

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
