#!/usr/bin/env python3
"""
Testes da Interface Numérica - ControlLab
Validação da integração simbólico ↔ numérico
"""

import unittest
import sympy as sp
import warnings
from typing import Dict, Any

# Importar módulos do projeto
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from controllab.core.symbolic_tf import SymbolicTransferFunction
from controllab.core.symbolic_ss import SymbolicStateSpace
from controllab.numerical.interface import NumericalInterface
from controllab.numerical.compiler import ExpressionCompiler
from controllab.numerical.validation import NumericalValidator
from controllab.numerical.factory import NumericalSystemFactory

# Verificar dependências
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import control
    CONTROL_AVAILABLE = True
except ImportError:
    CONTROL_AVAILABLE = False
    control = None


class TestNumericalInterface(unittest.TestCase):
    """Testes da interface numérica principal"""
    
    def setUp(self):
        """Configuração inicial dos testes"""
        self.interface = NumericalInterface()
        self.s = sp.Symbol('s')
        
        # Sistema teste simples: G(s) = 1/(s+1)
        self.simple_tf = SymbolicTransferFunction(1, self.s + 1, self.s)
        
        # Sistema teste com parâmetro: G(s) = K/(s^2 + 2s + 1)  
        self.K = sp.Symbol('K')
        self.param_tf = SymbolicTransferFunction(self.K, self.s**2 + 2*self.s + 1, self.s)
    
    @unittest.skipUnless(CONTROL_AVAILABLE, "python-control não disponível")
    def test_symbolic_to_control_tf_simple(self):
        """Testa conversão de TF simples para python-control"""
        try:
            ctrl_tf = self.interface.symbolic_to_control_tf(self.simple_tf)
            
            # Verificar que é um objeto TransferFunction
            self.assertIsNotNone(ctrl_tf)
            
            # Verificar order (ao invés de nstates que pode não existir)
            if hasattr(ctrl_tf, 'nstates') and ctrl_tf.nstates is not None:
                self.assertEqual(ctrl_tf.nstates, 1)
            elif hasattr(ctrl_tf, 'poles') and hasattr(ctrl_tf, 'zeros'):
                # Verificar que tem pelo menos um polo
                poles = ctrl_tf.poles()
                self.assertGreater(len(poles), 0)
            
            print(f"✓ TF simples convertida: {ctrl_tf}")
            
        except Exception as e:
            self.fail(f"Conversão de TF simples falhou: {e}")
    
    @unittest.skipUnless(CONTROL_AVAILABLE, "python-control não disponível")
    def test_symbolic_to_control_tf_with_substitutions(self):
        """Testa conversão de TF com substituições"""
        try:
            substitutions = {self.K: 5.0}
            ctrl_tf = self.interface.symbolic_to_control_tf(self.param_tf, substitutions)
            
            self.assertIsNotNone(ctrl_tf)
            print(f"✓ TF com substituição convertida: {ctrl_tf}")
            
        except Exception as e:
            self.fail(f"Conversão de TF com substituições falhou: {e}")
    
    @unittest.skipUnless(CONTROL_AVAILABLE, "python-control não disponível")
    def test_step_response_computation(self):
        """Testa cálculo de resposta ao degrau"""
        try:
            time, response = self.interface.compute_step_response(self.simple_tf)
            
            self.assertIsNotNone(time)
            self.assertIsNotNone(response)
            self.assertEqual(len(time), len(response))
            
            print(f"✓ Resposta ao degrau calculada: {len(time)} pontos")
            
        except Exception as e:
            self.fail(f"Cálculo de resposta ao degrau falhou: {e}")
    
    @unittest.skipUnless(CONTROL_AVAILABLE and NUMPY_AVAILABLE, "Dependências não disponíveis")
    def test_frequency_response_computation(self):
        """Testa cálculo de resposta em frequência"""
        try:
            omega = np.logspace(-2, 2, 100)
            freq, mag, phase = self.interface.compute_frequency_response(self.simple_tf, omega)
            
            self.assertIsNotNone(freq)
            self.assertIsNotNone(mag)
            self.assertIsNotNone(phase)
            self.assertEqual(len(freq), len(mag))
            self.assertEqual(len(freq), len(phase))
            
            print(f"✓ Resposta em frequência calculada: {len(freq)} pontos")
            
        except Exception as e:
            self.fail(f"Cálculo de resposta em frequência falhou: {e}")


class TestExpressionCompiler(unittest.TestCase):
    """Testes do compilador de expressões"""
    
    def setUp(self):
        """Configuração inicial"""
        self.compiler = ExpressionCompiler()
        self.x, self.y = sp.symbols('x y')
    
    def test_compile_simple_expression(self):
        """Testa compilação de expressão simples"""
        try:
            expr = self.x**2 + 2*self.x + 1
            compiled_func = self.compiler.compile_expression(expr, [self.x])
            
            self.assertIsNotNone(compiled_func)
            
            # Testar avaliação
            result = compiled_func(3)
            expected = 3**2 + 2*3 + 1  # = 16
            
            self.assertAlmostEqual(complex(result).real, expected, places=10)
            print(f"✓ Expressão compilada e avaliada: f(3) = {result}")
            
        except Exception as e:
            self.fail(f"Compilação de expressão simples falhou: {e}")
    
    def test_compile_multiple_variables(self):
        """Testa compilação com múltiplas variáveis"""
        try:
            expr = self.x**2 + self.y**2
            compiled_func = self.compiler.compile_expression(expr, [self.x, self.y])
            
            result = compiled_func(3, 4)
            expected = 3**2 + 4**2  # = 25
            
            self.assertAlmostEqual(complex(result).real, expected, places=10)
            print(f"✓ Expressão multivariável compilada: f(3,4) = {result}")
            
        except Exception as e:
            self.fail(f"Compilação multivariável falhou: {e}")
    
    @unittest.skipUnless(NUMPY_AVAILABLE, "NumPy não disponível")
    def test_compile_matrix(self):
        """Testa compilação de matriz"""
        try:
            matrix = sp.Matrix([
                [self.x, self.y],
                [self.y, self.x]
            ])
            
            compiled_func = self.compiler.compile_matrix(matrix, [self.x, self.y])
            
            result = compiled_func(1, 2)
            
            # Verificar resultado
            if isinstance(result, np.ndarray):
                self.assertEqual(result.shape, (2, 2))
                self.assertAlmostEqual(result[0, 0].real, 1, places=10)
                self.assertAlmostEqual(result[0, 1].real, 2, places=10)
            
            print(f"✓ Matriz compilada: {result}")
            
        except Exception as e:
            self.fail(f"Compilação de matriz falhou: {e}")


class TestNumericalValidator(unittest.TestCase):
    """Testes do validador numérico"""
    
    def setUp(self):
        """Configuração inicial"""
        self.validator = NumericalValidator()
        self.x, self.y = sp.symbols('x y')
    
    def test_validate_substitutions_complete(self):
        """Testa validação de substituições completas"""
        try:
            expr = self.x**2 + self.y
            substitutions = {self.x: 2, self.y: 3}
            
            result = self.validator.validate_numerical_substitutions(expr, substitutions)
            
            self.assertTrue(result['valid'])
            self.assertEqual(len(result['missing_symbols']), 0)
            self.assertEqual(len(result['invalid_values']), 0)
            
            print(f"✓ Substituições válidas: {result}")
            
        except Exception as e:
            self.fail(f"Validação de substituições falhou: {e}")
    
    def test_validate_substitutions_incomplete(self):
        """Testa validação de substituições incompletas"""
        try:
            expr = self.x**2 + self.y
            substitutions = {self.x: 2}  # Falta y
            
            result = self.validator.validate_numerical_substitutions(expr, substitutions)
            
            self.assertFalse(result['valid'])
            self.assertIn(self.y, result['missing_symbols'])
            
            print(f"✓ Substituições incompletas detectadas: {result['missing_symbols']}")
            
        except Exception as e:
            self.fail(f"Validação de substituições incompletas falhou: {e}")
    
    def test_stability_check_stable(self):
        """Testa verificação de estabilidade - sistema estável"""
        try:
            poles = [-1, -2, -3+2j, -3-2j]  # Todos no semiplano esquerdo
            
            result = self.validator.check_stability_numerical(poles)
            
            self.assertTrue(result['is_stable'])
            self.assertEqual(len(result['unstable_poles']), 0)
            
            print(f"✓ Sistema estável detectado: {len(result['stable_poles'])} polos estáveis")
            
        except Exception as e:
            self.fail(f"Verificação de estabilidade falhou: {e}")
    
    def test_stability_check_unstable(self):
        """Testa verificação de estabilidade - sistema instável"""
        try:
            poles = [-1, 1, -2]  # Um polo no semiplano direito
            
            result = self.validator.check_stability_numerical(poles)
            
            self.assertFalse(result['is_stable'])
            self.assertEqual(len(result['unstable_poles']), 1)
            
            print(f"✓ Sistema instável detectado: {len(result['unstable_poles'])} polos instáveis")
            
        except Exception as e:
            self.fail(f"Verificação de instabilidade falhou: {e}")


class TestNumericalSystemFactory(unittest.TestCase):
    """Testes da factory de sistemas numéricos"""
    
    def setUp(self):
        """Configuração inicial"""
        self.factory = NumericalSystemFactory()
        self.s = sp.Symbol('s')
        self.tf = SymbolicTransferFunction(1, self.s + 1, self.s)
    
    @unittest.skipUnless(CONTROL_AVAILABLE, "python-control não disponível")
    def test_create_tf_system(self):
        """Testa criação de sistema TF"""
        try:
            ctrl_tf = self.factory.create_tf_system(self.tf)
            
            self.assertIsNotNone(ctrl_tf)
            print(f"✓ Sistema TF criado via factory: {ctrl_tf}")
            
        except Exception as e:
            self.fail(f"Criação de sistema TF via factory falhou: {e}")
    
    @unittest.skipUnless(CONTROL_AVAILABLE, "python-control não disponível")
    def test_cache_functionality(self):
        """Testa funcionalidade de cache"""
        try:
            # Primeira criação
            ctrl_tf1 = self.factory.create_tf_system(self.tf)
            
            # Segunda criação (deve vir do cache)
            ctrl_tf2 = self.factory.create_tf_system(self.tf)
            
            # Verificar que são o mesmo objeto (cache hit)
            self.assertIs(ctrl_tf1, ctrl_tf2)
            
            # Verificar estatísticas do cache
            stats = self.factory.get_cache_stats()
            self.assertGreater(stats['cached_systems'], 0)
            
            print(f"✓ Cache funcionando: {stats}")
            
        except Exception as e:
            self.fail(f"Teste de cache falhou: {e}")


class TestIntegrationSymbolicNumerical(unittest.TestCase):
    """Testes de integração simbólico-numérico"""
    
    def setUp(self):
        """Configuração inicial"""
        self.interface = NumericalInterface()
        self.s = sp.Symbol('s')
        
    @unittest.skipUnless(CONTROL_AVAILABLE, "python-control não disponível")
    def test_complete_workflow(self):
        """Testa fluxo completo: simbólico → numérico → análise"""
        try:
            # 1. Criar sistema simbólico
            K = sp.Symbol('K')
            tf_symbolic = SymbolicTransferFunction(K, self.s**2 + 2*self.s + K, self.s)
            
            # 2. Converter para numérico
            substitutions = {K: 10}
            tf_numeric = self.interface.symbolic_to_control_tf(tf_symbolic, substitutions)
            
            # 3. Analisar sistema numérico
            time, response = self.interface.compute_step_response(tf_numeric)
            
            # 4. Verificações
            self.assertIsNotNone(tf_numeric)
            self.assertIsNotNone(time)
            self.assertIsNotNone(response)
            self.assertEqual(len(time), len(response))
            
            print(f"✓ Fluxo completo executado:")
            print(f"  - Sistema simbólico: {tf_symbolic}")
            print(f"  - Sistema numérico: {tf_numeric}")
            print(f"  - Resposta calculada: {len(time)} pontos")
            
        except Exception as e:
            self.fail(f"Fluxo completo falhou: {e}")


def run_numerical_tests():
    """Executa todos os testes da interface numérica"""
    print("=" * 80)
    print("TESTES DA INTERFACE NUMÉRICA - ControlLab")
    print("=" * 80)
    
    # Verificar dependências
    print("\n📋 Verificação de Dependências:")
    print(f"  SymPy: ✓ Disponível")
    print(f"  NumPy: {'✓ Disponível' if NUMPY_AVAILABLE else '✗ Não disponível'}")
    print(f"  python-control: {'✓ Disponível' if CONTROL_AVAILABLE else '✗ Não disponível'}")
    
    if not CONTROL_AVAILABLE:
        print("\n⚠️  AVISO: python-control não disponível - alguns testes serão ignorados")
        print("   Para instalar: pip install control")
    
    # Configurar suite de testes
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Adicionar classes de teste
    test_classes = [
        TestNumericalInterface,
        TestExpressionCompiler,
        TestNumericalValidator,
        TestNumericalSystemFactory,
        TestIntegrationSymbolicNumerical
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Executar testes
    print(f"\n🧪 Executando testes da interface numérica...")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Resumo final
    print(f"\n" + "=" * 50)
    print("RESUMO DOS TESTES:")
    print(f"Executados: {result.testsRun}")
    print(f"✓ Sucessos: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"✗ Falhas: {len(result.failures)}")
    print(f"❌ Erros: {len(result.errors)}")
    print(f"⏭️  Ignorados: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    if result.failures:
        print("\n❌ FALHAS:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError: ')[-1].split()[0] if 'AssertionError:' in traceback else 'Erro'}")
    
    if result.errors:
        print("\n❌ ERROS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split()[-1] if traceback else 'Erro desconhecido'}")
    
    print("\n🎯 Interface numérica testada!")
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_numerical_tests()
    exit(0 if success else 1)
