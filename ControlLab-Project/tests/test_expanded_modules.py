#!/usr/bin/env python3
"""
Testes para Módulos Expandidos - ControlLab Numerical
Testes para performance, conversions e bode_asymptotic
"""

import unittest
import sys
import os

# Adicionar o caminho do projeto ao sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
src_path = os.path.join(project_root, 'src')
sys.path.append(src_path)

try:
    import sympy as sp
    import numpy as np
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False

from controllab.core.symbolic_tf import SymbolicTransferFunction
from controllab.core.symbolic_ss import SymbolicStateSpace

if SYMPY_AVAILABLE:
    from controllab.numerical.performance import PerformanceAnalyzer
    from controllab.numerical.conversions import StateSpaceConverter
    from controllab.numerical.bode_asymptotic import BodeAsymptoticAnalyzer
    from controllab.numerical.interface import NumericalInterface


@unittest.skipIf(not SYMPY_AVAILABLE, "SymPy não disponível")
class TestPerformanceAnalyzer(unittest.TestCase):
    """Testes para PerformanceAnalyzer"""
    
    def setUp(self):
        self.analyzer = PerformanceAnalyzer()
        self.s = sp.Symbol('s')
    
    def test_steady_state_error_type0(self):
        """Teste de erro steady-state para sistema tipo 0"""
        # Sistema: G(s) = 10/(s+1)
        tf_system = SymbolicTransferFunction(10, self.s + 1, self.s)
        
        result = self.analyzer.analyze_steady_state_error(tf_system, 'step')
        
        self.assertEqual(result['system_type'], 0)
        self.assertAlmostEqual(result['position_constant_Kp'], 10.0, places=5)
        self.assertAlmostEqual(result['steady_state_error'], 1.0/11.0, places=5)
    
    def test_steady_state_error_type1(self):
        """Teste de erro steady-state para sistema tipo 1"""
        # Sistema: G(s) = 10/(s*(s+1))
        tf_system = SymbolicTransferFunction(10, self.s * (self.s + 1), self.s)
        
        result = self.analyzer.analyze_steady_state_error(tf_system, 'step')
        
        self.assertEqual(result['system_type'], 1)
        self.assertEqual(result['steady_state_error'], 0.0)
        
        # Teste para rampa
        result_ramp = self.analyzer.analyze_steady_state_error(tf_system, 'ramp')
        self.assertAlmostEqual(result_ramp['velocity_constant_Kv'], 10.0, places=5)
        self.assertAlmostEqual(result_ramp['steady_state_error'], 0.1, places=5)
    
    def test_second_order_analysis(self):
        """Teste de análise de sistema de segunda ordem"""
        # Sistema: G(s) = 100/(s² + 2*s + 100)
        # ωn = 10, ζ = 0.1 (subamortecido)
        tf_system = SymbolicTransferFunction(100, self.s**2 + 2*self.s + 100, self.s)
        
        result = self.analyzer.analyze_second_order_parameters(tf_system)
        
        self.assertEqual(result['order'], 2)
        self.assertAlmostEqual(result['natural_frequency_wn'], 10.0, places=5)
        self.assertAlmostEqual(result['damping_ratio_zeta'], 0.1, places=5)
        self.assertEqual(result['response_type'], "Subamortecido")
        self.assertIsNotNone(result['theoretical_overshoot'])
    
    def test_time_response_specs(self):
        """Teste de especificações de resposta temporal"""
        # Simular resposta ao degrau com overshoot
        time = np.linspace(0, 10, 1000)
        # Resposta subamortecida com overshoot
        response = 1 - np.exp(-0.5*time) * np.cos(2*time) - 0.25*np.exp(-0.5*time) * np.sin(2*time)
        
        specs = self.analyzer.analyze_time_response_specs(response, time, 'step')
        
        self.assertIn('overshoot_percent', specs)
        self.assertIn('rise_time', specs)
        self.assertIn('settling_time_2_percent', specs)
        self.assertIn('peak_time', specs)
        # Verificar se tem algum overshoot (pode ser pequeno)
        self.assertGreaterEqual(specs['overshoot_percent'], 0)


@unittest.skipIf(not SYMPY_AVAILABLE, "SymPy não disponível")
class TestStateSpaceConverter(unittest.TestCase):
    """Testes para StateSpaceConverter"""
    
    def setUp(self):
        self.converter = StateSpaceConverter()
        self.s = sp.Symbol('s')
    
    def test_tf_to_ss_controllable(self):
        """Teste de conversão TF → SS controlável"""
        # Sistema: G(s) = 2/(s² + 3s + 2)
        tf_system = SymbolicTransferFunction(2, self.s**2 + 3*self.s + 2, self.s)
        
        ss_system = self.converter.tf_to_ss_canonical_controllable(tf_system)
        
        self.assertIsInstance(ss_system, SymbolicStateSpace)
        self.assertEqual(ss_system.A.shape, (2, 2))
        self.assertEqual(ss_system.B.shape, (2, 1))
        self.assertEqual(ss_system.C.shape, (1, 2))
        self.assertEqual(ss_system.D.shape, (1, 1))
        
        # Verificar matriz A (forma controlável)
        self.assertEqual(ss_system.A[0, 0], -2)  # -a0
        self.assertEqual(ss_system.A[0, 1], -3)  # -a1
        self.assertEqual(ss_system.A[1, 0], 1)   # identidade
        self.assertEqual(ss_system.A[1, 1], 0)
    
    def test_tf_to_ss_observable(self):
        """Teste de conversão TF → SS observável"""
        # Sistema: G(s) = 2/(s² + 3s + 2)
        tf_system = SymbolicTransferFunction(2, self.s**2 + 3*self.s + 2, self.s)
        
        ss_system = self.converter.tf_to_ss_canonical_observable(tf_system)
        
        self.assertIsInstance(ss_system, SymbolicStateSpace)
        self.assertEqual(ss_system.A.shape, (2, 2))
        
        # Verificar matriz A (forma observável - última coluna)
        self.assertEqual(ss_system.A[0, 1], -2)  # -a0
        self.assertEqual(ss_system.A[1, 1], -3)  # -a1
    
    def test_ss_to_tf_conversion(self):
        """Teste de conversão SS → TF"""
        # Criar sistema SS simples
        A = sp.Matrix([[-2, -3], [1, 0]])
        B = sp.Matrix([[1], [0]])
        C = sp.Matrix([[0, 2]])
        D = sp.Matrix([[0]])
        
        ss_system = SymbolicStateSpace(A, B, C, D)
        ss_system.variable = self.s
        
        tf_system = self.converter.ss_to_tf_via_characteristic(ss_system)
        
        self.assertIsInstance(tf_system, SymbolicTransferFunction)
        # Verificar se a conversão está correta simbolicamente
        self.assertEqual(tf_system.variable, self.s)
    
    def test_controllability_check(self):
        """Teste de verificação de controlabilidade"""
        # Sistema controlável
        A = sp.Matrix([[-1, -2], [1, 0]])
        B = sp.Matrix([[1], [0]])
        C = sp.Matrix([[0, 1]])
        D = sp.Matrix([[0]])
        
        ss_system = SymbolicStateSpace(A, B, C, D)
        ss_system.variable = self.s
        
        result = self.converter.check_controllability(ss_system)
        
        self.assertIn('is_controllable', result)
        self.assertIn('rank', result)
        self.assertIn('controllability_matrix', result)
        self.assertEqual(result['expected_rank'], 2)
    
    def test_observability_check(self):
        """Teste de verificação de observabilidade"""
        # Sistema observável
        A = sp.Matrix([[-1, -2], [1, 0]])
        B = sp.Matrix([[1], [0]])
        C = sp.Matrix([[0, 1]])
        D = sp.Matrix([[0]])
        
        ss_system = SymbolicStateSpace(A, B, C, D)
        ss_system.variable = self.s
        
        result = self.converter.check_observability(ss_system)
        
        self.assertIn('is_observable', result)
        self.assertIn('rank', result)
        self.assertIn('observability_matrix', result)
        self.assertEqual(result['expected_rank'], 2)


@unittest.skipIf(not SYMPY_AVAILABLE, "SymPy não disponível")
class TestBodeAsymptoticAnalyzer(unittest.TestCase):
    """Testes para BodeAsymptoticAnalyzer"""
    
    def setUp(self):
        self.analyzer = BodeAsymptoticAnalyzer()
        self.s = sp.Symbol('s')
    
    def test_tf_factors_analysis(self):
        """Teste de análise de fatores da TF"""
        # Sistema: G(s) = 10*(s+2)/((s+1)*(s+5))
        tf_system = SymbolicTransferFunction(
            10 * (self.s + 2), 
            (self.s + 1) * (self.s + 5), 
            self.s
        )
        
        result = self.analyzer.analyze_tf_factors(tf_system)
        
        self.assertIn('bode_factors', result)
        self.assertIn('static_gain', result)
        self.assertIn('zeros_analysis', result)
        self.assertIn('poles_analysis', result)
        
        # Deve identificar ganho, zero em s=-2, polos em s=-1 e s=-5
        self.assertGreater(len(result['bode_factors']), 0)
    
    def test_system_type_determination(self):
        """Teste de determinação do tipo do sistema"""
        # Sistema tipo 1: G(s) = 10/(s*(s+1))
        tf_system = SymbolicTransferFunction(10, self.s * (self.s + 1), self.s)
        
        result = self.analyzer.analyze_tf_factors(tf_system)
        
        self.assertEqual(result['system_type']['integrator_order'], 1)
        self.assertEqual(result['system_type']['system_type'], 1)
    
    def test_asymptotic_bode_generation(self):
        """Teste de geração de Bode assintótico"""
        # Sistema simples: G(s) = 100/(s+10)
        tf_system = SymbolicTransferFunction(100, self.s + 10, self.s)
        
        result = self.analyzer.generate_asymptotic_bode(tf_system)
        
        self.assertIn('frequencies', result)
        self.assertIn('magnitude_db', result)
        self.assertIn('phase_deg', result)
        self.assertIn('corner_frequencies', result)
        self.assertIn('breakpoints', result)
        
        # Verificar se tem dados válidos
        self.assertGreater(len(result['frequencies']), 0)
        self.assertEqual(len(result['frequencies']), len(result['magnitude_db']))
    
    def test_bode_rules_summary(self):
        """Teste de resumo das regras de Bode"""
        # Sistema: G(s) = 20/(s*(s+5))
        tf_system = SymbolicTransferFunction(20, self.s * (self.s + 5), self.s)
        
        bode_data = self.analyzer.generate_asymptotic_bode(tf_system)
        rules = self.analyzer.get_bode_rules_summary(bode_data)
        
        self.assertIn('initial_slope_db_decade', rules)
        self.assertIn('final_slope_db_decade', rules)
        self.assertIn('slope_changes', rules)
        self.assertIn('rules_applied', rules)
        self.assertIn('bode_construction_steps', rules)


@unittest.skipIf(not SYMPY_AVAILABLE, "SymPy não disponível")
class TestExpandedNumericalInterface(unittest.TestCase):
    """Testes para interface expandida"""
    
    def setUp(self):
        self.interface = NumericalInterface()
        self.s = sp.Symbol('s')
    
    def test_enhanced_summary(self):
        """Teste do resumo expandido"""
        summary = self.interface.get_enhanced_summary()
        
        self.assertIn('basic_conversions', summary)
        self.assertIn('analysis_methods', summary)
        self.assertIn('performance_analysis', summary)
        self.assertIn('state_space_advanced', summary)
        self.assertIn('bode_asymptotic', summary)
        
        # Verificar se novos métodos estão listados
        self.assertIn('analyze_steady_state_error', summary['performance_analysis'])
        self.assertIn('tf_to_ss_controllable_canonical', summary['state_space_advanced'])
        self.assertIn('generate_asymptotic_bode', summary['bode_asymptotic'])
        
        # Verificar contagem total
        self.assertEqual(summary['total_methods'], 29)
    
    def test_integration_performance_analysis(self):
        """Teste de integração - análise de desempenho"""
        # Sistema de segunda ordem
        tf_system = SymbolicTransferFunction(25, self.s**2 + 2*self.s + 25, self.s)
        
        # Teste de erro steady-state
        error_analysis = self.interface.analyze_steady_state_error(tf_system, 'step')
        self.assertIn('steady_state_error', error_analysis)
        
        # Teste de parâmetros de 2ª ordem
        second_order = self.interface.analyze_second_order_parameters(tf_system)
        self.assertIn('natural_frequency_wn', second_order)
        self.assertIn('damping_ratio_zeta', second_order)
        
        # Teste de resumo completo
        summary = self.interface.get_performance_summary(tf_system)
        self.assertIn('steady_state_analysis', summary)
        self.assertIn('second_order_analysis', summary)
    
    def test_integration_state_space_conversions(self):
        """Teste de integração - conversões SS"""
        # Sistema: G(s) = 5/(s² + 4s + 3)
        tf_system = SymbolicTransferFunction(5, self.s**2 + 4*self.s + 3, self.s)
        
        # Conversão para forma controlável
        ss_controllable = self.interface.tf_to_ss_controllable_canonical(tf_system)
        self.assertIsInstance(ss_controllable, SymbolicStateSpace)
        
        # Verificar controlabilidade
        control_analysis = self.interface.check_controllability(ss_controllable)
        self.assertIn('is_controllable', control_analysis)
        
        # Conversão de volta para TF
        tf_recovered = self.interface.ss_to_tf_symbolic(ss_controllable)
        self.assertIsInstance(tf_recovered, SymbolicTransferFunction)
    
    def test_integration_bode_asymptotic(self):
        """Teste de integração - Bode assintótico"""
        # Sistema: G(s) = 50*(s+2)/((s+1)*(s+10))
        tf_system = SymbolicTransferFunction(
            50 * (self.s + 2), 
            (self.s + 1) * (self.s + 10), 
            self.s
        )
        
        # Análise de fatores
        factors = self.interface.analyze_tf_factors(tf_system)
        self.assertIn('bode_factors', factors)
        
        # Geração de Bode assintótico
        bode_data = self.interface.generate_asymptotic_bode(tf_system)
        self.assertIn('frequencies', bode_data)
        self.assertIn('magnitude_db', bode_data)
        
        # Regras de construção
        rules = self.interface.get_bode_construction_rules(tf_system)
        self.assertIn('initial_slope_db_decade', rules)
        self.assertIn('bode_construction_steps', rules)


def run_expanded_tests():
    """Executa todos os testes dos módulos expandidos"""
    
    if not SYMPY_AVAILABLE:
        print("❌ SymPy não disponível - testes pulados")
        return False
    
    print("🧪 INICIANDO TESTES DOS MÓDULOS EXPANDIDOS")
    print("=" * 60)
    
    # Criar suite de testes
    test_suite = unittest.TestSuite()
    
    # Adicionar classes de teste
    test_classes = [
        TestPerformanceAnalyzer,
        TestStateSpaceConverter,
        TestBodeAsymptoticAnalyzer,
        TestExpandedNumericalInterface
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Executar testes
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Resumo
    print("\n" + "=" * 60)
    print("📊 RESUMO DOS TESTES EXPANDIDOS")
    print(f"✅ Testes executados: {result.testsRun}")
    print(f"❌ Falhas: {len(result.failures)}")
    print(f"⚠️  Erros: {len(result.errors)}")
    
    if result.failures:
        print("\n🔍 FALHAS:")
        for test, traceback in result.failures:
            print(f"  • {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print("\n🚨 ERROS:")
        for test, traceback in result.errors:
            print(f"  • {test}: {traceback.split('Exception:')[-1].strip()}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    
    if success:
        print("\n🎉 TODOS OS TESTES DOS MÓDULOS EXPANDIDOS PASSARAM!")
    else:
        print(f"\n⚠️  {len(result.failures + result.errors)} teste(s) falharam")
    
    return success


if __name__ == "__main__":
    success = run_expanded_tests()
    exit(0 if success else 1)
