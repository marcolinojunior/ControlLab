"""
Testes de integração para análise de estabilidade
==============================================

Este arquivo testa a integração entre todos os módulos de análise de estabilidade,
incluindo validação cruzada e consistência entre métodos.
"""

import unittest
import sympy as sp
import sys
from pathlib import Path

# Adicionar src ao path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from controllab.analysis import (
    StabilityAnalysisEngine,
    analyze_stability,
    quick_stability_check,
    validate_stability_methods
)
from controllab.core import SymbolicTransferFunction


class TestStabilityIntegration(unittest.TestCase):
    """Testes de integração entre métodos de análise"""
    
    def setUp(self):
        self.engine = StabilityAnalysisEngine()
        self.s = sp.Symbol('s')
        
    def test_comprehensive_analysis(self):
        """Teste análise completa integrada"""
        G = SymbolicTransferFunction(1, self.s**3 + 2*self.s**2 + 3*self.s + 1)
        
        report = self.engine.comprehensive_analysis(G, show_all_steps=True)
        
        self.assertIsNotNone(report)
        self.assertIsNotNone(report.routh_hurwitz_results)
        self.assertIsNotNone(report.root_locus_results)
        self.assertIsNotNone(report.frequency_response_results)
        
    def test_analyze_complete_stability(self):
        """Teste método analyze_complete_stability"""
        G = SymbolicTransferFunction(1, self.s**2 + 2*self.s + 1)
        
        report = self.engine.analyze_complete_stability(G, show_steps=True)
        
        self.assertIsNotNone(report)
        
    def test_quick_stability_check_integration(self):
        """Teste verificação rápida integrada"""
        G = SymbolicTransferFunction(1, self.s + 1)
        
        result = self.engine.quick_stability_check(G)
        
        self.assertIsNotNone(result)
        self.assertIn('is_stable', result)
        self.assertIn('method_used', result)
        
    def test_comparative_analysis(self):
        """Teste análise comparativa entre sistemas"""
        G1 = SymbolicTransferFunction(1, self.s + 1)  # Estável
        G2 = SymbolicTransferFunction(1, self.s - 1)  # Instável
        
        comparison = self.engine.comparative_analysis([G1, G2], ['Estável', 'Instável'])
        
        self.assertIsNotNone(comparison)
        self.assertIn('systems', comparison)


class TestCrossValidation(unittest.TestCase):
    """Testes de validação cruzada entre métodos"""
    
    def setUp(self):
        self.s = sp.Symbol('s')
        
    def test_validation_between_methods(self):
        """Teste validação entre métodos de análise"""
        G = SymbolicTransferFunction(1, self.s**2 + 2*self.s + 1)
        
        validation = validate_stability_methods(G, show_steps=True)
        
        self.assertIsNotNone(validation)
        self.assertIn('routh_hurwitz', validation)
        self.assertIn('root_analysis', validation)
        self.assertIn('frequency_analysis', validation)
        
    def test_consistency_stable_system(self):
        """Teste consistência para sistema estável"""
        G = SymbolicTransferFunction(1, self.s + 1)
        
        validation = validate_stability_methods(G)
        
        # Todos os métodos devem concordar que é estável
        routh_stable = validation['routh_hurwitz'].is_stable
        root_stable = validation['root_analysis']['is_stable']
        
        self.assertEqual(routh_stable, root_stable)
        
    def test_consistency_unstable_system(self):
        """Teste consistência para sistema instável"""
        G = SymbolicTransferFunction(1, self.s - 1)  # Polo no semiplano direito
        
        validation = validate_stability_methods(G)
        
        # Todos os métodos devem concordar que é instável
        routh_stable = validation['routh_hurwitz'].is_stable
        root_stable = validation['root_analysis']['is_stable']
        
        self.assertEqual(routh_stable, root_stable)
        self.assertFalse(routh_stable)


class TestConvenienceFunctions(unittest.TestCase):
    """Testes das funções de conveniência"""
    
    def setUp(self):
        self.s = sp.Symbol('s')
        
    def test_analyze_stability_function(self):
        """Teste função analyze_stability"""
        G = SymbolicTransferFunction(1, self.s + 1)
        
        result = analyze_stability(G, show_steps=True)
        
        self.assertIsNotNone(result)
        
    def test_quick_stability_check_function(self):
        """Teste função quick_stability_check"""
        G = SymbolicTransferFunction(1, self.s + 1)
        
        is_stable = quick_stability_check(G)
        
        self.assertIsInstance(is_stable, bool)
        self.assertTrue(is_stable)  # Sistema de primeira ordem deve ser estável


class TestHighOrderSystems(unittest.TestCase):
    """Testes com sistemas de alta ordem"""
    
    def setUp(self):
        self.engine = StabilityAnalysisEngine()
        self.s = sp.Symbol('s')
        
    def test_high_order_stable_system(self):
        """Teste sistema estável de alta ordem"""
        # Sistema de 5ª ordem estável
        G = SymbolicTransferFunction(
            1, 
            self.s**5 + 5*self.s**4 + 10*self.s**3 + 10*self.s**2 + 5*self.s + 1
        )
        
        report = self.engine.comprehensive_analysis(G)
        
        self.assertIsNotNone(report)
        
    def test_high_order_unstable_system(self):
        """Teste sistema instável de alta ordem"""
        # Sistema de 4ª ordem com alguns coeficientes negativos
        G = SymbolicTransferFunction(
            1, 
            self.s**4 - self.s**3 + 2*self.s**2 + self.s + 1
        )
        
        report = self.engine.comprehensive_analysis(G)
        
        self.assertIsNotNone(report)


class TestParametricSystems(unittest.TestCase):
    """Testes com sistemas paramétricos"""
    
    def setUp(self):
        self.engine = StabilityAnalysisEngine()
        self.s, self.K = sp.symbols('s K')
        
    def test_parametric_stability_analysis(self):
        """Teste análise de estabilidade paramétrica"""
        G = SymbolicTransferFunction(self.K, self.s**3 + 2*self.s**2 + self.s + self.K)
        
        report = self.engine.comprehensive_analysis(G, include_parametric=True)
        
        self.assertIsNotNone(report)
        
    def test_parametric_validation(self):
        """Teste validação de sistema paramétrico"""
        poly = self.s**3 + 2*self.s**2 + self.s + self.K
        
        validation = validate_stability_methods(poly)
        
        self.assertIsNotNone(validation)


class TestEdgeCases(unittest.TestCase):
    """Testes com casos extremos"""
    
    def setUp(self):
        self.engine = StabilityAnalysisEngine()
        self.s = sp.Symbol('s')
        
    def test_marginally_stable_system(self):
        """Teste sistema marginalmente estável"""
        # Sistema com polos no eixo jω
        G = SymbolicTransferFunction(1, self.s**2 + 1)
        
        report = self.engine.comprehensive_analysis(G)
        
        self.assertIsNotNone(report)
        
    def test_pure_integrator(self):
        """Teste integrador puro"""
        G = SymbolicTransferFunction(1, self.s)
        
        report = self.engine.comprehensive_analysis(G)
        
        self.assertIsNotNone(report)
        
    def test_first_order_systems(self):
        """Teste diferentes sistemas de primeira ordem"""
        systems = [
            SymbolicTransferFunction(1, self.s + 1),    # Estável
            SymbolicTransferFunction(1, self.s - 1),    # Instável
            SymbolicTransferFunction(1, self.s + 0.1),  # Lento mas estável
        ]
        
        for G in systems:
            with self.subTest(system=G):
                report = self.engine.comprehensive_analysis(G)
                self.assertIsNotNone(report)


if __name__ == '__main__':
    unittest.main()
