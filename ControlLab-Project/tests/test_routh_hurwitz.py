"""
Testes específicos para o módulo Routh-Hurwitz
==============================================

Este arquivo testa todos os aspectos do critério de Routh-Hurwitz,
incluindo casos especiais e análise pedagógica.
"""

import unittest
import sympy as sp
import sys
from pathlib import Path

# Adicionar src ao path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from controllab.analysis.routh_hurwitz import (
    RouthHurwitzAnalyzer,
    build_routh_array,
    analyze_stability,
    handle_zero_in_first_column,
    handle_row_of_zeros
)


class TestRouthHurwitzBasic(unittest.TestCase):
    """Testes básicos do critério de Routh-Hurwitz"""

    def setUp(self):
        self.analyzer = RouthHurwitzAnalyzer()
        self.s = sp.Symbol('s')

    def test_stable_system(self):
        """Teste com sistema estável"""
        # s³ + 3s² + 3s + 1 (estável)
        poly = self.s**3 + 3*self.s**2 + 3*self.s + 1

        result = self.analyzer.analyze(poly)

        self.assertEqual(result['conclusion'], 'ESTÁVEL')

    def test_unstable_system(self):
        """Teste com sistema instável"""
        # s³ - 2s² + s + 1 (instável)
        poly = self.s**3 - 2*self.s**2 + self.s + 1

        result = self.analyzer.analyze(poly)

        self.assertIn('INSTÁVEL', result['conclusion'])


class TestRouthHurwitzSpecialCases(unittest.TestCase):
    """Testes dos casos especiais do Routh-Hurwitz"""

    def setUp(self):
        self.analyzer = RouthHurwitzAnalyzer()
        self.s = sp.Symbol('s')

    def test_zero_in_first_column(self):
        """Teste caso especial: zero na primeira coluna"""
        # s⁴ + s³ + 2s² + 2s + 3 (tem zero na primeira coluna)
        poly = self.s**4 + self.s**3 + 2*self.s**2 + 2*self.s + 3

        result = self.analyzer.analyze(poly)

        # Deve lidar com o caso especial
        self.assertIsNotNone(result)

    def test_row_of_zeros(self):
        """Teste caso especial: linha de zeros"""
        # s⁴ + 2s³ + 3s² + 2s + 1 (pode ter linha de zeros)
        poly = self.s**4 + 2*self.s**3 + 3*self.s**2 + 2*self.s + 1

        result = self.analyzer.analyze(poly)

        # Deve lidar com o caso especial
        self.assertIsNotNone(result)


class TestRouthHurwitzParametric(unittest.TestCase):
    """Testes com sistemas paramétricos"""

    def setUp(self):
        self.analyzer = RouthHurwitzAnalyzer()
        self.s, self.K = sp.symbols('s K')

    def test_parametric_stability(self):
        """Teste análise paramétrica"""
        # s³ + 2s² + s + K
        poly = self.s**3 + 2*self.s**2 + self.s + self.K

        result = self.analyzer.analyze(poly)

        # Deve ter informações sobre range de estabilidade
        self.assertIn('steps', result)


class TestRouthHurwitzPedagogical(unittest.TestCase):
    """Testes das funcionalidades pedagógicas"""

    def setUp(self):
        self.analyzer = RouthHurwitzAnalyzer()
        self.s = sp.Symbol('s')

    def test_step_by_step_analysis(self):
        """Teste análise passo a passo"""
        poly = self.s**3 + 2*self.s**2 + 3*self.s + 1

        result = self.analyzer.analyze(poly, show_steps=True)

        # Deve ter histórico pedagógico
        self.assertIn('steps', result)
        self.assertIn('summary', result)
        self.assertIn('conclusion', result)

    def test_standalone_functions(self):
        """Teste funções standalone"""
        poly = self.s**3 + 2*self.s**2 + 3*self.s + 1

        # Testar função build_routh_array standalone
        routh_array = build_routh_array(poly)
        self.assertIsNotNone(routh_array)

        # Testar função analyze_stability standalone
        result = analyze_stability(poly)
        self.assertIsNotNone(result)
        self.assertIn('conclusion', result)


if __name__ == '__main__':
    unittest.main()
