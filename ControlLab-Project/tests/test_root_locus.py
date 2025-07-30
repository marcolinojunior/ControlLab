"""
Testes específicos para o módulo Root Locus
==========================================

Este arquivo testa todas as funcionalidades do lugar geométrico das raízes,
incluindo as 6 regras fundamentais e análise pedagógica.
"""

import unittest
import sympy as sp
import sys
from pathlib import Path

# Adicionar src ao path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from controllab.analysis.root_locus import (
    RootLocusAnalyzer,
    get_locus_features,
    calculate_asymptotes,
    find_breakaway_points,
    find_jw_crossings,
    calculate_locus_points
)
from controllab.core import SymbolicTransferFunction


class TestRootLocusBasic(unittest.TestCase):
    """Testes básicos do Root Locus"""
    
    def setUp(self):
        self.analyzer = RootLocusAnalyzer()
        self.s = sp.Symbol('s')
        
    def test_simple_system(self):
        """Teste com sistema simples G(s) = 1/(s*(s+1))"""
        G = SymbolicTransferFunction(1, self.s*(self.s+1))
        
        features = self.analyzer.get_locus_features(G, show_steps=True)
        
        self.assertIsNotNone(features)
        self.assertIsNotNone(features.asymptotes)
        
    def test_third_order_system(self):
        """Teste com sistema de terceira ordem G(s) = 1/(s*(s+1)*(s+2))"""
        G = SymbolicTransferFunction(1, self.s*(self.s+1)*(self.s+2))
        
        features = self.analyzer.get_locus_features(G, show_steps=True)
        
        self.assertIsNotNone(features)
        self.assertTrue(len(features.poles) == 3)


class TestRootLocusSixRules(unittest.TestCase):
    """Testes das 6 regras fundamentais do Root Locus"""
    
    def setUp(self):
        self.analyzer = RootLocusAnalyzer()
        self.s = sp.Symbol('s')
        
    def test_rule_1_start_end_points(self):
        """Regra 1: Pontos de partida e chegada"""
        G = SymbolicTransferFunction(1, self.s*(self.s+1)*(self.s+2))
        
        features = self.analyzer.get_locus_features(G)
        
        # Deve identificar polos (partida) e zeros (chegada)
        self.assertIsNotNone(features.poles)
        self.assertIsNotNone(features.zeros)
        
    def test_rule_2_number_of_branches(self):
        """Regra 2: Número de ramos"""
        G = SymbolicTransferFunction(1, self.s*(self.s+1)*(self.s+2))
        
        features = self.analyzer.get_locus_features(G)
        
        # Número de ramos = max(n_poles, n_zeros)
        expected_branches = max(len(features.poles), len(features.zeros))
        self.assertEqual(features.num_branches, expected_branches)
        
    def test_rule_3_asymptotes(self):
        """Regra 3: Assíntotas"""
        zeros = []
        poles = [0, -1, -2]
        
        asymptotes = calculate_asymptotes(zeros, poles)
        
        self.assertIn('angles', asymptotes)
        self.assertIn('centroid', asymptotes)
        
        # Para 3 polos e 0 zeros, devem ser 3 assíntotas
        self.assertEqual(len(asymptotes['angles']), 3)
        
    def test_rule_4_breakaway_points(self):
        """Regra 4: Pontos de breakaway/break-in"""
        G = SymbolicTransferFunction(1, self.s*(self.s+2))
        
        breakaway_points = find_breakaway_points(G)
        
        self.assertIsInstance(breakaway_points, list)
        # Para este sistema simples, deve ter pontos de breakaway
        
    def test_rule_5_jw_crossings(self):
        """Regra 5: Cruzamentos do eixo jω"""
        G = SymbolicTransferFunction(1, self.s**3 + 2*self.s**2 + self.s + 1)
        
        jw_crossings = find_jw_crossings(G)
        
        self.assertIsInstance(jw_crossings, list)
        
    def test_rule_6_departure_arrival_angles(self):
        """Regra 6: Ângulos de partida/chegada"""
        # Sistema com polos complexos
        G = SymbolicTransferFunction(1, self.s**2 + 2*self.s + 5)  # polos em -1±2j
        
        features = self.analyzer.get_locus_features(G)
        
        # Deve calcular ângulos para polos complexos
        self.assertIsNotNone(features.departure_angles)


class TestRootLocusPoints(unittest.TestCase):
    """Testes do cálculo de pontos do locus"""
    
    def setUp(self):
        self.analyzer = RootLocusAnalyzer()
        self.s = sp.Symbol('s')
        
    def test_locus_points_calculation(self):
        """Teste cálculo de pontos específicos do locus"""
        G = SymbolicTransferFunction(1, self.s*(self.s+1))
        
        k_range = [0.1, 0.5, 1.0, 2.0, 5.0]
        locus_points = self.analyzer.calculate_locus_points(G, k_range)
        
        self.assertIn('k_values', locus_points)
        self.assertIn('roots', locus_points)
        self.assertIn('locus_points', locus_points)
        
        # Deve ter calculado pontos para todos os valores de K
        self.assertEqual(len(locus_points['k_values']), len(k_range))


class TestRootLocusPedagogical(unittest.TestCase):
    """Testes das funcionalidades pedagógicas"""
    
    def setUp(self):
        self.analyzer = RootLocusAnalyzer()
        self.s = sp.Symbol('s')
        
    def test_step_by_step_analysis(self):
        """Teste análise passo a passo"""
        G = SymbolicTransferFunction(1, self.s*(self.s+1)*(self.s+2))
        
        features = self.analyzer.get_locus_features(G, show_steps=True)
        
        # Deve ter histórico detalhado
        self.assertIsNotNone(features.history)
        
    def test_standalone_functions(self):
        """Teste funções standalone"""
        G = SymbolicTransferFunction(1, self.s*(self.s+1))
        
        # Testar função get_locus_features standalone
        features = get_locus_features(G)
        self.assertIsNotNone(features)
        
        # Testar calculate_asymptotes standalone
        asymptotes = calculate_asymptotes([], [0, -1])
        self.assertIsNotNone(asymptotes)
        
        # Testar find_breakaway_points standalone
        breakaway = find_breakaway_points(G)
        self.assertIsInstance(breakaway, list)
        
        # Testar find_jw_crossings standalone
        jw_cross = find_jw_crossings(G)
        self.assertIsInstance(jw_cross, list)
        
        # Testar calculate_locus_points standalone
        points = calculate_locus_points(G, [1.0, 2.0])
        self.assertIsNotNone(points)


if __name__ == '__main__':
    unittest.main()
