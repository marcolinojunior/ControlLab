"""
Testes específicos para o módulo Frequency Response
================================================

Este arquivo testa todas as funcionalidades de análise de resposta em frequência,
incluindo critério de Nyquist e margens de estabilidade.
"""

import unittest
import sympy as sp
import numpy as np
import sys
from pathlib import Path

# Adicionar src ao path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from controllab.analysis.frequency_response import (
    FrequencyAnalyzer,
    get_nyquist_contour,
    calculate_frequency_response,
    apply_nyquist_criterion,
    calculate_gain_phase_margins
)
from controllab.core import SymbolicTransferFunction


class TestFrequencyResponseBasic(unittest.TestCase):
    """Testes básicos de resposta em frequência"""
    
    def setUp(self):
        self.analyzer = FrequencyAnalyzer()
        self.s = sp.Symbol('s')
        
    def test_simple_first_order(self):
        """Teste com sistema de primeira ordem G(s) = 1/(s+1)"""
        G = SymbolicTransferFunction(1, self.s + 1)
        
        omega_range = np.logspace(-2, 2, 50)
        freq_response = self.analyzer.calculate_frequency_response(G, omega_range)
        
        self.assertIsNotNone(freq_response)
        self.assertIsNotNone(freq_response.magnitude)
        self.assertIsNotNone(freq_response.phase)
        
    def test_second_order_system(self):
        """Teste com sistema de segunda ordem"""
        G = SymbolicTransferFunction(1, self.s**2 + 2*self.s + 1)
        
        omega_range = np.logspace(-2, 2, 50)
        freq_response = self.analyzer.calculate_frequency_response(G, omega_range)
        
        self.assertIsNotNone(freq_response)


class TestNyquistCriterion(unittest.TestCase):
    """Testes do critério de Nyquist"""
    
    def setUp(self):
        self.analyzer = FrequencyAnalyzer()
        self.s = sp.Symbol('s')
        
    def test_nyquist_contour_construction(self):
        """Teste construção do contorno de Nyquist"""
        G = SymbolicTransferFunction(1, self.s + 1)
        
        contour = self.analyzer.get_nyquist_contour(G, radius=100, epsilon=1e-3)
        
        self.assertIsNotNone(contour)
        self.assertIsNotNone(contour.real_part)
        self.assertIsNotNone(contour.imag_part)
        
    def test_nyquist_criterion_application(self):
        """Teste aplicação do critério de Nyquist"""
        G = SymbolicTransferFunction(1, self.s + 1)
        
        contour = self.analyzer.get_nyquist_contour(G)
        nyquist_result = self.analyzer.apply_nyquist_criterion(G, contour)
        
        self.assertIsNotNone(nyquist_result)
        self.assertIn('encirclements', nyquist_result)
        self.assertIn('is_stable', nyquist_result)
        
    def test_poles_on_jw_axis(self):
        """Teste tratamento de polos no eixo jω"""
        # Sistema com polo na origem
        G = SymbolicTransferFunction(1, self.s*(self.s + 1))
        
        contour = self.analyzer.get_nyquist_contour(G, show_steps=True)
        
        # Deve lidar com polo na origem (indentação)
        self.assertIsNotNone(contour)


class TestStabilityMargins(unittest.TestCase):
    """Testes das margens de estabilidade"""
    
    def setUp(self):
        self.analyzer = FrequencyAnalyzer()
        self.s = sp.Symbol('s')
        
    def test_gain_phase_margins(self):
        """Teste cálculo de margens de ganho e fase"""
        G = SymbolicTransferFunction(1, self.s*(self.s + 1)*(self.s + 2))
        
        margins = self.analyzer.calculate_gain_phase_margins(G, show_steps=True)
        
        self.assertIsNotNone(margins)
        self.assertIsNotNone(margins.gain_margin)
        self.assertIsNotNone(margins.phase_margin)
        self.assertIsNotNone(margins.gain_margin_db)
        
    def test_stable_system_margins(self):
        """Teste margens de sistema estável"""
        G = SymbolicTransferFunction(1, self.s + 1)
        
        margins = self.analyzer.calculate_gain_phase_margins(G)
        
        # Sistema de primeira ordem deve ser estável
        self.assertTrue(margins.is_stable)
        
    def test_marginally_stable_system(self):
        """Teste sistema marginalmente estável"""
        # Sistema com integrador puro
        G = SymbolicTransferFunction(1, self.s)
        
        margins = self.analyzer.calculate_gain_phase_margins(G)
        
        self.assertIsNotNone(margins)


class TestFrequencyResponsePedagogical(unittest.TestCase):
    """Testes das funcionalidades pedagógicas"""
    
    def setUp(self):
        self.analyzer = FrequencyAnalyzer()
        self.s = sp.Symbol('s')
        
    def test_step_by_step_analysis(self):
        """Teste análise passo a passo"""
        G = SymbolicTransferFunction(1, self.s + 1)
        
        margins = self.analyzer.calculate_gain_phase_margins(G, show_steps=True)
        
        # Deve ter histórico pedagógico
        self.assertIsNotNone(margins.history)
        
    def test_standalone_functions(self):
        """Teste funções standalone"""
        G = SymbolicTransferFunction(1, self.s + 1)
        
        # Testar get_nyquist_contour standalone
        contour = get_nyquist_contour(G)
        self.assertIsNotNone(contour)
        
        # Testar calculate_frequency_response standalone
        omega_range = np.logspace(-1, 1, 20)
        freq_resp = calculate_frequency_response(G, omega_range)
        self.assertIsNotNone(freq_resp)
        
        # Testar apply_nyquist_criterion standalone
        nyquist = apply_nyquist_criterion(G)
        self.assertIsNotNone(nyquist)
        
        # Testar calculate_gain_phase_margins standalone
        margins = calculate_gain_phase_margins(G)
        self.assertIsNotNone(margins)


class TestAdvancedFrequencyResponse(unittest.TestCase):
    """Testes avançados de resposta em frequência"""
    
    def setUp(self):
        self.analyzer = FrequencyAnalyzer()
        self.s = sp.Symbol('s')
        
    def test_complex_system(self):
        """Teste com sistema complexo de alta ordem"""
        # Sistema de 4ª ordem
        G = SymbolicTransferFunction(
            1, 
            self.s**4 + 2*self.s**3 + 3*self.s**2 + 2*self.s + 1
        )
        
        margins = self.analyzer.calculate_gain_phase_margins(G)
        
        self.assertIsNotNone(margins)
        
    def test_system_with_zeros(self):
        """Teste sistema com zeros"""
        G = SymbolicTransferFunction(
            self.s + 2, 
            (self.s + 1)*(self.s + 3)
        )
        
        margins = self.analyzer.calculate_gain_phase_margins(G)
        
        self.assertIsNotNone(margins)


if __name__ == '__main__':
    unittest.main()
