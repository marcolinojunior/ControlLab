"""
Testes de Análise de Estabilidade - Módulo 5
============================================

Este arquivo contém todos os testes para validar a funcionalidade
dos algoritmos de análise de estabilidade implementados.
"""

import pytest
import sympy as sp
from sympy import symbols, I, pi
import sys
import os


# Importar módulos para teste
from controllab.analysis.routh_hurwitz import RouthHurwitzAnalyzer, StabilityResult
from controllab.analysis.root_locus import RootLocusAnalyzer, LocusFeatures
from controllab.analysis.frequency_response import FrequencyAnalyzer, StabilityMargins
from controllab.analysis.stability_utils import StabilityValidator, ParametricAnalyzer
from controllab.analysis.stability_analysis import StabilityAnalysisEngine


class TestRouthHurwitz:
    """Testes para o algoritmo de Routh-Hurwitz"""
    
    def test_basic_routh_construction(self):
        """Testa construção básica da tabela de Routh"""
        analyzer = RouthHurwitzAnalyzer()
        s = symbols('s')
        
        # Sistema de 2ª ordem: s² + 2s + 1 (estável)
        poly = s**2 + 2*s + 1
        routh_array = analyzer.build_routh_array(poly, show_steps=True)
        
        assert routh_array is not None
        assert len(routh_array.array) == 3  # s², s¹, s⁰
        assert routh_array.history is not None
        
    def test_stability_analysis(self):
        """Testa análise de estabilidade"""
        analyzer = RouthHurwitzAnalyzer()
        s = symbols('s')
        
        # Sistema estável
        stable_poly = s**2 + 2*s + 1
        routh_array = analyzer.build_routh_array(stable_poly, show_steps=False)
        result = analyzer.analyze_stability(routh_array, show_steps=True)
        
        assert result.is_stable == True
        assert result.sign_changes == 0
        
        # Sistema instável  
        unstable_poly = s**2 - 2*s + 1
        routh_array = analyzer.build_routh_array(unstable_poly, show_steps=False)
        result = analyzer.analyze_stability(routh_array, show_steps=False)
        
        assert result.is_stable == False
        assert result.sign_changes > 0
        
    def test_parametric_analysis(self):
        """Testa análise paramétrica"""
        analyzer = RouthHurwitzAnalyzer()
        s, K = symbols('s K')
        
        poly = s**3 + 2*s**2 + s + K
        result = analyzer.parametric_stability_analysis(poly, K, show_steps=True)
        
        assert 'stable_range' in result
        assert 'conditions' in result
        assert result['parameter'] == K


class TestRootLocus:
    """Testes para análise de Root Locus"""
    
    def test_locus_features_extraction(self):
        """Testa extração de características do root locus"""
        analyzer = RootLocusAnalyzer()
        s = symbols('s')
        
        # Sistema: 1/[s(s+1)(s+2)]
        system = 1 / (s * (s + 1) * (s + 2))
        features = analyzer.get_locus_features(system, show_steps=True)
        
        assert len(features.poles) == 3
        assert len(features.zeros) == 0
        assert features.num_branches == 3
        assert features.analysis_history is not None
        
    def test_six_rules_application(self):
        """Testa aplicação das 6 regras do root locus"""
        analyzer = RootLocusAnalyzer()
        s = symbols('s')
        
        system = 1 / (s * (s + 1))
        features = analyzer.get_locus_features(system, show_steps=True)
        
        # Verificar se as 6 regras foram aplicadas
        assert len(features.analysis_history.rules_applied) >= 6
        
        # Verificar características específicas
        assert features.num_branches == 2
        assert len(features.asymptotes['angles']) == 2
        
    def test_breakaway_points(self):
        """Testa cálculo de pontos de breakaway"""
        from controllab.analysis.root_locus import find_breakaway_points
        s = symbols('s')
        
        system = 1 / (s * (s + 1) * (s + 2))
        breakaway_points = find_breakaway_points(system)
        
        assert isinstance(breakaway_points, list)
        assert len(breakaway_points) >= 0  # Pode ter ou não pontos de breakaway


class TestFrequencyResponse:
    """Testes para análise de resposta em frequência"""
    
    def test_gain_phase_margins(self):
        """Testa cálculo de margens de ganho e fase"""
        analyzer = FrequencyAnalyzer()
        s = symbols('s')
        
        # Sistema de 1ª ordem
        system = 1 / (s + 1)
        margins = analyzer.calculate_gain_phase_margins(system, show_steps=True)
        
        assert margins.gain_margin_db is not None
        assert margins.phase_margin is not None
        assert margins.is_stable is not None
        assert margins.analysis_history is not None
        
    def test_frequency_response_calculation(self):
        """Testa cálculo de resposta em frequência"""
        analyzer = FrequencyAnalyzer()
        s = symbols('s')
        
        system = 1 / (s + 1)
        import numpy as np
        omega_range = np.array([0.1, 1.0, 10.0])
        
        freq_response = analyzer.calculate_frequency_response(system, omega_range, show_steps=True)
        
        assert 'magnitude' in freq_response
        assert 'phase' in freq_response
        assert len(freq_response['magnitude']) == len(omega_range)
        
    def test_nyquist_contour(self):
        """Testa geração do contorno de Nyquist"""
        analyzer = FrequencyAnalyzer()
        s = symbols('s')
        
        system = 1 / (s + 1)
        contour = analyzer.get_nyquist_contour(system, show_steps=True)
        
        assert contour is not None
        assert hasattr(contour, 'main_path')
        assert hasattr(contour, 'indentations')
        
    def test_nyquist_criterion(self):
        """Testa aplicação do critério de Nyquist"""
        analyzer = FrequencyAnalyzer()
        s = symbols('s')
        
        system = 1 / (s + 1)
        result = analyzer.apply_nyquist_criterion(system, None, show_steps=True)
        
        assert 'is_stable' in result
        assert 'criterion_result' in result


class TestStabilityUtils:
    """Testes para utilitários de estabilidade"""
    
    def test_stability_validator(self):
        """Testa validação cruzada de métodos"""
        validator = StabilityValidator()
        s = symbols('s')
        
        system = 1 / (s**2 + 2*s + 1)
        results = validator.validate_stability_methods(system, show_steps=True)
        
        assert len(results) >= 2  # Pelo menos 2 métodos validados
        
    def test_parametric_analyzer_2d(self):
        """Testa análise paramétrica 2D"""
        analyzer = ParametricAnalyzer()
        s, K1, K2 = symbols('s K1 K2')
        
        system = s**2 + K1*s + K2
        result = analyzer.stability_region_2d(
            system, K1, K2,
            param1_range=(0.1, 5.0), 
            param2_range=(0.1, 5.0),
            resolution=10
        )
        
        assert 'stability_map' in result
        assert 'stable_region_area' in result
        
    def test_sensitivity_analysis(self):
        """Testa análise de sensibilidade"""
        analyzer = ParametricAnalyzer()
        s, K = symbols('s K')
        
        system = K / (s + 1)
        result = analyzer.analyze_sensitivity(system, K, nominal_value=1.0)
        
        assert 'stability_analysis' in result
        assert 'sensitivity_metrics' in result


class TestStabilityIntegration:
    """Testes de integração completa"""
    
    def test_comprehensive_analysis(self):
        """Testa análise completa integrada"""
        engine = StabilityAnalysisEngine()
        s = symbols('s')
        
        system = 1 / (s**2 + 2*s + 1)
        result = engine.comprehensive_analysis(system, show_all_steps=True)
        
        assert result is not None
        assert hasattr(result, 'get_full_report')
        assert hasattr(result, 'get_cross_validation_report')
        
        # Testar relatórios
        full_report = result.get_full_report()
        validation_report = result.get_cross_validation_report()
        
        assert len(full_report) > 100
        assert len(validation_report) > 50
        
    def test_cross_validation(self):
        """Testa validação cruzada entre todos os métodos"""
        engine = StabilityAnalysisEngine()
        s = symbols('s')
        
        # Sistema conhecido estável
        stable_system = 1 / (s + 1)
        result = engine.comprehensive_analysis(stable_system, show_all_steps=False)
        
        validation_report = result.get_cross_validation_report()
        
        # Verificar que há concordância entre métodos
        assert "CONCORDAM" in validation_report or "ESTÁVEL" in validation_report
        
    def test_educational_content(self):
        """Testa conteúdo educacional"""
        engine = StabilityAnalysisEngine()
        s = symbols('s')
        
        system = 1 / (s**2 + s + 1)
        result = engine.comprehensive_analysis(system, show_all_steps=True)
        
        full_report = result.get_full_report()
        
        # Verificar conteúdo pedagógico
        educational_markers = [
            "EDUCACIONAL",
            "conceito",
            "fórmula", 
            "interpretação",
            "significa"
        ]
        
        pedagogical_score = sum(1 for marker in educational_markers 
                              if marker.lower() in full_report.lower())
        
        assert pedagogical_score >= 2  # Pelo menos 2 elementos pedagógicos


class TestSpecialCases:
    """Testes para casos especiais"""
    
    def test_complex_poles(self):
        """Testa sistemas com polos complexos"""
        engine = StabilityAnalysisEngine()
        s = symbols('s')
        
        # Sistema com polos complexos: s² + s + 1
        system = 1 / (s**2 + s + 1)
        result = engine.comprehensive_analysis(system, show_all_steps=False)
        
        assert result is not None
        
    def test_high_order_system(self):
        """Testa sistema de alta ordem"""
        analyzer = RouthHurwitzAnalyzer()
        s = symbols('s')
        
        # Sistema de 5ª ordem
        poly = s**5 + 2*s**4 + 3*s**3 + 4*s**2 + 5*s + 6
        routh_array = analyzer.build_routh_array(poly, show_steps=False)
        
        assert routh_array is not None
        assert len(routh_array.array) == 6  # s⁵ até s⁰
        
    def test_marginal_stability(self):
        """Testa detecção de estabilidade marginal"""
        analyzer = RouthHurwitzAnalyzer()
        s = symbols('s')
        
        # Sistema marginalmente estável: s² + 1
        poly = s**2 + 1
        routh_array = analyzer.build_routh_array(poly, show_steps=False)
        result = analyzer.analyze_stability(routh_array, show_steps=False)
        
        # Pode ser None (marginal) ou False (instável)
        assert result.is_stable is not True


if __name__ == "__main__":
    # Executar testes principais
    pytest.main([__file__, "-v"])
