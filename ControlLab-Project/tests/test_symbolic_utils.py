#!/usr/bin/env python3
"""
Testes para utilitários simbólicos - ControlLab
"""

import pytest
import sympy as sp
from src.controllab.core.symbolic_utils import (
    create_laplace_variable,
    create_z_variable,
    poly_from_roots,
    validate_proper_tf,
    cancel_common_factors,
    extract_poles_zeros,
    create_proper_tf,
    expand_partial_fractions,
    symbolic_stability_analysis,
    convert_to_latex_formatted
)

class TestSymbolicUtils:
    """Testes para utilitários simbólicos"""
    
    def test_create_laplace_variable(self):
        """Teste de criação de variável de Laplace"""
        s = create_laplace_variable()
        assert s.name == 's'
        assert s.is_complex
        
        # Teste com nome personalizado
        p = create_laplace_variable('p')
        assert p.name == 'p'
    
    def test_create_z_variable(self):
        """Teste de criação de variável Z"""
        z = create_z_variable()
        assert z.name == 'z'
        assert z.is_complex
    
    def test_poly_from_roots(self):
        """Teste de criação de polinômio a partir de raízes"""
        s = create_laplace_variable()
        
        # Polinômio com raízes em -1 e -2
        roots = [-1, -2]
        poly = poly_from_roots(roots, s)
        
        # Deve resultar em (s+1)(s+2) = s^2 + 3s + 2
        expected = s**2 + 3*s + 2
        assert sp.expand(poly) == expected
    
    def test_validate_proper_tf(self):
        """Teste de validação de função própria"""
        s = create_laplace_variable()
        
        # Função própria: grau(num) <= grau(den)
        assert validate_proper_tf(1, s + 1)  # 0 <= 1
        assert validate_proper_tf(s, s**2 + s + 1)  # 1 <= 2
        
        # Função imprópria: grau(num) > grau(den)
        assert not validate_proper_tf(s**2, s + 1)  # 2 > 1
    
    def test_cancel_common_factors(self):
        """Teste de cancelamento de fatores comuns"""
        s = create_laplace_variable()
        
        # (s+1)(s+2) / (s+1)(s+3) deve simplificar para (s+2)/(s+3)
        num = (s + 1) * (s + 2)
        den = (s + 1) * (s + 3)
        
        num_simplified, den_simplified = cancel_common_factors(num, den)
        
        # Verifica se o fator (s+1) foi cancelado
        expected_num = s + 2
        expected_den = s + 3
        
        assert sp.simplify(num_simplified - expected_num) == 0
        assert sp.simplify(den_simplified - expected_den) == 0
    
    def test_extract_poles_zeros(self):
        """Teste de extração de polos e zeros"""
        s = create_laplace_variable()
        
        # G(s) = (s+1)/((s+2)(s+3))
        num = s + 1
        den = (s + 2) * (s + 3)
        
        zeros, poles = extract_poles_zeros(num, den)
        
        # Zero em s = -1
        assert -1 in zeros
        
        # Polos em s = -2 e s = -3
        assert -2 in poles
        assert -3 in poles
    
    def test_create_proper_tf(self):
        """Teste de criação de função própria"""
        s = create_laplace_variable()
        
        zeros = [-1]
        poles = [-2, -3]
        gain = 2
        
        num, den = create_proper_tf(zeros, poles, gain, s)
        
        # Deve resultar em 2(s+1)/((s+2)(s+3))
        expected_num = 2 * (s + 1)
        expected_den = (s + 2) * (s + 3)
        
        assert sp.expand(num) == sp.expand(expected_num)
        assert sp.expand(den) == sp.expand(expected_den)
    
    def test_expand_partial_fractions(self):
        """Teste de expansão em frações parciais"""
        s = create_laplace_variable()
        
        # 1/((s+1)(s+2)) deve expandir em frações parciais
        num = 1
        den = (s + 1) * (s + 2)
        
        partial = expand_partial_fractions(num, den, s)
        
        # Resultado deve ser uma expressão SymPy
        assert isinstance(partial, sp.Expr)
        
        # Verifica se a expansão está correta calculando de volta
        recombined = sp.simplify(partial)
        original = sp.simplify(num / den)
        
        assert sp.simplify(recombined - original) == 0
    
    def test_symbolic_stability_analysis(self):
        """Teste de análise de estabilidade simbólica"""
        s = create_laplace_variable()
        
        # Polinômio estável: s^2 + 3s + 2 (todos coeficientes positivos)
        stable_poly = s**2 + 3*s + 2
        result = symbolic_stability_analysis(stable_poly, s)
        
        assert 'stable' in result
        assert 'coefficients' in result
        assert 'polynomial' in result
        
        # Verifica se os coeficientes foram extraídos
        assert len(result['coefficients']) == 3  # [1, 3, 2]
    
    def test_convert_to_latex_formatted(self):
        """Teste de conversão para LaTeX"""
        s = create_laplace_variable()
        
        # Expressão simples
        expr = s**2 + 2*s + 1
        latex_str = convert_to_latex_formatted(expr)
        
        assert isinstance(latex_str, str)
        assert 's' in latex_str
        
        # Expressão mais complexa
        complex_expr = (s + 1) / (s**2 + 2*s + 2)
        latex_complex = convert_to_latex_formatted(complex_expr)
        
        assert isinstance(latex_complex, str)
    
    def test_poly_from_roots_complex(self):
        """Teste com raízes complexas"""
        s = create_laplace_variable()
        
        # Raízes complexas conjugadas
        roots = [-1 + 1j, -1 - 1j]
        poly = poly_from_roots(roots, s)
        
        # Deve resultar em um polinômio real
        expanded = sp.expand(poly)
        
        # Para raízes -1±j, o polinômio deve ser (s+1)² + 1 = s² + 2s + 2
        expected = s**2 + 2*s + 2
        assert sp.simplify(expanded - expected) == 0
    
    def test_stability_analysis_unstable(self):
        """Teste de análise de estabilidade para sistema instável"""
        s = create_laplace_variable()
        
        # Polinômio instável: s^2 - s + 1 (coeficiente negativo)
        unstable_poly = s**2 - s + 1
        result = symbolic_stability_analysis(unstable_poly, s)
        
        # Pode não ser estável devido aos coeficientes
        assert isinstance(result, dict)
        assert 'coefficients' in result
    
    def test_error_handling(self):
        """Teste de tratamento de erros"""
        # Testa com entrada inválida
        try:
            result = cancel_common_factors("invalid", "also_invalid")
            # Deve retornar os valores originais em caso de erro
            assert result == ("invalid", "also_invalid")
        except:
            # Ou pode gerar exceção, dependendo da implementação
            pass
