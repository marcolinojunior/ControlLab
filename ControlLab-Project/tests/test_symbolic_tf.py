#!/usr/bin/env python3
"""
Testes para SymbolicTransferFunction - ControlLab
"""

import pytest
import sympy as sp
from src.controllab.core import SymbolicTransferFunction, create_laplace_variable

class TestSymbolicTransferFunction:
    """Testes para a classe SymbolicTransferFunction"""
    
    def setup_method(self):
        """Setup para cada teste"""
        self.s = create_laplace_variable()
        self.K = sp.Symbol('K', positive=True)
        self.T = sp.Symbol('T', positive=True)
    
    def test_creation(self):
        """Teste de criação de função de transferência"""
        # Teste básico
        tf = SymbolicTransferFunction(1, self.s + 1)
        assert tf.numerator == 1
        assert tf.denominator == self.s + 1
        assert tf.variable == self.s
        
        # Teste com parâmetros simbólicos
        tf2 = SymbolicTransferFunction(self.K, self.T * self.s + 1)
        assert tf2.numerator == self.K
        assert tf2.denominator == self.T * self.s + 1
    
    def test_string_representation(self):
        """Teste de representação em string"""
        tf = SymbolicTransferFunction(self.K, self.s + 1)
        str_repr = str(tf)
        assert 'K' in str_repr
        assert 's + 1' in str_repr
    
    def test_multiplication(self):
        """Teste de multiplicação de funções de transferência"""
        # G1(s) = K/(s+1)
        G1 = SymbolicTransferFunction(self.K, self.s + 1)
        
        # G2(s) = 1/(s+2)
        G2 = SymbolicTransferFunction(1, self.s + 2)
        
        # G(s) = G1 * G2 = K/((s+1)(s+2))
        G = G1 * G2
        
        expected_num = self.K
        expected_den = (self.s + 1) * (self.s + 2)
        
        assert sp.simplify(G.numerator - expected_num) == 0
        assert sp.simplify(G.denominator - expected_den) == 0
    
    def test_addition(self):
        """Teste de adição de funções de transferência"""
        # G1(s) = 1/(s+1)
        G1 = SymbolicTransferFunction(1, self.s + 1)
        
        # G2(s) = 1/(s+2)
        G2 = SymbolicTransferFunction(1, self.s + 2)
        
        # G(s) = G1 + G2 = (s+2 + s+1)/((s+1)(s+2)) = (2s+3)/((s+1)(s+2))
        G = G1 + G2
        
        # Verifica se o resultado está correto após simplificação
        expected_num = 2*self.s + 3
        expected_den = (self.s + 1) * (self.s + 2)
        
        # Simplifica ambos para comparar
        result_simplified = sp.simplify(G.numerator / G.denominator)
        expected_simplified = sp.simplify(expected_num / expected_den)
        
        assert sp.simplify(result_simplified - expected_simplified) == 0
    
    def test_scalar_multiplication(self):
        """Teste de multiplicação por escalar"""
        tf = SymbolicTransferFunction(1, self.s + 1)
        
        # Multiplica por 5
        tf_scaled = tf * 5
        
        assert tf_scaled.numerator == 5
        assert tf_scaled.denominator == self.s + 1
    
    def test_simplification(self):
        """Teste de simplificação"""
        # Cria função com fatores comuns: (s+1)(s+2) / (s+1)(s+3)
        num = (self.s + 1) * (self.s + 2)
        den = (self.s + 1) * (self.s + 3)
        
        tf = SymbolicTransferFunction(num, den)
        tf_simplified = tf.simplify()
        
        # Deve simplificar para (s+2)/(s+3)
        expected_num = self.s + 2
        expected_den = self.s + 3
        
        assert sp.simplify(tf_simplified.numerator - expected_num) == 0
        assert sp.simplify(tf_simplified.denominator - expected_den) == 0
    
    def test_substitution(self):
        """Teste de substituição de símbolos"""
        tf = SymbolicTransferFunction(self.K, self.T * self.s + 1)
        
        # Substitui K=2, T=0.5
        tf_sub = tf.substitute({self.K: 2, self.T: 0.5})
        
        assert tf_sub.numerator == 2
        assert tf_sub.denominator == 0.5 * self.s + 1
    
    def test_poles_zeros(self):
        """Teste de cálculo de polos e zeros"""
        # G(s) = (s+1)/((s+2)(s+3)) = (s+1)/(s^2+5s+6)
        num = self.s + 1
        den = (self.s + 2) * (self.s + 3)
        
        tf = SymbolicTransferFunction(num, den)
        
        zeros = tf.zeros()
        poles = tf.poles()
        
        # Zero em s = -1
        assert -1 in zeros
        
        # Polos em s = -2 e s = -3
        assert -2 in poles
        assert -3 in poles
    
    def test_partial_fractions(self):
        """Teste de expansão em frações parciais"""
        # G(s) = 1/((s+1)(s+2))
        den = (self.s + 1) * (self.s + 2)
        tf = SymbolicTransferFunction(1, den)
        
        # Expande em frações parciais
        partial = tf.partial_fractions()
        
        # Deve resultar em algo como 1/(s+1) - 1/(s+2)
        assert isinstance(partial, sp.Expr)
    
    def test_evaluation(self):
        """Teste de avaliação numérica"""
        tf = SymbolicTransferFunction(1, self.s + 1)
        
        # Avalia em s = 0
        value = tf.evaluate_at(0)
        assert value == 1.0
        
        # Avalia em s = -0.5
        value = tf.evaluate_at(-0.5)
        assert abs(value - 2.0) < 1e-10
    
    def test_proper_transfer_function(self):
        """Teste de verificação de função própria"""
        # Função própria: grau(num) <= grau(den)
        tf1 = SymbolicTransferFunction(1, self.s + 1)
        assert tf1.is_proper
        
        # Função imprópria: grau(num) > grau(den)
        tf2 = SymbolicTransferFunction(self.s**2, self.s + 1)
        assert not tf2.is_proper
    
    def test_latex_output(self):
        """Teste de conversão para LaTeX"""
        tf = SymbolicTransferFunction(self.K, self.s + 1)
        latex_str = tf.to_latex()
        
        assert isinstance(latex_str, str)
        assert 'K' in latex_str
        assert 's' in latex_str
    
    def test_history_tracking(self):
        """Teste de rastreamento de histórico"""
        tf1 = SymbolicTransferFunction(1, self.s + 1)
        tf2 = SymbolicTransferFunction(1, self.s + 2)
        
        # Operação que deve ser registrada
        tf_result = tf1 * tf2
        
        # Verifica se o histórico foi registrado
        assert len(tf_result.history.steps) > 0
        
        # Verifica se contém operação de multiplicação
        operations = [step.operation for step in tf_result.history.steps]
        assert "MULTIPLICAÇÃO" in operations
    
    def test_expansion_and_factoring(self):
        """Teste de expansão e fatoração"""
        # Função com expressão fatorada
        tf = SymbolicTransferFunction((self.s + 1), (self.s + 2) * (self.s + 3))
        
        # Expande denominador
        tf_expanded = tf.expand()
        expanded_den = sp.expand((self.s + 2) * (self.s + 3))
        assert sp.simplify(tf_expanded.denominator - expanded_den) == 0
        
        # Fatora de volta
        tf_factored = tf_expanded.factor()
        # Após fatorar, deve voltar ao formato original (ou equivalente)
        assert isinstance(tf_factored.denominator, sp.Expr)
    
    def test_division(self):
        """Teste de divisão de funções de transferência"""
        G1 = SymbolicTransferFunction(self.K, self.s + 1)
        G2 = SymbolicTransferFunction(1, self.s + 2)
        
        # G1 / G2 = K(s+2) / (s+1)
        G = G1 / G2
        
        expected_num = self.K * (self.s + 2)
        expected_den = self.s + 1
        
        assert sp.simplify(G.numerator - expected_num) == 0
        assert sp.simplify(G.denominator - expected_den) == 0
