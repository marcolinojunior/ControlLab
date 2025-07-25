#!/usr/bin/env python3
"""
Testes para SymbolicStateSpace - ControlLab
"""

import pytest
import sympy as sp
from src.controllab.core import SymbolicStateSpace, create_laplace_variable

class TestSymbolicStateSpace:
    """Testes para a classe SymbolicStateSpace"""
    
    def setup_method(self):
        """Setup para cada teste"""
        self.s = create_laplace_variable()
        self.a = sp.Symbol('a')
        self.b = sp.Symbol('b')
        self.c = sp.Symbol('c')
    
    def test_creation(self):
        """Teste de criação de sistema em espaço de estados"""
        # Sistema simples 1x1x1
        A = sp.Matrix([[-1]])
        B = sp.Matrix([[1]])
        C = sp.Matrix([[1]])
        D = sp.Matrix([[0]])
        
        ss = SymbolicStateSpace(A, B, C, D)
        
        assert ss.A == A
        assert ss.B == B
        assert ss.C == C
        assert ss.D == D
        assert ss.n_states == 1
        assert ss.n_inputs == 1
        assert ss.n_outputs == 1
    
    def test_dimensions(self):
        """Teste de propriedades de dimensão"""
        # Sistema 2x1x1
        A = sp.Matrix([[-1, 1], [0, -2]])
        B = sp.Matrix([[0], [1]])
        C = sp.Matrix([[1, 0]])
        D = sp.Matrix([[0]])
        
        ss = SymbolicStateSpace(A, B, C, D)
        
        assert ss.n_states == 2
        assert ss.n_inputs == 1
        assert ss.n_outputs == 1
    
    def test_dimension_validation(self):
        """Teste de validação de dimensões"""
        # Dimensões incompatíveis devem gerar erro
        A = sp.Matrix([[-1]])
        B = sp.Matrix([[1], [1]])  # B tem mais linhas que A
        C = sp.Matrix([[1]])
        D = sp.Matrix([[0]])
        
        with pytest.raises(ValueError):
            SymbolicStateSpace(A, B, C, D)
    
    def test_substitution(self):
        """Teste de substituição de símbolos"""
        A = sp.Matrix([[self.a]])
        B = sp.Matrix([[self.b]])
        C = sp.Matrix([[self.c]])
        D = sp.Matrix([[0]])
        
        ss = SymbolicStateSpace(A, B, C, D)
        
        # Substitui a=1, b=2, c=3
        ss_sub = ss.substitute({self.a: 1, self.b: 2, self.c: 3})
        
        assert ss_sub.A[0, 0] == 1
        assert ss_sub.B[0, 0] == 2
        assert ss_sub.C[0, 0] == 3
    
    def test_simplification(self):
        """Teste de simplificação"""
        # Cria sistema com expressões que podem ser simplificadas
        A = sp.Matrix([[self.a + self.a]])  # 2*a
        B = sp.Matrix([[self.b * 2 / 2]])   # b
        C = sp.Matrix([[self.c]])
        D = sp.Matrix([[0]])
        
        ss = SymbolicStateSpace(A, B, C, D)
        ss_simplified = ss.simplify()
        
        assert ss_simplified.A[0, 0] == 2 * self.a
        assert ss_simplified.B[0, 0] == self.b
    
    def test_eigenvalues(self):
        """Teste de cálculo de autovalores"""
        # Sistema com autovalores conhecidos
        A = sp.Matrix([[-1, 0], [0, -2]])
        B = sp.Matrix([[1], [1]])
        C = sp.Matrix([[1, 1]])
        D = sp.Matrix([[0]])
        
        ss = SymbolicStateSpace(A, B, C, D)
        eigenvals = ss.eigenvalues()
        
        assert -1 in eigenvals
        assert -2 in eigenvals
    
    def test_characteristic_polynomial(self):
        """Teste de polinômio característico"""
        # Sistema 1x1 simples
        A = sp.Matrix([[-2]])
        B = sp.Matrix([[1]])
        C = sp.Matrix([[1]])
        D = sp.Matrix([[0]])
        
        ss = SymbolicStateSpace(A, B, C, D)
        char_poly = ss.characteristic_polynomial(self.s)
        
        # Para A = [-2], o polinômio característico deve ser s + 2
        expected = self.s + 2
        assert sp.simplify(char_poly - expected) == 0
    
    def test_transfer_function_conversion(self):
        """Teste de conversão para função de transferência"""
        # Sistema em espaço de estados conhecido
        A = sp.Matrix([[-1]])
        B = sp.Matrix([[1]])
        C = sp.Matrix([[1]])
        D = sp.Matrix([[0]])
        
        ss = SymbolicStateSpace(A, B, C, D)
        G = ss.transfer_function(self.s)
        
        # Para este sistema, G(s) = 1/(s+1)
        expected = 1 / (self.s + 1)
        
        if G is not None:
            assert sp.simplify(G[0, 0] - expected) == 0
    
    def test_controllability_simple(self):
        """Teste básico de controlabilidade"""
        # Sistema controlável simples
        A = sp.Matrix([[-1]])
        B = sp.Matrix([[1]])
        C = sp.Matrix([[1]])
        D = sp.Matrix([[0]])
        
        ss = SymbolicStateSpace(A, B, C, D)
        
        # Sistema 1x1 com B≠0 deve ser controlável
        assert ss.is_controllable()
    
    def test_observability_simple(self):
        """Teste básico de observabilidade"""
        # Sistema observável simples
        A = sp.Matrix([[-1]])
        B = sp.Matrix([[1]])
        C = sp.Matrix([[1]])
        D = sp.Matrix([[0]])
        
        ss = SymbolicStateSpace(A, B, C, D)
        
        # Sistema 1x1 com C≠0 deve ser observável
        assert ss.is_observable()
    
    def test_controllability_2x2(self):
        """Teste de controlabilidade para sistema 2x2"""
        # Sistema controlável
        A = sp.Matrix([[0, 1], [-2, -3]])
        B = sp.Matrix([[0], [1]])
        C = sp.Matrix([[1, 0]])
        D = sp.Matrix([[0]])
        
        ss = SymbolicStateSpace(A, B, C, D)
        
        # Este sistema deve ser controlável
        # Matriz de controlabilidade: [B, AB] = [[0, 1], [1, -3]]
        # que tem posto 2
        assert ss.is_controllable()
    
    def test_observability_2x2(self):
        """Teste de observabilidade para sistema 2x2"""
        # Sistema observável
        A = sp.Matrix([[0, 1], [-2, -3]])
        B = sp.Matrix([[0], [1]])
        C = sp.Matrix([[1, 0]])
        D = sp.Matrix([[0]])
        
        ss = SymbolicStateSpace(A, B, C, D)
        
        # Este sistema deve ser observável
        # Matriz de observabilidade: [C; CA] = [[1, 0], [0, 1]]
        # que tem posto 2
        assert ss.is_observable()
    
    def test_string_representation(self):
        """Teste de representação em string"""
        A = sp.Matrix([[-1]])
        B = sp.Matrix([[1]])
        C = sp.Matrix([[1]])
        D = sp.Matrix([[0]])
        
        ss = SymbolicStateSpace(A, B, C, D)
        str_repr = str(ss)
        
        assert "StateSpace" in str_repr
        assert "1×1×1" in str_repr
    
    def test_latex_conversion(self):
        """Teste de conversão para LaTeX"""
        A = sp.Matrix([[-1]])
        B = sp.Matrix([[1]])
        C = sp.Matrix([[1]])
        D = sp.Matrix([[0]])
        
        ss = SymbolicStateSpace(A, B, C, D)
        latex_str = ss.to_latex()
        
        assert isinstance(latex_str, str)
        assert "begin{align}" in latex_str
        assert "dot{x}" in latex_str
    
    def test_history_tracking(self):
        """Teste de rastreamento de histórico"""
        A = sp.Matrix([[self.a]])
        B = sp.Matrix([[1]])
        C = sp.Matrix([[1]])
        D = sp.Matrix([[0]])
        
        ss = SymbolicStateSpace(A, B, C, D)
        
        # Operação que deve ser registrada
        ss_sub = ss.substitute({self.a: -1})
        
        # Verifica se o histórico foi registrado
        assert len(ss_sub.history.steps) > 0
        
        # Verifica se contém operação de substituição
        operations = [step.operation for step in ss_sub.history.steps]
        assert "SUBSTITUIÇÃO_SS" in operations
    
    def test_series_connection(self):
        """Teste de conexão em série"""
        # Sistema 1: saída escalar
        A1 = sp.Matrix([[-1]])
        B1 = sp.Matrix([[1]])
        C1 = sp.Matrix([[1]])
        D1 = sp.Matrix([[0]])
        ss1 = SymbolicStateSpace(A1, B1, C1, D1)
        
        # Sistema 2: entrada escalar
        A2 = sp.Matrix([[-2]])
        B2 = sp.Matrix([[1]])
        C2 = sp.Matrix([[1]])
        D2 = sp.Matrix([[0]])
        ss2 = SymbolicStateSpace(A2, B2, C2, D2)
        
        # Conecta em série
        ss_series = ss1.series(ss2)
        
        # Sistema resultante deve ter 2 estados
        assert ss_series.n_states == 2
        assert ss_series.n_inputs == 1
        assert ss_series.n_outputs == 1
