#!/usr/bin/env python3
"""
Sistema em Espaço de Estados Simbólico - ControlLab
Implementação da classe SymbolicStateSpace
"""

import sympy as sp
from typing import Union, Dict, Any, Tuple
from sympy import Matrix, Symbol, simplify, latex
from .history import OperationHistory
from .symbolic_utils import create_laplace_variable

class SymbolicStateSpace:
    """
    Classe para representação e manipulação de sistemas em espaço de estados simbólicos
    
    Representa um sistema linear da forma:
    ẋ = Ax + Bu
    y = Cx + Du
    
    onde A, B, C, D são matrizes simbólicas.
    """
    
    def __init__(self, A: Union[Matrix, list], B: Union[Matrix, list], 
                 C: Union[Matrix, list], D: Union[Matrix, list]):
        """
        Inicializa um sistema em espaço de estados simbólico
        
        Args:
            A: Matriz de estados (n×n)
            B: Matriz de entrada (n×m)
            C: Matriz de saída (p×n)
            D: Matriz de transmissão direta (p×m)
        """
        # Converte para matrizes SymPy
        self.A = Matrix(A) if not isinstance(A, Matrix) else A
        self.B = Matrix(B) if not isinstance(B, Matrix) else B
        self.C = Matrix(C) if not isinstance(C, Matrix) else C
        self.D = Matrix(D) if not isinstance(D, Matrix) else D
        
        # Valida dimensões
        self._validate_dimensions()
        
        # Inicializa histórico
        self.history = OperationHistory()
        
        # Registra criação no histórico
        self.history.add_step(
            "CRIAÇÃO_SS",
            f"Sistema em espaço de estados criado: {self.n_states} estados, {self.n_inputs} entradas, {self.n_outputs} saídas",
            None,
            self,
            {
                "dimensions": f"{self.n_states}×{self.n_inputs}×{self.n_outputs}",
                "matrices": {"A": self.A.shape, "B": self.B.shape, "C": self.C.shape, "D": self.D.shape}
            }
        )
    
    def _validate_dimensions(self):
        """Valida as dimensões das matrizes"""
        n, m, p = self.A.rows, self.B.cols, self.C.rows
        
        if self.A.cols != n:
            raise ValueError(f"Matriz A deve ser quadrada, mas tem dimensões {self.A.shape}")
        
        if self.B.rows != n:
            raise ValueError(f"Matriz B deve ter {n} linhas, mas tem {self.B.rows}")
        
        if self.C.cols != n:
            raise ValueError(f"Matriz C deve ter {n} colunas, mas tem {self.C.cols}")
        
        if self.D.shape != (p, m):
            raise ValueError(f"Matriz D deve ter dimensões ({p}, {m}), mas tem {self.D.shape}")
    
    @property
    def n_states(self) -> int:
        """Número de estados"""
        return self.A.rows
    
    @property
    def n_inputs(self) -> int:
        """Número de entradas"""
        return self.B.cols
    
    @property
    def n_outputs(self) -> int:
        """Número de saídas"""
        return self.C.rows
    
    def __str__(self):
        """Representação em string"""
        return f"StateSpace({self.n_states}×{self.n_inputs}×{self.n_outputs})"
    
    def __repr__(self):
        """Representação para debug"""
        return (f"SymbolicStateSpace(\n"
                f"  A={self.A},\n"
                f"  B={self.B},\n"
                f"  C={self.C},\n"
                f"  D={self.D}\n"
                f")")
    
    def substitute(self, substitutions: Dict[Symbol, Union[int, float, Symbol]]) -> 'SymbolicStateSpace':
        """
        Substitui símbolos no sistema em espaço de estados
        
        Args:
            substitutions: Dicionário com substituições {símbolo: valor}
            
        Returns:
            SymbolicStateSpace: Sistema com substituições aplicadas
        """
        original_str = str(self)
        
        # Aplica substituições em todas as matrizes
        new_A = self.A.subs(substitutions)
        new_B = self.B.subs(substitutions)
        new_C = self.C.subs(substitutions)
        new_D = self.D.subs(substitutions)
        
        result = SymbolicStateSpace(new_A, new_B, new_C, new_D)
        result.history.steps = self.history.steps.copy()
        result.history.add_step(
            "SUBSTITUIÇÃO_SS",
            f"Substituição de parâmetros: {substitutions}",
            original_str,
            str(result)
        )
        
        return result
    
    def simplify(self) -> 'SymbolicStateSpace':
        """
        Simplifica todas as matrizes do sistema
        
        Returns:
            SymbolicStateSpace: Sistema simplificado
        """
        original_str = str(self)
        
        # Simplifica cada matriz
        simplified_A = simplify(self.A)
        simplified_B = simplify(self.B)
        simplified_C = simplify(self.C)
        simplified_D = simplify(self.D)
        
        result = SymbolicStateSpace(simplified_A, simplified_B, simplified_C, simplified_D)
        result.history.steps = self.history.steps.copy()
        result.history.add_step(
            "SIMPLIFICAÇÃO_SS",
            "Simplificação algébrica das matrizes do sistema",
            original_str,
            str(result)
        )
        
        return result
    
    def eigenvalues(self) -> list:
        """
        Calcula os autovalores da matriz A (polos do sistema)
        
        Returns:
            list: Lista de autovalores
        """
        try:
            eigenvals = self.A.eigenvals()
            return list(eigenvals.keys())
        except Exception as e:
            self.history.add_step(
                "ERRO_AUTOVALORES",
                f"Erro ao calcular autovalores: {str(e)}",
                str(self.A),
                None
            )
            return []
    
    def characteristic_polynomial(self, variable: Symbol = None) -> sp.Expr:
        """
        Calcula o polinômio característico det(sI - A)
        
        Args:
            variable: Variável do polinômio (padrão 's')
            
        Returns:
            sp.Expr: Polinômio característico
        """
        if variable is None:
            variable = create_laplace_variable()
        
        I = sp.eye(self.n_states)
        char_matrix = variable * I - self.A
        char_poly = char_matrix.det()
        
        self.history.add_step(
            "POLINÔMIO_CARACTERÍSTICO",
            f"Cálculo do polinômio característico det({variable}I - A)",
            str(self.A),
            str(char_poly)
        )
        
        return char_poly
    
    def transfer_function(self, variable: Symbol = None) -> Matrix:
        """
        Calcula a função de transferência G(s) = C(sI - A)^(-1)B + D
        
        Args:
            variable: Variável da função de transferência (padrão 's')
            
        Returns:
            Matrix: Matriz de funções de transferência
        """
        if variable is None:
            variable = create_laplace_variable()
        
        try:
            I = sp.eye(self.n_states)
            sI_minus_A = variable * I - self.A
            
            # Calcula (sI - A)^(-1)
            inv_matrix = sI_minus_A.inv()
            
            # G(s) = C(sI - A)^(-1)B + D
            G = self.C * inv_matrix * self.B + self.D
            
            self.history.add_step(
                "FUNÇÃO_TRANSFERÊNCIA",
                f"Conversão para função de transferência G({variable}) = C({variable}I - A)^(-1)B + D",
                str(self),
                str(G)
            )
            
            return G
        
        except Exception as e:
            self.history.add_step(
                "ERRO_CONVERSÃO_TF",
                f"Erro na conversão para função de transferência: {str(e)}",
                str(self),
                None
            )
            return None
    
    def is_controllable(self) -> bool:
        """
        Verifica controlabilidade usando a matriz de controlabilidade
        
        Returns:
            bool: True se o sistema for controlável
        """
        try:
            # Matriz de controlabilidade: [B, AB, A²B, ..., A^(n-1)B]
            controllability_matrix = self.B
            A_power = sp.eye(self.n_states)
            
            for i in range(1, self.n_states):
                A_power = A_power * self.A
                controllability_matrix = controllability_matrix.row_join(A_power * self.B)
            
            # Sistema é controlável se a matriz tem posto completo
            rank = controllability_matrix.rank()
            is_controllable = rank == self.n_states
            
            self.history.add_step(
                "TESTE_CONTROLABILIDADE",
                f"Teste de controlabilidade: posto = {rank}, estados = {self.n_states}",
                "Matriz de controlabilidade",
                f"Controlável: {is_controllable}"
            )
            
            return is_controllable
        
        except Exception as e:
            self.history.add_step(
                "ERRO_CONTROLABILIDADE",
                f"Erro no teste de controlabilidade: {str(e)}",
                str(self),
                None
            )
            return False
    
    def is_observable(self) -> bool:
        """
        Verifica observabilidade usando a matriz de observabilidade
        
        Returns:
            bool: True se o sistema for observável
        """
        try:
            # Matriz de observabilidade: [C; CA; CA²; ...; CA^(n-1)]
            observability_matrix = self.C
            A_power = sp.eye(self.n_states)
            
            for i in range(1, self.n_states):
                A_power = A_power * self.A
                observability_matrix = observability_matrix.col_join(self.C * A_power)
            
            # Sistema é observável se a matriz tem posto completo
            rank = observability_matrix.rank()
            is_observable = rank == self.n_states
            
            self.history.add_step(
                "TESTE_OBSERVABILIDADE",
                f"Teste de observabilidade: posto = {rank}, estados = {self.n_states}",
                "Matriz de observabilidade",
                f"Observável: {is_observable}"
            )
            
            return is_observable
        
        except Exception as e:
            self.history.add_step(
                "ERRO_OBSERVABILIDADE",
                f"Erro no teste de observabilidade: {str(e)}",
                str(self),
                None
            )
            return False
    
    def to_latex(self) -> str:
        """
        Converte para representação LaTeX
        
        Returns:
            str: Código LaTeX do sistema em espaço de estados
        """
        try:
            latex_str = "\\begin{align}\n"
            latex_str += "\\dot{x} &= " + latex(self.A) + "x + " + latex(self.B) + "u \\\\\n"
            latex_str += "y &= " + latex(self.C) + "x + " + latex(self.D) + "u\n"
            latex_str += "\\end{align}"
            return latex_str
        except:
            return f"StateSpace({self.n_states}×{self.n_inputs}×{self.n_outputs})"
    
    def series(self, other: 'SymbolicStateSpace') -> 'SymbolicStateSpace':
        """
        Conexão em série com outro sistema
        
        Args:
            other: Outro sistema em espaço de estados
            
        Returns:
            SymbolicStateSpace: Sistema resultante da conexão em série
        """
        if self.n_outputs != other.n_inputs:
            raise ValueError(f"Incompatibilidade de dimensões: {self.n_outputs} saídas vs {other.n_inputs} entradas")
        
        # Sistema em série
        n1, n2 = self.n_states, other.n_states
        
        # Matrizes do sistema conectado
        A_series = sp.BlockMatrix([
            [self.A, sp.zeros(n1, n2)],
            [other.B * self.C, other.A]
        ])
        
        B_series = sp.Matrix([
            self.B,
            other.B * self.D
        ])
        
        C_series = sp.Matrix([[other.D * self.C, other.C]])
        D_series = other.D * self.D
        
        result = SymbolicStateSpace(A_series, B_series, C_series, D_series)
        
        # Combina históricos
        result.history.steps = self.history.steps.copy()
        result.history.steps.extend(other.history.steps)
        result.history.add_step(
            "CONEXÃO_SÉRIE",
            f"Conexão em série de sistemas",
            f"Sistema1: {self}, Sistema2: {other}",
            str(result)
        )
        
        return result
