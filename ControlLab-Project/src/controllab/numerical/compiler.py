#!/usr/bin/env python3
"""
Compilador de Expressões - ControlLab Numerical
Compila expressões simbólicas para funções numéricas eficientes
"""

import sympy as sp
import warnings
from typing import List, Dict, Any, Union, Callable, Optional
from functools import lru_cache

# Importações condicionais
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

from ..core.history import OperationHistory


class CompiledFunction:
    """
    Encapsula uma função simbólica compilada para avaliação numérica eficiente
    """
    
    def __init__(self, expression: sp.Expr, variables: List[sp.Symbol], 
                 backend: str = 'numpy', compiled_func: Callable = None):
        self.expression = expression
        self.variables = variables
        self.backend = backend
        self.compiled_func = compiled_func
        self.call_count = 0
    
    def __call__(self, *args, **kwargs):
        """Executa a função compilada"""
        self.call_count += 1
        
        if self.compiled_func is None:
            raise RuntimeError("Função não foi compilada corretamente")
        
        try:
            return self.compiled_func(*args, **kwargs)
        except Exception as e:
            raise RuntimeError(f"Erro na execução da função compilada: {e}")
    
    def __repr__(self):
        return f"CompiledFunction({self.expression}, vars={[str(v) for v in self.variables]}, backend={self.backend})"


class ExpressionCompiler:
    """
    Compila expressões simbólicas para funções numéricas eficientes
    """
    
    def __init__(self):
        self.history = OperationHistory()
        self._cache = {}
    
    def compile_expression(self, expression: sp.Expr, 
                          variables: List[sp.Symbol],
                          backend: str = 'numpy') -> CompiledFunction:
        """
        Compila uma expressão simbólica para função numérica
        
        Args:
            expression: Expressão SymPy
            variables: Lista de variáveis em ordem
            backend: Backend para compilação ('numpy', 'sympy', 'python')
            
        Returns:
            CompiledFunction: Função compilada
        """
        # Verificar cache
        cache_key = (str(expression), tuple(str(v) for v in variables), backend)
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        self.history.add_step(
            "COMPILAÇÃO_EXPRESSÃO",
            f"Compilando expressão com backend {backend}",
            str(expression),
            f"Variáveis: {[str(v) for v in variables]}"
        )
        
        try:
            if backend == 'numpy' and NUMPY_AVAILABLE:
                compiled_func = self._compile_numpy(expression, variables)
            elif backend == 'sympy':
                compiled_func = self._compile_sympy(expression, variables)
            elif backend == 'python':
                compiled_func = self._compile_python(expression, variables)
            else:
                # Fallback para sympy
                compiled_func = self._compile_sympy(expression, variables)
                backend = 'sympy'
                warnings.warn(f"Backend {backend} não disponível, usando sympy")
            
            result = CompiledFunction(expression, variables, backend, compiled_func)
            
            # Armazenar no cache
            self._cache[cache_key] = result
            
            self.history.add_step(
                "EXPRESSÃO_COMPILADA",
                f"Expressão compilada com sucesso usando {backend}",
                str(expression),
                f"Função: {result}"
            )
            
            return result
            
        except Exception as e:
            self.history.add_step(
                "ERRO_COMPILAÇÃO",
                f"Erro na compilação: {str(e)}",
                str(expression),
                None
            )
            raise
    
    def _compile_numpy(self, expression: sp.Expr, variables: List[sp.Symbol]) -> Callable:
        """Compila usando lambdify com backend NumPy"""
        if not NUMPY_AVAILABLE:
            raise ImportError("NumPy não disponível")
        
        # Usar lambdify do SymPy com backend numpy
        return sp.lambdify(variables, expression, 'numpy')
    
    def _compile_sympy(self, expression: sp.Expr, variables: List[sp.Symbol]) -> Callable:
        """Compila usando lambdify padrão do SymPy"""
        return sp.lambdify(variables, expression, 'math')
    
    def _compile_python(self, expression: sp.Expr, variables: List[sp.Symbol]) -> Callable:
        """Compila para Python puro"""
        return sp.lambdify(variables, expression, 'python')
    
    def compile_matrix(self, matrix: sp.Matrix, 
                      variables: List[sp.Symbol],
                      backend: str = 'numpy') -> CompiledFunction:
        """
        Compila uma matriz simbólica
        
        Args:
            matrix: Matriz SymPy
            variables: Lista de variáveis
            backend: Backend para compilação
            
        Returns:
            CompiledFunction: Função que retorna matriz numérica
        """
        self.history.add_step(
            "COMPILAÇÃO_MATRIZ",
            f"Compilando matriz {matrix.shape} com backend {backend}",
            f"Matriz: {matrix}",
            f"Variáveis: {[str(v) for v in variables]}"
        )
        
        try:
            if backend == 'numpy' and NUMPY_AVAILABLE:
                compiled_func = self._compile_matrix_numpy(matrix, variables)
            else:
                compiled_func = self._compile_matrix_sympy(matrix, variables)
                backend = 'sympy'
            
            result = CompiledFunction(matrix, variables, backend, compiled_func)
            
            self.history.add_step(
                "MATRIZ_COMPILADA",
                f"Matriz compilada com sucesso",
                f"Dimensões: {matrix.shape}",
                f"Função: {result}"
            )
            
            return result
            
        except Exception as e:
            self.history.add_step(
                "ERRO_COMPILAÇÃO_MATRIZ",
                f"Erro na compilação: {str(e)}",
                f"Matriz: {matrix}",
                None
            )
            raise
    
    def _compile_matrix_numpy(self, matrix: sp.Matrix, variables: List[sp.Symbol]) -> Callable:
        """Compila matriz usando NumPy"""
        if not NUMPY_AVAILABLE:
            raise ImportError("NumPy não disponível")
        
        # Compila cada elemento da matriz
        compiled_elements = []
        for i in range(matrix.rows):
            row = []
            for j in range(matrix.cols):
                element_func = sp.lambdify(variables, matrix[i, j], 'numpy')
                row.append(element_func)
            compiled_elements.append(row)
        
        def matrix_function(*args):
            """Função que avalia a matriz para os valores dados"""
            result = np.zeros((matrix.rows, matrix.cols), dtype=complex)
            for i in range(matrix.rows):
                for j in range(matrix.cols):
                    result[i, j] = compiled_elements[i][j](*args)
            return result
        
        return matrix_function
    
    def _compile_matrix_sympy(self, matrix: sp.Matrix, variables: List[sp.Symbol]) -> Callable:
        """Compila matriz usando SymPy puro"""
        # Função simples que substitui valores
        def matrix_function(*args):
            """Função que avalia a matriz para os valores dados"""
            substitutions = dict(zip(variables, args))
            evaluated_matrix = matrix.subs(substitutions)
            
            # Converte para lista de listas para compatibilidade
            result = []
            for i in range(evaluated_matrix.rows):
                row = []
                for j in range(evaluated_matrix.cols):
                    row.append(complex(evaluated_matrix[i, j]))
                result.append(row)
            
            return result
        
        return matrix_function
    
    def clear_cache(self):
        """Limpa o cache de funções compiladas"""
        self._cache.clear()
        self.history.add_step(
            "CACHE_LIMPO",
            "Cache de funções compiladas foi limpo",
            f"Eram {len(self._cache)} funções em cache",
            "Cache vazio"
        )
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do cache"""
        return {
            'cached_functions': len(self._cache),
            'total_calls': sum(func.call_count for func in self._cache.values()),
            'cache_keys': list(self._cache.keys())
        }


# Funções de conveniência para uso direto
def compile_expression(expression: sp.Expr, 
                      variables: List[sp.Symbol],
                      backend: str = 'numpy') -> CompiledFunction:
    """
    Função de conveniência para compilar expressões
    """
    compiler = ExpressionCompiler()
    return compiler.compile_expression(expression, variables, backend)


def compile_matrix(matrix: sp.Matrix, 
                  variables: List[sp.Symbol],
                  backend: str = 'numpy') -> CompiledFunction:
    """
    Função de conveniência para compilar matrizes
    """
    compiler = ExpressionCompiler()
    return compiler.compile_matrix(matrix, variables, backend)
