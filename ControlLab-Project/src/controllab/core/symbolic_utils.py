#!/usr/bin/env python3
"""
Utilitários Simbólicos - ControlLab
Funções auxiliares para manipulação simbólica
"""

import sympy as sp
from typing import List, Union, Tuple
from sympy import Symbol, Poly, simplify, cancel, gcd

def create_laplace_variable(name: str = 's') -> Symbol:
    """
    Cria uma variável de Laplace
    
    Args:
        name: Nome da variável (padrão 's')
    
    Returns:
        Symbol: Variável simbólica de Laplace
    """
    return Symbol(name, complex=True)

def create_z_variable(name: str = 'z') -> Symbol:
    """
    Cria uma variável de transformada Z
    
    Args:
        name: Nome da variável (padrão 'z')
    
    Returns:
        Symbol: Variável simbólica da transformada Z
    """
    return Symbol(name, complex=True)

def poly_from_roots(roots: List[Union[int, float, complex, Symbol]], variable: Symbol) -> sp.Expr:
    """
    Cria um polinômio a partir de suas raízes
    
    Args:
        roots: Lista de raízes
        variable: Variável do polinômio
    
    Returns:
        sp.Expr: Polinômio com as raízes especificadas
    """
    poly = 1
    for root in roots:
        poly *= (variable - root)
    
    return sp.expand(poly)  # Retorna expressão expandida em vez de Poly

def validate_proper_tf(numerator: Union[Poly, sp.Expr], denominator: Union[Poly, sp.Expr]) -> bool:
    """
    Valida se uma função de transferência é própria (grau do numerador <= grau do denominador)
    
    Args:
        numerator: Numerador da função de transferência
        denominator: Denominador da função de transferência
    
    Returns:
        bool: True se for própria, False caso contrário
    """
    try:
        if isinstance(numerator, sp.Expr):
            num_degree = sp.degree(numerator)
        else:
            num_degree = numerator.degree()
        
        if isinstance(denominator, sp.Expr):
            den_degree = sp.degree(denominator)
        else:
            den_degree = denominator.degree()
        
        return num_degree <= den_degree
    except:
        return True  # Se não conseguir determinar, assume que é válida

def cancel_common_factors(numerator: Union[Poly, sp.Expr], denominator: Union[Poly, sp.Expr]) -> Tuple[sp.Expr, sp.Expr]:
    """
    Cancela fatores comuns entre numerador e denominador
    
    Args:
        numerator: Numerador
        denominator: Denominador
    
    Returns:
        Tuple[sp.Expr, sp.Expr]: Numerador e denominador simplificados
    """
    try:
        # Converte para expressões SymPy se necessário
        if isinstance(numerator, Poly):
            num_expr = numerator.as_expr()
        else:
            num_expr = numerator
        
        if isinstance(denominator, Poly):
            den_expr = denominator.as_expr()
        else:
            den_expr = denominator
        
        # Usa cancel para simplificar a fração
        simplified = cancel(num_expr / den_expr)
        
        # Extrai numerador e denominador simplificados
        num_simplified = sp.numer(simplified)
        den_simplified = sp.denom(simplified)
        
        return num_simplified, den_simplified
    
    except Exception as e:
        # Em caso de erro, retorna os originais
        return numerator, denominator

def extract_poles_zeros(numerator: Union[Poly, sp.Expr], denominator: Union[Poly, sp.Expr]) -> Tuple[List, List]:
    """
    Extrai polos e zeros de uma função de transferência
    
    Args:
        numerator: Numerador da função de transferência
        denominator: Denominador da função de transferência
    
    Returns:
        Tuple[List, List]: Lista de zeros e lista de polos
    """
    try:
        # Converte para expressões se necessário
        if isinstance(numerator, Poly):
            num_expr = numerator.as_expr()
        else:
            num_expr = numerator
        
        if isinstance(denominator, Poly):
            den_expr = denominator.as_expr()
        else:
            den_expr = denominator
        
        # Identifica automaticamente a variável principal
        variables = num_expr.free_symbols.union(den_expr.free_symbols)
        if variables:
            # Usa a primeira variável encontrada (geralmente 's' ou 'z')
            main_var = list(variables)[0]
        else:
            return [], []
        
        # Calcula zeros (raízes do numerador)
        zeros = sp.solve(num_expr, main_var)
        
        # Calcula polos (raízes do denominador)
        poles = sp.solve(den_expr, main_var)
        
        return zeros, poles
    
    except Exception as e:
        return [], []

def create_proper_tf(zeros: List, poles: List, gain: float = 1.0, variable: Symbol = None) -> Tuple[sp.Expr, sp.Expr]:
    """
    Cria uma função de transferência própria a partir de polos, zeros e ganho
    
    Args:
        zeros: Lista de zeros
        poles: Lista de polos
        gain: Ganho da função de transferência
        variable: Variável da função (padrão 's')
    
    Returns:
        Tuple[sp.Expr, sp.Expr]: Numerador e denominador
    """
    if variable is None:
        variable = create_laplace_variable()
    
    # Constrói numerador a partir dos zeros
    numerator = gain
    for zero in zeros:
        numerator *= (variable - zero)
    
    # Constrói denominador a partir dos polos
    denominator = 1
    for pole in poles:
        denominator *= (variable - pole)
    
    return numerator, denominator

def expand_partial_fractions(numerator: sp.Expr, denominator: sp.Expr, variable: Symbol = None) -> sp.Expr:
    """
    Expande em frações parciais
    
    Args:
        numerator: Numerador
        denominator: Denominador
        variable: Variável (padrão 's')
    
    Returns:
        sp.Expr: Expansão em frações parciais
    """
    if variable is None:
        variable = create_laplace_variable()
    
    try:
        fraction = numerator / denominator
        return sp.apart(fraction, variable)
    except Exception as e:
        return numerator / denominator

def symbolic_stability_analysis(denominator: sp.Expr, variable: Symbol = None) -> dict:
    """
    Análise de estabilidade simbólica usando critério de Routh-Hurwitz
    
    Args:
        denominator: Polinômio característico
        variable: Variável do polinômio
    
    Returns:
        dict: Resultado da análise de estabilidade
    """
    if variable is None:
        variable = create_laplace_variable()
    
    try:
        # Extrai coeficientes do polinômio
        poly = Poly(denominator, variable)
        coeffs = poly.all_coeffs()
        
        # Verifica condições básicas de estabilidade
        stable = True
        reasons = []
        
        # Todos os coeficientes devem ter o mesmo sinal
        if len(coeffs) > 1:
            signs = [1 if c > 0 else -1 if c < 0 else 0 for c in coeffs if c != 0]
            if len(set(signs)) > 1:
                stable = False
                reasons.append("Coeficientes com sinais diferentes")
        
        return {
            'stable': stable,
            'coefficients': coeffs,
            'reasons': reasons,
            'polynomial': denominator
        }
    
    except Exception as e:
        return {
            'stable': None,
            'error': str(e),
            'polynomial': denominator
        }

def convert_to_latex_formatted(expression: sp.Expr) -> str:
    """
    Converte expressão para LaTeX formatado
    
    Args:
        expression: Expressão SymPy
    
    Returns:
        str: Código LaTeX formatado
    """
    try:
        return sp.latex(expression)
    except:
        return str(expression)
