#!/usr/bin/env python3
"""
Função de Transferência Simbólica - ControlLab
Implementação completa da classe SymbolicTransferFunction
"""

import sympy as sp
from typing import Union, Dict, Any
from sympy import Symbol, Poly, simplify, cancel, latex, expand, factor
from .history import OperationHistory
from .symbolic_utils import (
    create_laplace_variable, 
    cancel_common_factors, 
    validate_proper_tf,
    extract_poles_zeros,
    expand_partial_fractions
)

class SymbolicTransferFunction:
    def get_poles(self) -> list:
        """
        Alias para poles(), para compatibilidade retroativa.
        """
        return self.poles()
    """
    Classe para representação e manipulação de funções de transferência simbólicas
    
    Esta classe permite criar, manipular e analisar funções de transferência usando
    computação simbólica, mantendo um histórico completo das operações para fins pedagógicos.
    """
    
    def __init__(self, numerator: Union[sp.Expr, int, float], 
                 denominator: Union[sp.Expr, int, float], 
                 variable: Union[Symbol, str] = 's'):
        """
        Inicializa uma função de transferência simbólica
        
        Args:
            numerator: Numerador da função de transferência
            denominator: Denominador da função de transferência  
            variable: Variável da função de transferência (padrão 's')
        """
        # Define a variável
        if isinstance(variable, str):
            self.variable = create_laplace_variable(variable)
        else:
            self.variable = variable
        
        # Converte numerador e denominador para expressões SymPy
        self.numerator = sp.sympify(numerator)
        self.denominator = sp.sympify(denominator)
        
        # Inicializa histórico
        self.history = OperationHistory()
        
        # Registra criação no histórico
        self.history.add_step(
            "CRIAÇÃO", 
            f"Função de transferência criada: G({self.variable}) = ({self.numerator})/({self.denominator})",
            None,
            self,
            {"proper": validate_proper_tf(self.numerator, self.denominator)}
        )
    
    def __str__(self):
        """Representação em string"""
        return f"G({self.variable}) = ({self.numerator}) / ({self.denominator})"
    
    def __repr__(self):
        """Representação para debug"""
        return f"SymbolicTransferFunction({self.numerator}, {self.denominator}, {self.variable})"
    
    def __mul__(self, other: 'SymbolicTransferFunction') -> 'SymbolicTransferFunction':
        """
        Multiplicação de funções de transferência
        
        Args:
            other: Outra função de transferência
            
        Returns:
            SymbolicTransferFunction: Produto das funções de transferência
        """
        if not isinstance(other, SymbolicTransferFunction):
            # Permite multiplicação por escalar
            if isinstance(other, (int, float, sp.Basic)):
                new_num = self.numerator * other
                new_den = self.denominator
                
                result = SymbolicTransferFunction(new_num, new_den, self.variable)
                result.history.steps = self.history.steps.copy()
                result.history.add_step(
                    "MULTIPLICAÇÃO_ESCALAR",
                    f"Multiplicação por escalar: G(s) * {other}",
                    str(self),
                    str(result)
                )
                return result
            else:
                raise TypeError(f"Não é possível multiplicar SymbolicTransferFunction por {type(other)}")
        
        # Multiplicação de funções de transferência: G1*G2 = (num1*num2)/(den1*den2)
        new_numerator = self.numerator * other.numerator
        new_denominator = self.denominator * other.denominator
        
        # Simplifica o resultado
        new_numerator, new_denominator = cancel_common_factors(new_numerator, new_denominator)
        
        result = SymbolicTransferFunction(new_numerator, new_denominator, self.variable)
        
        # Combina históricos
        result.history.steps = self.history.steps.copy()
        result.history.steps.extend(other.history.steps)
        result.history.add_step(
            "MULTIPLICAÇÃO",
            f"Multiplicação de funções de transferência",
            f"G1(s) = {self}, G2(s) = {other}",
            str(result)
        )
        
        return result
    
    def __add__(self, other) -> 'SymbolicTransferFunction':
        """
        Adição de funções de transferência (conexão paralela)
        
        Args:
            other: Outra função de transferência ou número
            
        Returns:
            SymbolicTransferFunction: Soma das funções de transferência
        """
        if isinstance(other, (int, float)):
            # Converte número para função de transferência
            other = SymbolicTransferFunction(other, 1, self.variable)
        
        if not isinstance(other, SymbolicTransferFunction):
            raise TypeError(f"Não é possível somar SymbolicTransferFunction com {type(other)}")
        
        # Soma de funções de transferência: G1+G2 = (num1*den2 + num2*den1)/(den1*den2)
        new_numerator = self.numerator * other.denominator + other.numerator * self.denominator
        new_denominator = self.denominator * other.denominator
        
        # Simplifica o resultado
        new_numerator, new_denominator = cancel_common_factors(new_numerator, new_denominator)
        
        result = SymbolicTransferFunction(new_numerator, new_denominator, self.variable)
        
        # Combina históricos
        result.history.steps = self.history.steps.copy()
        result.history.steps.extend(other.history.steps)
        result.history.add_step(
            "ADIÇÃO",
            f"Adição de funções de transferência (conexão paralela)",
            f"G1(s) = {self}, G2(s) = {other}",
            str(result)
        )
        
        return result
    
    def __radd__(self, other) -> 'SymbolicTransferFunction':
        """
        Adição reversa (número + função de transferência)
        
        Args:
            other: Número para somar
            
        Returns:
            SymbolicTransferFunction: Soma
        """
        return self.__add__(other)
    
    def __truediv__(self, other: 'SymbolicTransferFunction') -> 'SymbolicTransferFunction':
        """
        Divisão de funções de transferência
        
        Args:
            other: Outra função de transferência
            
        Returns:
            SymbolicTransferFunction: Divisão das funções de transferência
        """
        if not isinstance(other, SymbolicTransferFunction):
            raise TypeError(f"Não é possível dividir SymbolicTransferFunction por {type(other)}")
        
        # Divisão: G1/G2 = (num1*den2)/(den1*num2)
        new_numerator = self.numerator * other.denominator
        new_denominator = self.denominator * other.numerator
        
        result = SymbolicTransferFunction(new_numerator, new_denominator, self.variable)
        
        # Combina históricos
        result.history.steps = self.history.steps.copy()
        result.history.steps.extend(other.history.steps)
        result.history.add_step(
            "DIVISÃO",
            f"Divisão de funções de transferência",
            f"G1(s) = {self}, G2(s) = {other}",
            str(result)
        )
        
        return result
    
    def simplify(self) -> 'SymbolicTransferFunction':
        """
        Simplifica a função de transferência
        
        Returns:
            SymbolicTransferFunction: Função de transferência simplificada
        """
        original_str = str(self)
        
        # Simplifica numerador e denominador
        simplified_num, simplified_den = cancel_common_factors(self.numerator, self.denominator)
        
        result = SymbolicTransferFunction(simplified_num, simplified_den, self.variable)
        result.history.steps = self.history.steps.copy()
        result.history.add_step(
            "SIMPLIFICAÇÃO",
            "Cancelamento de fatores comuns e simplificação algébrica",
            original_str,
            str(result)
        )
        
        return result
    
    def substitute(self, substitutions: Dict[Symbol, Union[int, float, Symbol]]) -> 'SymbolicTransferFunction':
        """
        Substitui símbolos na função de transferência
        
        Args:
            substitutions: Dicionário com substituições {símbolo: valor}
            
        Returns:
            SymbolicTransferFunction: Função com substituições aplicadas
        """
        original_str = str(self)
        
        # Aplica substituições
        new_numerator = self.numerator.subs(substitutions)
        new_denominator = self.denominator.subs(substitutions)
        
        result = SymbolicTransferFunction(new_numerator, new_denominator, self.variable)
        result.history.steps = self.history.steps.copy()
        result.history.add_step(
            "SUBSTITUIÇÃO",
            f"Substituição de parâmetros: {substitutions}",
            original_str,
            str(result)
        )
        
        return result
    
    def expand(self) -> 'SymbolicTransferFunction':
        """
        Expande numerador e denominador
        
        Returns:
            SymbolicTransferFunction: Função expandida
        """
        original_str = str(self)
        
        expanded_num = expand(self.numerator)
        expanded_den = expand(self.denominator)
        
        result = SymbolicTransferFunction(expanded_num, expanded_den, self.variable)
        result.history.steps = self.history.steps.copy()
        result.history.add_step(
            "EXPANSÃO",
            "Expansão algébrica de numerador e denominador",
            original_str,
            str(result)
        )
        
        return result
    
    def factor(self) -> 'SymbolicTransferFunction':
        """
        Fatora numerador e denominador
        
        Returns:
            SymbolicTransferFunction: Função fatorada
        """
        original_str = str(self)
        
        factored_num = factor(self.numerator)
        factored_den = factor(self.denominator)
        
        result = SymbolicTransferFunction(factored_num, factored_den, self.variable)
        result.history.steps = self.history.steps.copy()
        result.history.add_step(
            "FATORAÇÃO",
            "Fatoração de numerador e denominador",
            original_str,
            str(result)
        )
        
        return result
    
    def partial_fractions(self) -> sp.Expr:
        """
        Expande em frações parciais
        
        Returns:
            sp.Expr: Expansão em frações parciais
        """
        expansion = expand_partial_fractions(self.numerator, self.denominator, self.variable)
        
        self.history.add_step(
            "FRAÇÕES_PARCIAIS",
            "Expansão em frações parciais",
            str(self),
            str(expansion)
        )
        
        return expansion
    
    def poles(self) -> list:
        """
        Calcula os polos da função de transferência
        
        Returns:
            list: Lista de polos
        """
        zeros, poles = extract_poles_zeros(self.numerator, self.denominator)
        return poles
    
    def zeros(self) -> list:
        """
        Calcula os zeros da função de transferência
        
        Returns:
            list: Lista de zeros
        """
        zeros, poles = extract_poles_zeros(self.numerator, self.denominator)
        return zeros
    
    def to_latex(self) -> str:
        """
        Converte para representação LaTeX
        
        Returns:
            str: Código LaTeX da função de transferência
        """
        try:
            fraction = self.numerator / self.denominator
            return latex(fraction)
        except:
            return f"\\frac{{{latex(self.numerator)}}}{{{latex(self.denominator)}}}"
    
    def evaluate_at(self, value: Union[int, float, complex]) -> complex:
        """
        Avalia a função de transferência em um ponto específico
        
        Args:
            value: Valor para avaliar a função
            
        Returns:
            complex: Valor da função no ponto especificado
        """
        substitutions = {self.variable: value}
        num_val = complex(self.numerator.subs(substitutions))
        den_val = complex(self.denominator.subs(substitutions))
        
        if den_val == 0:
            raise ValueError(f"Denominador é zero em {self.variable} = {value}")
        
        return num_val / den_val
    
    @property
    def is_proper(self) -> bool:
        """Verifica se a função de transferência é própria"""
        return validate_proper_tf(self.numerator, self.denominator)
    
    @property
    def degree(self) -> tuple:
        """Retorna os graus do numerador e denominador"""
        num_degree = sp.degree(self.numerator, self.variable)
        den_degree = sp.degree(self.denominator, self.variable)
        return (num_degree, den_degree)

    def get_coefficients(self) -> dict:
        """
        Extrai coeficientes do numerador e denominador
        
        Returns:
            dict: Dicionário com coeficientes {'num': [...], 'den': [...]}
        """
        try:
            # Converte para polinômios para extrair coeficientes
            num_poly = sp.Poly(self.numerator, self.variable)
            den_poly = sp.Poly(self.denominator, self.variable)
            
            return {
                'num': num_poly.all_coeffs(),
                'den': den_poly.all_coeffs()
            }
        except:
            return {'num': [self.numerator], 'den': [self.denominator]}

    def apply_laplace_rules(self, time_expr: sp.Expr, initial_conditions: dict = None) -> 'SymbolicTransferFunction':
        """
        Aplica regras da transformada de Laplace
        
        Args:
            time_expr: Expressão no domínio do tempo
            initial_conditions: Condições iniciais
            
        Returns:
            SymbolicTransferFunction: Resultado da transformada
        """
        if initial_conditions is None:
            initial_conditions = {}
        
        # Aplica transformada de Laplace
        try:
            laplace_result = sp.laplace_transform(time_expr, sp.Symbol('t'), self.variable)
            
            # Cria nova função de transferência
            result = SymbolicTransferFunction(
                laplace_result[0].as_numer_denom()[0],
                laplace_result[0].as_numer_denom()[1],
                self.variable
            )
            
            result.history.steps = self.history.steps.copy()
            result.history.add_step(
                "TRANSFORMADA_LAPLACE",
                f"Aplicada transformada de Laplace: L{{{time_expr}}}",
                str(time_expr),
                str(result)
            )
            
            return result
        except Exception as e:
            self.history.add_step(
                "ERRO_LAPLACE",
                f"Erro na transformada de Laplace: {str(e)}",
                str(time_expr),
                None
            )
            return self

    def apply_tustin_transform(self, T: Union[float, sp.Symbol]) -> 'SymbolicTransferFunction':
        """
        Aplica transformação de Tustin (s -> (2/T)*(z-1)/(z+1))
        
        Args:
            T: Período de amostragem
            
        Returns:
            SymbolicTransferFunction: Sistema discretizado
        """
        original_str = str(self)
        
        # Define variável z
        z = sp.Symbol('z', complex=True)
        
        # Transformação de Tustin
        tustin_sub = (2/T) * (z - 1) / (z + 1)
        
        # Aplica substituição
        new_num = self.numerator.subs(self.variable, tustin_sub)
        new_den = self.denominator.subs(self.variable, tustin_sub)
        
        # Simplifica para obter forma polinomial
        combined = (new_num / new_den).simplify()
        final_num = sp.numer(combined)
        final_den = sp.denom(combined)
        
        result = SymbolicTransferFunction(final_num, final_den, z)
        result.history.steps = self.history.steps.copy()
        result.history.add_step(
            "TRANSFORMAÇÃO_TUSTIN",
            f"Aplicada transformação de Tustin com T={T}",
            original_str,
            str(result)
        )
        
        return result

    def get_margin_analysis(self) -> dict:
        """
        Análise de margens de ganho e fase
        
        Returns:
            dict: Margens calculadas simbolicamente quando possível
        """
        try:
            # Substitui s por jω para análise de frequência
            omega = sp.Symbol('omega', real=True)
            jw_expr = self.numerator / self.denominator
            jw_expr = jw_expr.subs(self.variable, sp.I * omega)
            
            # Magnitude e fase
            magnitude = sp.Abs(jw_expr)
            phase = sp.arg(jw_expr)
            
            self.history.add_step(
                "ANÁLISE_MARGENS",
                "Cálculo de margens de ganho e fase",
                str(self),
                f"Magnitude: {magnitude}, Fase: {phase}"
            )
            
            return {
                'magnitude_expr': magnitude,
                'phase_expr': phase,
                'frequency_var': omega
            }
        except Exception as e:
            return {'error': str(e)}

    def characteristic_equation(self) -> sp.Expr:
        """
        Retorna equação característica (denominador = 0)
        
        Returns:
            sp.Expr: Equação característica
        """
        return self.denominator
