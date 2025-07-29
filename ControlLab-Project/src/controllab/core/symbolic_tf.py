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
        self.history.add_step(
            operation="Criação do Objeto",
            description=f"Criação a partir de Numerador: {numerator} e Denominador: {denominator}",
            before={"numerador_inicial": numerator, "denominador_inicial": denominator},
            after=self,
            explanation="As expressões de entrada foram simplificadas e armazenadas."
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
                    operation="Multiplicação por Escalar",
                    description=f"Multiplicação por escalar: G(s) * {other}",
                    before=self,
                    after=result,
                    explanation="O numerador da função de transferência foi multiplicado pelo escalar."
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
        result.history.steps = list(self.history.steps)
        result.history.steps.extend(other.history.steps)
        result.history.add_step(
            operation="Multiplicação",
            description=f"Multiplicação de funções de transferência",
            before=f"G1(s) = {self}, G2(s) = {other}",
            after=result,
            explanation="Os numeradores e denominadores das funções de transferência foram multiplicados."
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
        result.history.steps = list(self.history.steps)
        result.history.steps.extend(other.history.steps)
        result.history.add_step(
            operation="Adição",
            description=f"Adição de funções de transferência (conexão paralela)",
            before=f"G1(s) = {self}, G2(s) = {other}",
            after=result,
            explanation="As funções de transferência foram somadas para representar uma conexão paralela."
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
        result.history.steps = list(self.history.steps)
        result.history.steps.extend(other.history.steps)
        result.history.add_step(
            operation="Divisão",
            description=f"Divisão de funções de transferência",
            before=f"G1(s) = {self}, G2(s) = {other}",
            after=result,
            explanation="A primeira função de transferência foi dividida pela segunda."
        )

        return result

    def simplify(self) -> 'SymbolicTransferFunction':
        """
        Simplifica a função de transferência

        Returns:
            SymbolicTransferFunction: Função de transferência simplificada
        """
        # Simplifica numerador e denominador
        simplified_num, simplified_den = cancel_common_factors(self.numerator, self.denominator)

        simplified_tf = SymbolicTransferFunction(simplified_num, simplified_den, self.variable)

        # LÓGICA DE HISTÓRICO A SER ADICIONADA:
        simplified_tf.history.steps = list(self.history.steps) # Copia a história do pai
        simplified_tf.history.add_step(
            operation="Simplificação",
            description="Expressão foi simplificada.",
            before=self,
            after=simplified_tf,
            explanation="Fatores comuns entre numerador e denominador podem ter sido cancelados."
        )
        return simplified_tf

    def substitute(self, substitutions: Dict[Symbol, Union[int, float, Symbol]]) -> 'SymbolicTransferFunction':
        """
        Substitui símbolos na função de transferência

        Args:
            substitutions: Dicionário com substituições {símbolo: valor}

        Returns:
            SymbolicTransferFunction: Função com substituições aplicadas
        """
        # Aplica substituições
        new_numerator = self.numerator.subs(substitutions)
        new_denominator = self.denominator.subs(substitutions)

        result = SymbolicTransferFunction(new_numerator, new_denominator, self.variable)
        result.history.steps = self.history.steps.copy()
        result.history.add_step(
            operation="Substituição",
            description=f"Substituição de parâmetros: {substitutions}",
            before=self,
            after=result,
            explanation="Símbolos na função de transferência foram substituídos por valores numéricos ou outros símbolos."
        )

        return result

    def expand(self) -> 'SymbolicTransferFunction':
        """
        Expande numerador e denominador

        Returns:
            SymbolicTransferFunction: Função expandida
        """
        expanded_num = expand(self.numerator)
        expanded_den = expand(self.denominator)

        result = SymbolicTransferFunction(expanded_num, expanded_den, self.variable)
        result.history.steps = self.history.steps.copy()
        result.history.add_step(
            operation="Expansão",
            description="Expansão algébrica de numerador e denominador",
            before=self,
            after=result,
            explanation="Os polinômios do numerador e do denominador foram expandidos."
        )

        return result

    def factor(self) -> 'SymbolicTransferFunction':
        """
        Fatora numerador e denominador

        Returns:
            SymbolicTransferFunction: Função fatorada
        """
        factored_num = factor(self.numerator)
        factored_den = factor(self.denominator)

        result = SymbolicTransferFunction(factored_num, factored_den, self.variable)
        result.history.steps = self.history.steps.copy()
        result.history.add_step(
            operation="Fatoração",
            description="Fatoração de numerador e denominador",
            before=self,
            after=result,
            explanation="Os polinômios do numerador e do denominador foram fatorados."
        )

        return result

    def substitute_param(self, param_symbol: sp.Symbol, value: float) -> 'SymbolicTransferFunction':
        """
        Substitui um parâmetro simbólico por um valor numérico e retorna
        uma nova instância da função de transferência.
        """
        # 1. Validação do Input (Boa prática)
        if not isinstance(param_symbol, sp.Symbol):
            raise TypeError("O parâmetro a ser substituído deve ser um símbolo do SymPy.")

        # 2. Executar a substituição
        new_num = self.numerator.subs(param_symbol, value)
        new_den = self.denominator.subs(param_symbol, value)

        # 3. Criar a nova instância
        new_tf = SymbolicTransferFunction(new_num, new_den, self.variable)

        # 4. Registrar a operação no histórico
        new_tf.history.steps = list(self.history.steps) # Herda a história
        new_tf.history.add_step(
            operation="Substituição de Parâmetro",
            description=f"O símbolo '{param_symbol}' foi substituído pelo valor '{value}'.",
            before=self,
            after=new_tf,
            explanation="Esta operação é fundamental para análises paramétricas e interativas."
        )

        return new_tf

    def feedback(self, H: 'SymbolicTransferFunction' = None, sign: int = -1) -> 'SymbolicTransferFunction':
        """
        Calcula a função de transferência de malha fechada.
        G(s) / (1 + G(s)H(s)) for negative feedback
        G(s) / (1 - G(s)H(s)) for positive feedback

        Args:
            H (SymbolicTransferFunction, optional): Função de transferência da malha de realimentação. Defaults to None (realimentação unitária).
            sign (int, optional): Sinal da realimentação. -1 para negativa, 1 para positiva. Defaults to -1.

        Returns:
            SymbolicTransferFunction: Função de transferência de malha fechada.
        """
        if H is None:
            H = SymbolicTransferFunction(1, 1, self.variable)

        G = self
        closed_loop_num = G.numerator * H.denominator
        closed_loop_den = G.denominator * H.denominator - sign * G.numerator * H.numerator

        new_tf = SymbolicTransferFunction(closed_loop_num, closed_loop_den, self.variable)

        new_tf.history.steps = list(self.history.steps)
        if H.history:
            new_tf.history.steps.extend(H.history.steps)

        new_tf.history.add_step(
            operation="Realimentação",
            description=f"Cálculo da malha fechada com H(s) = {H} e sinal {sign}",
            before=self,
            after=new_tf,
            explanation="A função de transferência de malha fechada foi calculada usando a fórmula G/(1+GH)."
        )
        return new_tf

    def partial_fractions(self) -> sp.Expr:
        """
        Expande em frações parciais

        Returns:
            sp.Expr: Expansão em frações parciais
        """
        expansion = expand_partial_fractions(self.numerator, self.denominator, self.variable)

        self.history.add_step(
            operation="Frações Parciais",
            description="Expansão em frações parciais",
            before=self,
            after=str(expansion),
            explanation="A função de transferência foi decomposta em uma soma de frações mais simples."
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

    def evaluate(self, value: complex) -> complex:
        """
        Avalia a função de transferência para um valor complexo de s.
        """
        # Usa sympy.subs para substituir a variável pelo valor e depois avalia
        # numericamente com .evalf()
        num_val = self.numerator.subs(self.variable, value).evalf()
        den_val = self.denominator.subs(self.variable, value).evalf()

        if den_val == 0:
            # Retorna infinito complexo se o denominador for zero (um polo)
            return sp.zoo

        return complex(num_val / den_val)

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
            laplace_result, _, _ = sp.laplace_transform(time_expr, sp.Symbol('t'), self.variable)

            # Cria nova função de transferência
            result = SymbolicTransferFunction(
                sp.numer(laplace_result),
                sp.denom(laplace_result),
                self.variable
            )

            result.history.steps = self.history.steps.copy()
            result.history.add_step(
                operation="Transformada de Laplace",
                description=f"Aplicada transformada de Laplace: L{{{time_expr}}}",
                before=str(time_expr),
                after=result,
                explanation="A transformada de Laplace foi aplicada a uma expressão no domínio do tempo."
            )

            return result
        except Exception as e:
            self.history.add_step(
                operation="Erro na Transformada de Laplace",
                description=f"Erro na transformada de Laplace: {str(e)}",
                before=str(time_expr),
                after=None,
                explanation="Ocorreu um erro ao tentar aplicar a transformada de Laplace."
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
            operation="Transformação de Tustin",
            description=f"Aplicada transformação de Tustin com T={T}",
            before=self,
            after=result,
            explanation="A transformação de Tustin foi usada para discretizar o sistema."
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
                operation="Análise de Margens",
                description="Cálculo de margens de ganho e fase",
                before=self,
                after=f"Magnitude: {magnitude}, Fase: {phase}",
                explanation="A resposta em frequência foi calculada para análise de estabilidade."
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
