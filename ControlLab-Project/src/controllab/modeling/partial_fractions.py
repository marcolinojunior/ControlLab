"""
Módulo de Expansão em Frações Parciais
=====================================

Este módulo implementa expansão em frações parciais com explicações
pedagógicas detalhadas dos métodos de cálculo.

Classes:
    PartialFractionExpander: Expansor com histórico pedagógico
    
Funções:
    explain_partial_fractions: Explica a expansão passo a passo
    find_residues_symbolic: Calcula resíduos simbolicamente
    handle_repeated_poles: Trata polos repetidos
    handle_complex_poles: Trata pares de polos complexos
"""

import sympy as sp
from sympy import symbols, apart, Poly, factor, expand, simplify, cancel
from sympy import residue, limit, diff, I, conjugate, re, im
from typing import Dict, List, Tuple, Any, Optional
import warnings


class PartialFractionHistory:
    """Histórico detalhado para expansão em frações parciais"""
    
    def __init__(self):
        self.steps = []
        self.method_used = ""
        self.residues = {}
        self.explanation = ""
        
    def add_step(self, step_type: str, description: str, expression: Any, 
                 method: str = "", details: str = ""):
        """Adiciona um passo ao histórico"""
        step = {
            'type': step_type,
            'description': description,
            'expression': expression,
            'method': method,
            'details': details,
            'step_number': len(self.steps) + 1
        }
        self.steps.append(step)
        
    def get_formatted_explanation(self) -> str:
        """Retorna explicação formatada da expansão"""
        if not self.steps:
            return "Nenhuma expansão foi realizada."
            
        explanation = "📐 EXPANSÃO EM FRAÇÕES PARCIAIS:\n"
        explanation += "=" * 60 + "\n"
        
        for step in self.steps:
            explanation += f"\n🔸 Passo {step['step_number']}: {step['description']}\n"
            explanation += f"   Expressão: {step['expression']}\n"
            
            if step['method']:
                explanation += f"   Método: {step['method']}\n"
            if step['details']:
                explanation += f"   Detalhes: {step['details']}\n"
            
            explanation += "-" * 40 + "\n"
        
        if self.residues:
            explanation += "\n📊 RESÍDUOS CALCULADOS:\n"
            for pole, residue_val in self.residues.items():
                explanation += f"   Polo s = {pole}: Resíduo = {residue_val}\n"
        
        return explanation


class PartialFractionExpander:
    """
    Expansor de frações parciais com capacidades pedagógicas avançadas
    """
    
    def __init__(self, variable='s'):
        self.var = symbols(variable)
        self.history = PartialFractionHistory()
        
    def expand(self, tf_expr, variable='s', show_steps: bool = True):
        """
        Método de conveniência para expansão em frações parciais
        
        Args:
            tf_expr: Expressão da função de transferência
            variable: Variável (padrão 's')
            show_steps: Se deve mostrar os passos
            
        Returns:
            Expansão em frações parciais
        """
        if hasattr(tf_expr, 'as_numer_denom'):
            # Se for uma expressão SymPy
            numerator, denominator = tf_expr.as_numer_denom()
        else:
            # Se for uma fração já separada
            numerator = tf_expr
            denominator = 1
            
        return self.expand_rational_function(numerator, denominator, show_steps)
        
    def expand_rational_function(self, numerator, denominator, show_steps: bool = True):
        """
        Expande uma função racional em frações parciais
        
        Args:
            numerator: Numerador da função
            denominator: Denominador da função  
            show_steps: Se deve mostrar os passos
            
        Returns:
            Expansão em frações parciais
        """
        if show_steps:
            self.history = PartialFractionHistory()
        
        # Função racional original
        rational_func = numerator / denominator
        
        if show_steps:
            self.history.add_step(
                "setup", 
                "Função racional original",
                rational_func,
                details=f"F(s) = ({numerator}) / ({denominator})"
            )
        
        try:
            # Verificar se é própria ou imprópria
            num_degree = sp.degree(numerator, self.var)
            den_degree = sp.degree(denominator, self.var)
            
            if num_degree >= den_degree:
                # Função imprópria - fazer divisão polinomial primeiro
                quotient, remainder = sp.div(numerator, denominator, self.var)
                
                if show_steps:
                    self.history.add_step(
                        "division",
                        "Divisão polinomial (função imprópria)",
                        f"{quotient} + ({remainder})/({denominator})",
                        method="Divisão longa de polinômios",
                        details=f"Grau num={num_degree} ≥ grau den={den_degree}"
                    )
                
                # Expandir apenas a parte fracionária
                if remainder != 0:
                    partial_expansion = self._expand_proper_fraction(
                        remainder, denominator, show_steps
                    )
                    total_expansion = quotient + partial_expansion
                else:
                    total_expansion = quotient
            else:
                # Função própria - expandir diretamente
                total_expansion = self._expand_proper_fraction(
                    numerator, denominator, show_steps
                )
            
            if show_steps:
                self.history.add_step(
                    "result",
                    "Expansão final em frações parciais",
                    total_expansion,
                    details="Resultado completo da expansão"
                )
            
            return total_expansion
            
        except Exception as e:
            if show_steps:
                self.history.add_step(
                    "error",
                    "Erro na expansão",
                    str(e)
                )
            raise ValueError(f"Erro na expansão em frações parciais: {e}")
    
    def _expand_proper_fraction(self, numerator, denominator, show_steps: bool):
        """Expande uma fração própria"""
        
        # Fatorar o denominador
        factored_den = factor(denominator)
        
        if show_steps:
            self.history.add_step(
                "factorization",
                "Fatoração do denominador",
                factored_den,
                method="Fatoração simbólica",
                details="Identificando polos e suas multiplicidades"
            )
        
        # Identificar polos e multiplicidades
        poles_info = self._analyze_poles(factored_den, show_steps)
        
        # Usar SymPy apart para expansão básica
        expansion = apart(numerator/denominator, self.var)
        
        if show_steps:
            self.history.add_step(
                "expansion",
                "Expansão automática",
                expansion,
                method="Algoritmo de frações parciais",
                details="Expansão usando método residual"
            )
        
        # Calcular resíduos manualmente para fins pedagógicos
        self._calculate_residues_manually(numerator, denominator, poles_info, show_steps)
        
        return expansion
    
    def _analyze_poles(self, factored_expr, show_steps: bool) -> Dict:
        """Analisa polos e suas multiplicidades"""
        poles_info = {}
        
        # Extrair fatores do denominador
        factors = sp.Mul.make_args(factored_expr)
        
        for factor_expr in factors:
            if factor_expr.has(self.var):
                # Encontrar raízes do fator
                roots = sp.solve(factor_expr, self.var)
                for root in roots:
                    # Determinar multiplicidade
                    multiplicity = self._get_pole_multiplicity(factored_expr, root)
                    poles_info[root] = {
                        'multiplicity': multiplicity,
                        'type': self._classify_pole(root)
                    }
        
        if show_steps:
            poles_summary = []
            for pole, info in poles_info.items():
                poles_summary.append(f"s = {pole} (mult={info['multiplicity']}, {info['type']})")
            
            self.history.add_step(
                "poles",
                "Análise de polos",
                ", ".join(poles_summary),
                details=f"Total de {len(poles_info)} polos distintos"
            )
        
        return poles_info
    
    def _get_pole_multiplicity(self, factored_expr, pole) -> int:
        """Determina a multiplicidade de um polo"""
        # Substituir s por pole+ε e ver ordem do zero
        epsilon = symbols('epsilon')
        substituted = factored_expr.subs(self.var, pole + epsilon)
        
        # Encontrar a menor potência de epsilon
        series = sp.series(substituted, epsilon, 0)
        
        # Extrair a ordem
        for term in series.args:
            if term.has(epsilon):
                order = sp.degree(term, epsilon)
                if order > 0:
                    return order
        
        return 1  # Default para polos simples
    
    def _classify_pole(self, pole) -> str:
        """Classifica o tipo de polo"""
        if pole.is_real:
            return "real"
        elif pole.has(I):
            return "complexo"
        else:
            return "simbólico"
    
    def _calculate_residues_manually(self, numerator, denominator, poles_info, show_steps: bool):
        """Calcula resíduos manualmente para demonstração"""
        
        if not show_steps:
            return
        
        self.history.add_step(
            "residue_start",
            "Início do cálculo de resíduos",
            "Aplicando método cover-up e derivadas",
            method="Método dos resíduos"
        )
        
        for pole, info in poles_info.items():
            if info['multiplicity'] == 1:
                # Polo simples - método cover-up
                residue_val = self._cover_up_method(numerator, denominator, pole)
                
                self.history.add_step(
                    "residue_simple",
                    f"Resíduo para polo simples s = {pole}",
                    residue_val,
                    method="Método cover-up",
                    details=f"R = lim(s→{pole}) (s-{pole})*F(s)"
                )
                
            else:
                # Polo múltiplo - método das derivadas
                residues = self._repeated_pole_method(numerator, denominator, pole, info['multiplicity'])
                
                for order, res_val in enumerate(residues):
                    self.history.add_step(
                        "residue_multiple",
                        f"Resíduo para polo múltiplo s = {pole} (ordem {order+1})",
                        res_val,
                        method="Método das derivadas",
                        details=f"R_{order+1} calculado por derivação"
                    )
            
            self.history.residues[pole] = residue_val if info['multiplicity'] == 1 else residues
    
    def _cover_up_method(self, numerator, denominator, pole):
        """Método cover-up para polos simples"""
        # Criar fator correspondente ao polo
        factor = (self.var - pole)
        
        # "Cobrir" o fator no denominador e calcular limite
        covered_expr = (factor * numerator) / denominator
        residue_val = limit(covered_expr, self.var, pole)
        
        return residue_val
    
    def _repeated_pole_method(self, numerator, denominator, pole, multiplicity):
        """Método das derivadas para polos múltiplos"""
        residues = []
        
        # Fator múltiplo
        factor = (self.var - pole)**multiplicity
        
        # Expressão base
        base_expr = (factor * numerator) / denominator
        
        for k in range(multiplicity):
            # k-ésima derivada
            if k == 0:
                derivative = base_expr
            else:
                derivative = diff(base_expr, self.var, k)
            
            # Calcular limite
            residue_val = limit(derivative, self.var, pole) / sp.factorial(k)
            residues.append(residue_val)
        
        return residues


def explain_partial_fractions(tf_expr, variable='s'):
    """
    Explica a expansão em frações parciais passo a passo
    
    Args:
        tf_expr: Expressão da função de transferência
        variable: Variável (padrão 's')
        
    Returns:
        Objeto com expansão e explicação
    """
    expander = PartialFractionExpander(variable)
    
    # Extrair numerador e denominador
    numerator = sp.numer(tf_expr)
    denominator = sp.denom(tf_expr)
    
    # Expandir
    expansion = expander.expand_rational_function(numerator, denominator, show_steps=True)
    
    # Criar objeto resultado
    class PartialFractionResult:
        def __init__(self, expansion, history):
            self.expansion = expansion
            self.history = history
            
        def show_explanation(self):
            return self.history.get_formatted_explanation()
            
        def __str__(self):
            return str(self.expansion)
            
        def __repr__(self):
            return f"PartialFractionResult({self.expansion})"
    
    return PartialFractionResult(expansion, expander.history)


def find_residues_symbolic(numerator, denominator, poles):
    """
    Encontra resíduos simbolicamente para lista de polos
    
    Args:
        numerator: Numerador da função
        denominator: Denominador da função
        poles: Lista de polos
        
    Returns:
        Dicionário {polo: resíduo}
    """
    var = list(numerator.free_symbols | denominator.free_symbols)[0]
    expander = PartialFractionExpander(str(var))
    
    residues = {}
    
    for pole in poles:
        try:
            # Usar método cover-up
            residue_val = expander._cover_up_method(numerator, denominator, pole)
            residues[pole] = residue_val
        except Exception as e:
            residues[pole] = f"Erro: {e}"
    
    return residues


def handle_repeated_poles(tf_expr, pole, multiplicity):
    """
    Trata polos repetidos especificamente
    
    Args:
        tf_expr: Função de transferência
        pole: Polo repetido
        multiplicity: Multiplicidade do polo
        
    Returns:
        Lista de resíduos para cada ordem
    """
    var = list(tf_expr.free_symbols)[0]
    numerator = sp.numer(tf_expr)
    denominator = sp.denom(tf_expr)
    
    expander = PartialFractionExpander(str(var))
    residues = expander._repeated_pole_method(numerator, denominator, pole, multiplicity)
    
    return residues


def handle_complex_poles(tf_expr, complex_pole_pair):
    """
    Trata pares de polos complexos conjugados
    
    Args:
        tf_expr: Função de transferência
        complex_pole_pair: Par de polos complexos
        
    Returns:
        Expansão para polos complexos
    """
    var = list(tf_expr.free_symbols)[0]
    
    # Expandir usando apart do SymPy
    expansion = apart(tf_expr, var)
    
    # Identificar termos correspondentes aos polos complexos
    complex_terms = []
    for term in sp.Add.make_args(expansion):
        if any(pole in term.free_symbols for pole in complex_pole_pair if pole.has(I)):
            complex_terms.append(term)
    
    return complex_terms


# Funções auxiliares para casos especiais
def convert_to_quadratic_form(complex_pole_pair, residue_pair):
    """
    Converte par de polos complexos para forma quadrática real
    
    Args:
        complex_pole_pair: Par de polos complexos conjugados
        residue_pair: Par de resíduos correspondentes
        
    Returns:
        Forma quadrática equivalente
    """
    pole1, pole2 = complex_pole_pair
    res1, res2 = residue_pair
    
    # Assumindo polos na forma σ ± jω
    sigma = re(pole1)
    omega = abs(im(pole1))
    
    # Formar quadrática (s² + 2σs + σ² + ω²)
    s = symbols('s')
    quadratic = s**2 - 2*sigma*s + (sigma**2 + omega**2)
    
    # Calcular coeficientes para forma (As + B)/(s² + 2σs + σ² + ω²)
    A = 2*re(res1)
    B = -2*im(res1)*sigma - 2*re(res1)*sigma
    
    return (A*s + B) / quadratic
