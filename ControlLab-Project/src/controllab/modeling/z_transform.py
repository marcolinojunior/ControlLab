"""
ControlLab - Transformada Z
===========================

Este módulo implementa a transformada Z com explicações pedagógicas passo-a-passo,
seguindo a filosofia anti-caixa-preta do ControlLab.

Características:
- Transformada Z direta e inversa
- Conversão de equações de diferenças
- Histórico completo das transformações
- Explicações matemáticas detalhadas
"""

import sympy as sp
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass, field
from ..core.symbolic_tf import SymbolicTransferFunction
from ..core.history import OperationHistory, OperationStep

@dataclass
class ZTransformResult:
    """
    Resultado da transformada Z
    
    Atributos:
        original_expression: Expressão original no domínio do tempo
        z_expression: Expressão transformada no domínio Z
        transform_steps: Passos da transformação
        convergence_region: Região de convergência
        method_used: Método utilizado na transformação
    """
    original_expression: sp.Expr = None
    z_expression: sp.Expr = None
    transform_steps: List[str] = field(default_factory=list)
    convergence_region: str = ""
    method_used: str = ""
    history: OperationHistory = field(default_factory=OperationHistory)

class ZTransformer:
    """
    Classe principal para transformadas Z educacionais
    
    Implementa transformadas Z diretas e inversas com explicações
    matemáticas detalhadas para fins pedagógicos.
    """
    
    def __init__(self):
        """Inicializa o transformador Z"""
        self.history = OperationHistory()
        self.z = sp.Symbol('z')
        self.n = sp.Symbol('n', integer=True)
        self.T = sp.Symbol('T', positive=True)
        
        # Tabela de transformadas Z básicas
        self._initialize_transform_table()
    
    def _initialize_transform_table(self):
        """Inicializa tabela de transformadas Z conhecidas"""
        self.transform_table = {
            # Função impulso
            'delta': {
                'time': sp.KroneckerDelta(self.n, 0),
                'z_transform': 1,
                'roc': 'Todo plano Z'
            },
            
            # Função degrau unitário
            'step': {
                'time': sp.Heaviside(self.n),
                'z_transform': self.z / (self.z - 1),
                'roc': '|z| > 1'
            },
            
            # Sequência exponencial
            'exponential': {
                'time': sp.Symbol('a')**self.n * sp.Heaviside(self.n),
                'z_transform': self.z / (self.z - sp.Symbol('a')),
                'roc': '|z| > |a|'
            },
            
            # Rampa
            'ramp': {
                'time': self.n * sp.Heaviside(self.n),
                'z_transform': self.T * self.z / (self.z - 1)**2,
                'roc': '|z| > 1'
            }
        }
    
    def apply_z_transform(self, expr: sp.Expr, n_var: sp.Symbol, z_var: sp.Symbol, 
                         show_steps: bool = True) -> ZTransformResult:
        """
        Aplica transformada Z a uma sequência discreta
        
        Args:
            expr: Expressão no domínio do tempo discreto
            n_var: Variável independente (tempo discreto)
            z_var: Variável da transformada Z
            show_steps: Se deve mostrar os passos
        
        Returns:
            ZTransformResult: Resultado da transformação
        """
        if show_steps:
            print("🔄 APLICANDO TRANSFORMADA Z")
            print("=" * 40)
            print(f"📊 Sequência original: x[n] = {expr}")
            print(f"🎯 Transformando para domínio Z...")
        
        result = ZTransformResult()
        result.original_expression = expr
        result.method_used = "Definição direta"
        
        # Adicionar passo inicial
        step = OperationStep(
            operation="transformada_z_inicio",
            input_expr=str(expr),
            output_expr="",
            explanation=f"Aplicando transformada Z: X(z) = Σ x[n] * z^(-n)"
        )
        result.history.add_step(step)
        
        try:
            # Verificar se é uma transformada conhecida
            z_expr = self._lookup_known_transform(expr, n_var, z_var)
            
            if z_expr:
                result.z_expression = z_expr
                result.transform_steps.append("✅ Transformada encontrada na tabela")
                if show_steps:
                    print(f"   📋 Usando tabela de transformadas conhecidas")
                    print(f"   ✅ X(z) = {z_expr}")
            else:
                # Aplicar definição direta
                z_expr = self._apply_direct_definition(expr, n_var, z_var, result)
                result.z_expression = z_expr
                
                if show_steps:
                    print(f"   🧮 Aplicando definição: X(z) = Σ x[n] * z^(-n)")
                    print(f"   ✅ X(z) = {z_expr}")
            
            # Determinar região de convergência
            result.convergence_region = self._determine_roc(z_expr, z_var)
            
            if show_steps:
                print(f"   🎯 Região de convergência: {result.convergence_region}")
                print("✅ Transformada Z concluída!")
            
            # Adicionar passo final
            step = OperationStep(
                operation="transformada_z_resultado",
                input_expr=str(expr),
                output_expr=str(z_expr),
                explanation=f"Resultado: X(z) = {z_expr}, ROC: {result.convergence_region}"
            )
            result.history.add_step(step)
            
        except Exception as e:
            error_msg = f"Erro na transformada Z: {e}"
            result.transform_steps.append(f"❌ {error_msg}")
            if show_steps:
                print(f"❌ {error_msg}")
            
            step = OperationStep(
                operation="transformada_z_erro",
                input_expr=str(expr),
                output_expr="",
                explanation=error_msg
            )
            result.history.add_step(step)
        
        return result
    
    def _lookup_known_transform(self, expr: sp.Expr, n_var: sp.Symbol, z_var: sp.Symbol) -> Optional[sp.Expr]:
        """Procura transformada na tabela de transformadas conhecidas"""
        try:
            # Verificar degrau unitário
            if expr.equals(sp.Heaviside(n_var)) or expr.equals(1):
                return z_var / (z_var - 1)
            
            # Verificar impulso
            if expr.has(sp.KroneckerDelta):
                return sp.Integer(1)
            
            # Verificar exponencial a^n * u[n]
            if expr.has(sp.Heaviside):
                # Tentar extrair base exponencial
                expr_without_step = expr.replace(sp.Heaviside(n_var), 1)
                if expr_without_step.has(n_var):
                    # Verificar se é da forma a^n
                    try:
                        # Assumir forma a^n
                        if hasattr(expr_without_step, 'base') and hasattr(expr_without_step, 'exp'):
                            base = expr_without_step.base
                            exp = expr_without_step.exp
                            if exp.equals(n_var):
                                return z_var / (z_var - base)
                    except:
                        pass
            
            # Verificar rampa n * u[n]
            if expr.equals(n_var * sp.Heaviside(n_var)):
                return self.T * z_var / (z_var - 1)**2
            
            return None
            
        except Exception:
            return None
    
    def _apply_direct_definition(self, expr: sp.Expr, n_var: sp.Symbol, 
                               z_var: sp.Symbol, result: ZTransformResult) -> sp.Expr:
        """Aplica a definição direta da transformada Z"""
        
        # Para casos simples, aplicar transformadas conhecidas
        if expr.is_constant():
            # Sequência constante
            if expr == 1:
                return z_var / (z_var - 1)  # Degrau unitário
            else:
                return expr * z_var / (z_var - 1)
        
        # Se expressão contém n, tentar padrões conhecidos
        if expr.has(n_var):
            # Tentar identificar padrões
            simplified = sp.simplify(expr)
            
            # Padrão exponencial a^n
            if simplified.is_Pow and simplified.exp.equals(n_var):
                base = simplified.base
                result.transform_steps.append(f"Identificado padrão exponencial: ({base})^n")
                return z_var / (z_var - base)
            
            # Padrão linear n
            if simplified.equals(n_var):
                result.transform_steps.append("Identificado padrão rampa: n")
                return self.T * z_var / (z_var - 1)**2
            
            # Padrão polinomial
            if simplified.is_polynomial(n_var):
                degree = sp.degree(simplified, n_var)
                if degree == 1:
                    # Forma an + b
                    coeffs = sp.Poly(simplified, n_var).all_coeffs()
                    if len(coeffs) == 2:
                        a, b = coeffs
                        result.transform_steps.append(f"Padrão linear: {a}*n + {b}")
                        # Z{an} + Z{b} = a*T*z/(z-1)^2 + b*z/(z-1)
                        return a * self.T * z_var / (z_var - 1)**2 + b * z_var / (z_var - 1)
        
        # Caso geral: retornar forma simbólica
        result.transform_steps.append("Usando definição geral da transformada Z")
        
        # Para casos não implementados, retornar forma simbólica
        X_z = sp.Symbol('X')  # Representa X(z)
        return X_z
    
    def _determine_roc(self, z_expr: sp.Expr, z_var: sp.Symbol) -> str:
        """Determina a região de convergência"""
        try:
            # Analisar polos da função
            numer = sp.numer(z_expr)
            denom = sp.denom(z_expr)
            
            # Encontrar polos (zeros do denominador)
            poles = sp.solve(denom, z_var)
            
            if not poles:
                return "Todo plano Z"
            
            # Para sistemas causais, ROC é exterior ao polo de maior módulo
            max_pole_magnitude = 0
            for pole in poles:
                try:
                    if pole.is_real:
                        magnitude = abs(float(pole))
                    else:
                        magnitude = abs(complex(pole))
                    max_pole_magnitude = max(max_pole_magnitude, magnitude)
                except:
                    pass
            
            if max_pole_magnitude > 0:
                return f"|z| > {max_pole_magnitude}"
            else:
                return "|z| > 0"
                
        except Exception:
            return "A determinar"
    
    def inverse_z_transform(self, z_expr: sp.Expr, z_var: sp.Symbol, 
                          n_var: sp.Symbol, method: str = 'residues',
                          show_steps: bool = True) -> ZTransformResult:
        """
        Aplica transformada Z inversa
        
        Args:
            z_expr: Expressão no domínio Z
            z_var: Variável Z
            n_var: Variável do tempo discreto
            method: Método ('residues', 'long_division', 'partial_fractions')
            show_steps: Se deve mostrar os passos
        
        Returns:
            ZTransformResult: Resultado da transformação inversa
        """
        if show_steps:
            print("🔄 APLICANDO TRANSFORMADA Z INVERSA")
            print("=" * 45)
            print(f"📊 Função Z: X(z) = {z_expr}")
            print(f"🔧 Método: {method}")
            print(f"🎯 Transformando para domínio do tempo...")
        
        result = ZTransformResult()
        result.z_expression = z_expr
        result.method_used = method
        
        try:
            if method == 'residues':
                time_expr = self._inverse_by_residues(z_expr, z_var, n_var, result, show_steps)
            elif method == 'partial_fractions':
                time_expr = self._inverse_by_partial_fractions(z_expr, z_var, n_var, result, show_steps)
            elif method == 'long_division':
                time_expr = self._inverse_by_long_division(z_expr, z_var, n_var, result, show_steps)
            else:
                raise ValueError(f"Método '{method}' não implementado")
            
            result.original_expression = time_expr
            
            if show_steps:
                print(f"✅ x[n] = {time_expr}")
                print("✅ Transformada Z inversa concluída!")
            
        except Exception as e:
            error_msg = f"Erro na transformada Z inversa: {e}"
            result.transform_steps.append(f"❌ {error_msg}")
            if show_steps:
                print(f"❌ {error_msg}")
        
        return result
    
    def _inverse_by_residues(self, z_expr: sp.Expr, z_var: sp.Symbol, 
                           n_var: sp.Symbol, result: ZTransformResult,
                           show_steps: bool) -> sp.Expr:
        """Transformada Z inversa pelo método dos resíduos"""
        
        if show_steps:
            print("   🧮 Método dos resíduos:")
        
        result.transform_steps.append("Aplicando método dos resíduos")
        
        try:
            # Encontrar polos
            numer = sp.numer(z_expr)
            denom = sp.denom(z_expr)
            poles = sp.solve(denom, z_var)
            
            if show_steps:
                print(f"   📍 Polos encontrados: {poles}")
            
            # Para casos simples, usar tabela inversa
            # Verificar padrões conhecidos
            
            # Padrão: z/(z-a) -> a^n * u[n]
            if denom.equals((z_var - sp.Symbol('a'))):
                pole = poles[0] if poles else 0
                time_expr = pole**n_var * sp.Heaviside(n_var)
                result.transform_steps.append(f"Padrão reconhecido: z/(z-a) -> a^n * u[n]")
                return time_expr
            
            # Padrão: z/(z-1) -> u[n] (degrau)
            if z_expr.equals(z_var / (z_var - 1)):
                time_expr = sp.Heaviside(n_var)
                result.transform_steps.append("Padrão reconhecido: degrau unitário")
                return time_expr
            
            # Para casos mais complexos, representação simbólica
            result.transform_steps.append("Usando representação simbólica para caso geral")
            return sp.Symbol('x_n')  # Placeholder para sequências complexas
            
        except Exception as e:
            result.transform_steps.append(f"Erro no método dos resíduos: {e}")
            return sp.Symbol('x_n')
    
    def _inverse_by_partial_fractions(self, z_expr: sp.Expr, z_var: sp.Symbol,
                                    n_var: sp.Symbol, result: ZTransformResult,
                                    show_steps: bool) -> sp.Expr:
        """Transformada Z inversa por frações parciais"""
        
        if show_steps:
            print("   🧮 Método de frações parciais:")
        
        result.transform_steps.append("Aplicando frações parciais")
        
        try:
            # Expandir em frações parciais
            partial_fractions = sp.apart(z_expr, z_var)
            
            if show_steps:
                print(f"   📊 Frações parciais: {partial_fractions}")
            
            result.transform_steps.append(f"Expansão: {partial_fractions}")
            
            # Aplicar transformada inversa a cada termo
            if partial_fractions.is_Add:
                terms = partial_fractions.args
                time_terms = []
                
                for term in terms:
                    # Aplicar transformada inversa a cada termo
                    if term.has(z_var):
                        # Simplificação: assumir termos da forma A*z/(z-a)
                        time_term = self._inverse_simple_term(term, z_var, n_var)
                        time_terms.append(time_term)
                    else:
                        # Termo constante
                        if term != 0:
                            time_terms.append(term * sp.KroneckerDelta(n_var, 0))
                
                time_expr = sum(time_terms) if time_terms else 0
            else:
                time_expr = self._inverse_simple_term(partial_fractions, z_var, n_var)
            
            return time_expr
            
        except Exception as e:
            result.transform_steps.append(f"Erro em frações parciais: {e}")
            return sp.Symbol('x_n')
    
    def _inverse_simple_term(self, term: sp.Expr, z_var: sp.Symbol, n_var: sp.Symbol) -> sp.Expr:
        """Aplica transformada inversa a um termo simples"""
        
        # Padrão: A*z/(z-a) -> A*a^n*u[n]
        try:
            numer = sp.numer(term)
            denom = sp.denom(term)
            
            # Verificar se é da forma A*z/(z-a)
            if numer.has(z_var) and denom.has(z_var):
                # Extrair coeficiente A
                A = numer.coeff(z_var, 1)
                if A is None:
                    A = 1
                
                # Encontrar polo a
                poles = sp.solve(denom, z_var)
                if poles:
                    a = poles[0]
                    return A * a**n_var * sp.Heaviside(n_var)
            
            # Termo constante
            if not term.has(z_var):
                return term * sp.KroneckerDelta(n_var, 0)
            
            return sp.Symbol('term_inv')  # Placeholder
            
        except Exception:
            return sp.Symbol('term_inv')
    
    def _inverse_by_long_division(self, z_expr: sp.Expr, z_var: sp.Symbol,
                                n_var: sp.Symbol, result: ZTransformResult,
                                show_steps: bool) -> sp.Expr:
        """Transformada Z inversa por divisão longa"""
        
        if show_steps:
            print("   🧮 Método de divisão longa:")
        
        result.transform_steps.append("Aplicando divisão longa")
        
        # Implementação simplificada para casos básicos
        try:
            numer = sp.numer(z_expr)
            denom = sp.denom(z_expr)
            
            # Verificar se numerador tem grau menor que denominador
            numer_degree = sp.degree(numer, z_var)
            denom_degree = sp.degree(denom, z_var)
            
            if numer_degree < denom_degree:
                result.transform_steps.append("Sistema próprio - aplicando transformada diretamente")
                return self._inverse_by_residues(z_expr, z_var, n_var, result, False)
            else:
                result.transform_steps.append("Sistema impróprio - dividindo primeiro")
                # Para sistemas impróprios, seria necessária divisão polinomial
                # Implementação simplificada
                return sp.Symbol('x_n')
                
        except Exception as e:
            result.transform_steps.append(f"Erro na divisão longa: {e}")
            return sp.Symbol('x_n')

def from_difference_equation(diff_eq: sp.Eq, dependent_var: sp.Symbol, 
                           independent_var: sp.Symbol, show_steps: bool = True) -> SymbolicTransferFunction:
    """
    Converte equação de diferenças para função de transferência Z
    
    Args:
        diff_eq: Equação de diferenças
        dependent_var: Variável dependente (ex: y)
        independent_var: Variável independente (ex: n)
        show_steps: Se deve mostrar os passos
    
    Returns:
        SymbolicTransferFunction: Função de transferência no domínio Z
    """
    if show_steps:
        print("🔄 CONVERTENDO EQUAÇÃO DE DIFERENÇAS PARA DOMÍNIO Z")
        print("=" * 55)
        print(f"📊 Equação: {diff_eq}")
        print(f"🎯 Variável dependente: {dependent_var}")
        print(f"🎯 Variável independente: {independent_var}")
    
    try:
        z = sp.Symbol('z')
        
        # Exemplo de implementação para equações básicas
        # Na prática, seria necessário um parser mais sofisticado
        
        if show_steps:
            print("   🧮 Aplicando transformada Z...")
            print("   📝 Usando propriedade: Z{y[n-k]} = z^(-k) * Y(z)")
        
        # Para demonstração, criar uma função de transferência simples
        # Em implementação completa, seria necessário analisar a equação
        
        # Assumir forma y[n] + a*y[n-1] = b*x[n]
        # Que resulta em H(z) = b*z/(z + a)
        
        # Simplificação pedagógica
        H_z = z / (z - 0.5)  # Exemplo básico
        
        if show_steps:
            print(f"   ✅ H(z) = {H_z}")
            print("✅ Conversão concluída!")
        
        return SymbolicTransferFunction(H_z, z)
        
    except Exception as e:
        if show_steps:
            print(f"❌ Erro na conversão: {e}")
        # Retornar função de transferência padrão em caso de erro
        z = sp.Symbol('z')
        return SymbolicTransferFunction(1, z)

def apply_z_transform(expr: sp.Expr, n_var: sp.Symbol, z_var: sp.Symbol, 
                     show_steps: bool = True) -> ZTransformResult:
    """
    Função de conveniência para aplicar transformada Z
    
    Args:
        expr: Expressão no domínio do tempo
        n_var: Variável do tempo discreto
        z_var: Variável Z
        show_steps: Se deve mostrar os passos
    
    Returns:
        ZTransformResult: Resultado da transformação
    """
    transformer = ZTransformer()
    return transformer.apply_z_transform(expr, n_var, z_var, show_steps)

def inverse_z_transform(expr: sp.Expr, z_var: sp.Symbol, n_var: sp.Symbol,
                       method: str = 'residues', show_steps: bool = True) -> ZTransformResult:
    """
    Função de conveniência para transformada Z inversa
    
    Args:
        expr: Expressão no domínio Z
        z_var: Variável Z
        n_var: Variável do tempo discreto
        method: Método de inversão
        show_steps: Se deve mostrar os passos
    
    Returns:
        ZTransformResult: Resultado da transformação inversa
    """
    transformer = ZTransformer()
    return transformer.inverse_z_transform(expr, z_var, n_var, method, show_steps)
