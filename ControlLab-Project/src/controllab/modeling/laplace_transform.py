"""
Módulo de Transformada de Laplace
================================

Este módulo implementa transformadas de Laplace com foco pedagógico,
mostrando todos os passos intermediários para fins educacionais.

Classes:
    LaplaceTransformer: Classe principal para operações de transformadas
    
Funções:
    from_ode: Converte EDO para função de transferência
    apply_laplace_transform: Aplica transformada de Laplace
    inverse_laplace_transform: Aplica transformada inversa
"""

import sympy as sp
from sympy import symbols, Function, Eq, Heaviside, DiracDelta
from sympy.integrals.transforms import laplace_transform, inverse_laplace_transform as inv_laplace
from typing import Dict, List, Optional, Tuple, Any
import warnings


class TransformationHistory:
    """Histórico detalhado de transformações para fins pedagógicos"""
    
    def __init__(self):
        self.steps = []
        self.current_expression = None
        
    def add_step(self, description: str, expression: Any, explanation: str = ""):
        """Adiciona um passo ao histórico"""
        step = {
            'step': len(self.steps) + 1,
            'description': description,
            'expression': expression,
            'explanation': explanation
        }
        self.steps.append(step)
        self.current_expression = expression
        
    def get_formatted_steps(self) -> str:
        """Retorna os passos formatados para exibição"""
        if not self.steps:
            return "Nenhum passo registrado."
            
        formatted = "🔄 PASSOS DA TRANSFORMAÇÃO:\n"
        formatted += "=" * 50 + "\n"
        
        for step in self.steps:
            formatted += f"Passo {step['step']}: {step['description']}\n"
            formatted += f"Expressão: {step['expression']}\n"
            if step['explanation']:
                formatted += f"Explicação: {step['explanation']}\n"
            formatted += "-" * 30 + "\n"
            
        return formatted


class LaplaceTransformer:
    """
    Classe principal para operações de transformadas de Laplace
    com histórico pedagógico detalhado.
    """
    
    def __init__(self):
        self.t = symbols('t', real=True, positive=True)
        self.s = symbols('s', complex=True)
        self.history = TransformationHistory()
        
    def transform_function(self, expr, show_steps: bool = True) -> Tuple[Any, Any, Any]:
        """
        Aplica transformada de Laplace a uma expressão
        
        Args:
            expr: Expressão a ser transformada
            show_steps: Se deve mostrar os passos
            
        Returns:
            tuple: (transformed_expr, convergence_condition, original_expr)
        """
        if show_steps:
            self.history = TransformationHistory()
            self.history.add_step(
                "Expressão original",
                expr,
                f"Aplicando L[{expr}] com t → s"
            )
        
        try:
            # Aplicar transformada de Laplace
            result = laplace_transform(expr, self.t, self.s)
            transformed, convergence, _ = result
            
            if show_steps:
                self.history.add_step(
                    "Transformada aplicada",
                    transformed,
                    f"Condição de convergência: Re(s) > {convergence}"
                )
                
                # Simplificar se possível
                simplified = sp.simplify(transformed)
                if simplified != transformed:
                    self.history.add_step(
                        "Expressão simplificada",
                        simplified,
                        "Aplicando simplificações algébricas"
                    )
                    transformed = simplified
            
            return transformed, convergence, expr
            
        except Exception as e:
            if show_steps:
                self.history.add_step(
                    "Erro na transformação",
                    str(e),
                    "Não foi possível calcular a transformada"
                )
            raise ValueError(f"Erro ao calcular transformada de Laplace: {e}")
    
    def inverse_transform(self, expr, show_steps: bool = True) -> Any:
        """
        Aplica transformada inversa de Laplace
        
        Args:
            expr: Expressão no domínio s
            show_steps: Se deve mostrar os passos
            
        Returns:
            Expressão no domínio do tempo
        """
        if show_steps:
            self.history = TransformationHistory()
            self.history.add_step(
                "Expressão no domínio s",
                expr,
                f"Aplicando L⁻¹[{expr}] com s → t"
            )
        
        try:
            # Aplicar transformada inversa
            result = inv_laplace(expr, self.s, self.t)
            
            if show_steps:
                self.history.add_step(
                    "Transformada inversa aplicada",
                    result,
                    "Resultado no domínio do tempo"
                )
                
                # Simplificar se possível
                simplified = sp.simplify(result)
                if simplified != result:
                    self.history.add_step(
                        "Expressão simplificada",
                        simplified,
                        "Aplicando simplificações algébricas"
                    )
                    result = simplified
            
            return result
            
        except Exception as e:
            if show_steps:
                self.history.add_step(
                    "Erro na transformação inversa",
                    str(e),
                    "Não foi possível calcular a transformada inversa"
                )
            raise ValueError(f"Erro ao calcular transformada inversa: {e}")
    
    def solve_ode_with_laplace(self, ode_eq, func, initial_conditions: Dict = None,
                             show_steps: bool = True) -> Any:
        """
        Resolve EDO usando transformada de Laplace
        
        Args:
            ode_eq: Equação diferencial
            func: Função dependente
            initial_conditions: Condições iniciais
            show_steps: Se deve mostrar os passos
            
        Returns:
            Solução da EDO
        """
        if show_steps:
            self.history = TransformationHistory()
            self.history.add_step(
                "Equação diferencial original",
                ode_eq,
                "EDO a ser resolvida"
            )
        
        # Aplicar condições iniciais padrão se não fornecidas
        if initial_conditions is None:
            initial_conditions = {}
        
        try:
            # Aplicar transformada de Laplace à EDO
            transformed_eq = self._transform_ode(ode_eq, func, initial_conditions, show_steps)
            
            if show_steps:
                self.history.add_step(
                    "EDO transformada",
                    transformed_eq,
                    "Equação algébrica no domínio s"
                )
            
            # Resolver para a função transformada
            func_s = symbols(f"{func.func.__name__}_s")
            solution_s = sp.solve(transformed_eq, func_s)[0]
            
            if show_steps:
                self.history.add_step(
                    "Solução no domínio s",
                    solution_s,
                    f"{func.func.__name__}(s) isolado"
                )
            
            # Aplicar transformada inversa
            solution_t = self.inverse_transform(solution_s, show_steps=False)
            
            if show_steps:
                self.history.add_step(
                    "Solução final",
                    solution_t,
                    "Transformada inversa aplicada"
                )
            
            return solution_t
            
        except Exception as e:
            raise ValueError(f"Erro ao resolver EDO com Laplace: {e}")
    
    def _transform_ode(self, ode_eq, func, initial_conditions: Dict, show_steps: bool) -> Any:
        """Transforma uma EDO usando propriedades da transformada de Laplace"""
        # Esta é uma implementação simplificada
        # Em uma implementação completa, seria necessário tratar derivadas automaticamente
        
        # Por agora, assumimos que o usuário fornece a EDO já em forma adequada
        # para transformação direta
        
        func_name = func.func.__name__
        func_s = symbols(f"{func_name}_s")
        
        # Substituir a função original pela versão transformada
        transformed = ode_eq.subs(func, func_s)
        
        # Aplicar condições iniciais (implementação básica)
        for condition, value in initial_conditions.items():
            if condition.startswith(func_name):
                # Implementação simplificada para condições iniciais
                pass
        
        return transformed


def from_ode(ode_expr, dependent_var, independent_var, initial_conditions=None):
    """
    Converte uma EDO para função de transferência usando Laplace
    
    Args:
        ode_expr: Expressão da EDO
        dependent_var: Variável dependente (função)
        independent_var: Variável independente (tempo)
        initial_conditions: Condições iniciais
        
    Returns:
        Função de transferência da EDO
    """
    try:
        # Para uma EDO linear da forma:
        # a_n*y^(n) + ... + a_1*y' + a_0*y = b_m*u^(m) + ... + b_1*u' + b_0*u
        # A TF é: G(s) = (b_m*s^m + ... + b_1*s + b_0)/(a_n*s^n + ... + a_1*s + a_0)
        
        s = symbols('s')
        
        # Extrair os coeficientes da EDO
        # Por simplicidade, vamos assumir uma forma padrão
        # Exemplo: m*x'' + b*x' + k*x = F
        
        # Identificar derivadas e função base
        func_name = dependent_var.func.__name__
        base_func = dependent_var
        
        # Extrair lado esquerdo (sistema) e lado direito (entrada)
        lhs = ode_expr.lhs if hasattr(ode_expr, 'lhs') else ode_expr
        rhs = ode_expr.rhs if hasattr(ode_expr, 'rhs') else 0
        
        # Simplificação: para EDO de segunda ordem como massa-mola-amortecedor
        # m*x'' + b*x' + k*x = F
        # TF = 1/(m*s^2 + b*s + k)
        
        # Coletar termos com derivadas
        terms = sp.Add.make_args(lhs)
        
        # Extrair coeficientes
        coeffs = {}
        for term in terms:
            if term.has(dependent_var.diff(independent_var, 2)):
                # Segunda derivada
                coeff = term.coeff(dependent_var.diff(independent_var, 2))
                coeffs[2] = coeff if coeff is not None else 1
            elif term.has(dependent_var.diff(independent_var)):
                # Primeira derivada
                coeff = term.coeff(dependent_var.diff(independent_var))
                coeffs[1] = coeff if coeff is not None else 1
            elif term.has(dependent_var):
                # Termo constante
                coeff = term.coeff(dependent_var)
                coeffs[0] = coeff if coeff is not None else 1
        
        # Construir denominador da TF
        denominator = 0
        for order, coeff in coeffs.items():
            denominator += coeff * s**order
        
        # Para entrada unitária, numerador = 1
        numerator = 1
        
        # Importar classe de TF se disponível
        try:
            from ..core.symbolic_tf import SymbolicTransferFunction
            return SymbolicTransferFunction(numerator, denominator, s)
        except ImportError:
            # Retornar expressão simples
            return numerator / denominator
        
    except Exception as e:
        return f"Erro na conversão EDO→TF: {e}"


def apply_laplace_transform(expr, t_var, s_var, show_steps: bool = True):
    """
    Aplica transformada de Laplace a uma expressão
    
    Args:
        expr: Expressão a ser transformada
        t_var: Variável do tempo
        s_var: Variável de Laplace
        show_steps: Se deve mostrar os passos
        
    Returns:
        Expressão transformada
    """
    transformer = LaplaceTransformer()
    transformer.t = t_var
    transformer.s = s_var
    
    transformed, convergence, original = transformer.transform_function(expr, show_steps)
    
    if show_steps:
        print(transformer.history.get_formatted_steps())
    
    return transformed


def inverse_laplace_transform(expr, s_var, t_var, show_steps: bool = True):
    """
    Aplica transformada inversa de Laplace
    
    Args:
        expr: Expressão no domínio s
        s_var: Variável de Laplace
        t_var: Variável do tempo
        show_steps: Se deve mostrar os passos
        
    Returns:
        Expressão no domínio do tempo
    """
    transformer = LaplaceTransformer()
    transformer.s = s_var
    transformer.t = t_var
    
    result = transformer.inverse_transform(expr, show_steps)
    
    if show_steps:
        print(transformer.history.get_formatted_steps())
    
    return result


# Funções de conveniência para transformadas comuns
def unit_step_laplace(s_var):
    """Transformada de Laplace da função degrau unitário"""
    return 1/s_var


def unit_impulse_laplace(s_var):
    """Transformada de Laplace da função impulso unitário"""
    return 1


def exponential_laplace(a, s_var):
    """Transformada de Laplace de e^(-at)"""
    return 1/(s_var + a)


def sinusoidal_laplace(omega, s_var):
    """Transformada de Laplace de sin(ωt)"""
    return omega/(s_var**2 + omega**2)


def cosinusoidal_laplace(omega, s_var):
    """Transformada de Laplace de cos(ωt)"""
    return s_var/(s_var**2 + omega**2)


def polynomial_laplace(n, s_var):
    """Transformada de Laplace de t^n"""
    return sp.factorial(n)/(s_var**(n+1))
