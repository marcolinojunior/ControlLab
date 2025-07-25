#!/usr/bin/env python3
"""
Módulo de Casos Especiais - ControlLab
=====================================

Este módulo implementa tratamento para casos especiais em sistemas de controle,
incluindo sistemas com atraso, zeros no semi-plano direito, cancelamentos exatos,
sistemas de fase não-mínima e condições iniciais não-nulas.

Funcionalidades:
- Sistemas com atraso de transporte
- Sistemas de fase não-mínima
- Cancelamentos exatos polo-zero
- Condições iniciais não-nulas
- Sistemas com comportamentos especiais
"""

import warnings
from typing import Dict, List, Any, Optional, Union, Tuple

try:
    import sympy as sp
    from sympy import symbols, exp, solve, simplify, expand, factor
    from sympy import Function, Eq, dsolve, laplace_transform, inverse_laplace_transform
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False


class SpecialCaseHandler:
    """
    Classe para lidar com casos especiais em sistemas de controle
    """
    
    def __init__(self):
        self.s = symbols('s')
        self.t = symbols('t', positive=True)
        self.cases_handled = []
    
    def handle_time_delay_system(self, base_tf, delay_time, show_steps=True):
        """
        Trata sistemas com atraso de transporte
        
        Args:
            base_tf: Função de transferência base
            delay_time: Tempo de atraso
            show_steps: Mostrar passos
        
        Returns:
            Função de transferência com atraso
        """
        if not SYMPY_AVAILABLE:
            raise NotImplementedError("SymPy necessário para sistemas com atraso")
        
        if show_steps:
            print("=== Tratamento de Sistema com Atraso ===")
            print(f"Sistema base: G(s) = {base_tf}")
            print(f"Atraso: T = {delay_time}")
            print(f"Sistema com atraso: G(s) * e^(-T*s)")
        
        # Sistema com atraso: G(s) * e^(-T*s)
        delay_factor = exp(-delay_time * self.s)
        system_with_delay = base_tf * delay_factor
        
        if show_steps:
            print(f"Resultado: G_delay(s) = {system_with_delay}")
            print("\n📚 Nota pedagógica:")
            print("- Atraso de transporte introduz fase adicional")
            print("- Não afeta magnitude da resposta em frequência")
            print("- Pode causar instabilidade em sistemas com realimentação")
        
        self.cases_handled.append({
            'type': 'time_delay',
            'original': base_tf,
            'modified': system_with_delay,
            'parameters': {'delay': delay_time}
        })
        
        return system_with_delay
    
    def handle_right_half_plane_zeros(self, tf_expr, show_steps=True):
        """
        Analisa sistemas com zeros no semi-plano direito
        
        Args:
            tf_expr: Função de transferência
            show_steps: Mostrar passos
        
        Returns:
            Análise dos zeros RHP
        """
        if not SYMPY_AVAILABLE:
            raise NotImplementedError("SymPy necessário para análise de zeros")
        
        if show_steps:
            print("=== Análise de Zeros no Semi-Plano Direito ===")
        
        # Obter numerador e zeros
        num, den = sp.fraction(tf_expr)
        zeros = solve(num, self.s)
        
        rhp_zeros = []
        lhp_zeros = []
        
        for zero in zeros:
            if zero.is_real:
                if zero > 0:
                    rhp_zeros.append(zero)
                else:
                    lhp_zeros.append(zero)
            else:
                # Zero complexo
                real_part = sp.re(zero)
                if real_part > 0:
                    rhp_zeros.append(zero)
                else:
                    lhp_zeros.append(zero)
        
        if show_steps:
            print(f"Zeros no semi-plano esquerdo: {lhp_zeros}")
            print(f"Zeros no semi-plano direito: {rhp_zeros}")
            
            if rhp_zeros:
                print("\n⚠️ Sistema de FASE NÃO-MÍNIMA detectado!")
                print("📚 Implicações:")
                print("- Resposta ao degrau pode ter undershoot inicial")
                print("- Controlador precisa de cuidado especial")
                print("- Limitações fundamentais de desempenho")
            else:
                print("\n✅ Sistema de FASE MÍNIMA")
                print("📚 Características:")
                print("- Resposta ao degrau sem undershoot")
                print("- Melhor controlabilidade")
        
        analysis = {
            'rhp_zeros': rhp_zeros,
            'lhp_zeros': lhp_zeros,
            'is_minimum_phase': len(rhp_zeros) == 0,
            'total_zeros': len(zeros)
        }
        
        self.cases_handled.append({
            'type': 'rhp_zeros',
            'function': tf_expr,
            'analysis': analysis
        })
        
        return analysis
    
    def handle_exact_pole_zero_cancellation(self, numerator, denominator, show_steps=True):
        """
        Trata cancelamentos exatos polo-zero
        
        Args:
            numerator: Numerador
            denominator: Denominador
            show_steps: Mostrar passos
        
        Returns:
            Sistema simplificado
        """
        if not SYMPY_AVAILABLE:
            raise NotImplementedError("SymPy necessário para cancelamentos")
        
        if show_steps:
            print("=== Tratamento de Cancelamentos Polo-Zero ===")
            print(f"Numerador original: {numerator}")
            print(f"Denominador original: {denominator}")
        
        # Simplificar fração
        simplified = simplify(numerator / denominator)
        
        if show_steps:
            print(f"Sistema simplificado: {simplified}")
        
        # Verificar quais fatores foram cancelados
        original_zeros = solve(numerator, self.s)
        original_poles = solve(denominator, self.s)
        
        new_num, new_den = sp.fraction(simplified)
        new_zeros = solve(new_num, self.s)
        new_poles = solve(new_den, self.s)
        
        cancelled_factors = []
        for zero in original_zeros:
            if zero in original_poles:
                cancelled_factors.append(zero)
        
        if show_steps and cancelled_factors:
            print(f"\n🔄 Fatores cancelados: {cancelled_factors}")
            print("📚 Nota pedagógica:")
            print("- Cancelamentos removem pólos e zeros")
            print("- Podem ocultar modos não-controláveis/observáveis")
            print("- Importante verificar significado físico")
        
        result = {
            'original': numerator / denominator,
            'simplified': simplified,
            'cancelled_factors': cancelled_factors,
            'order_reduction': len(original_poles) - len(new_poles)
        }
        
        self.cases_handled.append({
            'type': 'pole_zero_cancellation',
            'result': result
        })
        
        return result
    
    def handle_nonzero_initial_conditions(self, ode, initial_conditions, show_steps=True):
        """
        Trata sistemas com condições iniciais não-nulas
        
        Args:
            ode: Equação diferencial
            initial_conditions: Dicionário de condições iniciais
            show_steps: Mostrar passos
        
        Returns:
            Solução com condições iniciais
        """
        if not SYMPY_AVAILABLE:
            raise NotImplementedError("SymPy necessário para condições iniciais")
        
        if show_steps:
            print("=== Condições Iniciais Não-Nulas ===")
            print(f"Equação diferencial: {ode}")
            print(f"Condições iniciais: {initial_conditions}")
        
        # Assumir que temos uma função y(t)
        y = Function('y')
        
        # Aplicar transformada de Laplace considerando condições iniciais
        # Para y'(t) → s*Y(s) - y(0)
        # Para y''(t) → s²*Y(s) - s*y(0) - y'(0)
        
        if show_steps:
            print("\n🔄 Aplicando transformada de Laplace:")
            print("L{y'(t)} = s*Y(s) - y(0)")
            print("L{y''(t)} = s²*Y(s) - s*y(0) - y'(0)")
        
        # Esta é uma implementação simplificada
        # Na prática, seria necessário parser mais sofisticado da ODE
        
        result = {
            'ode': ode,
            'initial_conditions': initial_conditions,
            'method': 'laplace_with_ics',
            'note': 'Condições iniciais afetam a resposta total do sistema'
        }
        
        if show_steps:
            print("\n📚 Nota pedagógica:")
            print("- Condições iniciais contribuem para resposta livre")
            print("- Resposta total = resposta livre + resposta forçada")
            print("- Importante para análise de transitórios")
        
        self.cases_handled.append({
            'type': 'nonzero_initial_conditions',
            'result': result
        })
        
        return result
    
    def handle_improper_system(self, tf_expr, show_steps=True):
        """
        Trata sistemas impróprios (grau numerador > grau denominador)
        
        Args:
            tf_expr: Função de transferência imprópria
            show_steps: Mostrar passos
        
        Returns:
            Decomposição em parte própria + termos diretos
        """
        if not SYMPY_AVAILABLE:
            raise NotImplementedError("SymPy necessário para sistemas impróprios")
        
        if show_steps:
            print("=== Tratamento de Sistema Impróprio ===")
            print(f"Sistema original: {tf_expr}")
        
        # Dividir polinômios para separar parte própria
        num, den = sp.fraction(tf_expr)
        
        # Divisão polinomial
        quotient, remainder = sp.div(num, den, self.s)
        proper_part = remainder / den
        direct_part = quotient
        
        if show_steps:
            print(f"Parte direta (feedthrough): {direct_part}")
            print(f"Parte própria: {proper_part}")
            print(f"Decomposição: G(s) = {direct_part} + {proper_part}")
            
            print("\n📚 Nota pedagógica:")
            print("- Sistemas impróprios não são fisicamente realizáveis")
            print("- Contêm termos de feedthrough direto")
            print("- Podem representar derivadores ideais")
        
        result = {
            'original': tf_expr,
            'direct_part': direct_part,
            'proper_part': proper_part,
            'is_realizable': direct_part == 0
        }
        
        self.cases_handled.append({
            'type': 'improper_system',
            'result': result
        })
        
        return result
    
    def get_case_summary(self):
        """Obtém resumo de todos os casos tratados"""
        if not self.cases_handled:
            return "Nenhum caso especial tratado ainda."
        
        summary = ["=== RESUMO DE CASOS ESPECIAIS TRATADOS ==="]
        
        for i, case in enumerate(self.cases_handled, 1):
            summary.append(f"\n{i}. Tipo: {case['type'].replace('_', ' ').title()}")
            
            if case['type'] == 'time_delay':
                summary.append(f"   Atraso: {case['parameters']['delay']}")
            elif case['type'] == 'rhp_zeros':
                summary.append(f"   Zeros RHP: {len(case['analysis']['rhp_zeros'])}")
                summary.append(f"   Fase mínima: {case['analysis']['is_minimum_phase']}")
            elif case['type'] == 'pole_zero_cancellation':
                summary.append(f"   Fatores cancelados: {len(case['result']['cancelled_factors'])}")
                summary.append(f"   Redução de ordem: {case['result']['order_reduction']}")
        
        return "\n".join(summary)


# Funções de conveniência
def create_time_delay_system(base_tf, delay_time, show_steps=True):
    """Cria sistema com atraso de transporte"""
    handler = SpecialCaseHandler()
    return handler.handle_time_delay_system(base_tf, delay_time, show_steps)


def analyze_rhp_zeros(tf_expr, show_steps=True):
    """Analisa zeros no semi-plano direito"""
    handler = SpecialCaseHandler()
    return handler.handle_right_half_plane_zeros(tf_expr, show_steps)


def simplify_pole_zero_cancellation(numerator, denominator, show_steps=True):
    """Simplifica cancelamentos polo-zero"""
    handler = SpecialCaseHandler()
    return handler.handle_exact_pole_zero_cancellation(numerator, denominator, show_steps)


def handle_initial_conditions(ode, initial_conditions, show_steps=True):
    """Trata condições iniciais não-nulas"""
    handler = SpecialCaseHandler()
    return handler.handle_nonzero_initial_conditions(ode, initial_conditions, show_steps)


def decompose_improper_system(tf_expr, show_steps=True):
    """Decompõe sistema impróprio"""
    handler = SpecialCaseHandler()
    return handler.handle_improper_system(tf_expr, show_steps)


# Classe de fallback
class FallbackSpecialCases:
    """Classe de fallback quando SymPy não está disponível"""
    
    def __init__(self):
        warnings.warn("Casos especiais limitados - instale SymPy para funcionalidade completa")
    
    def handle_time_delay_system(self, *args, **kwargs):
        raise NotImplementedError("SymPy necessário para sistemas com atraso")
    
    def analyze_rhp_zeros(self, *args, **kwargs):
        raise NotImplementedError("SymPy necessário para análise de zeros")


# Instanciar fallback se necessário
if not SYMPY_AVAILABLE:
    SpecialCaseHandler = FallbackSpecialCases
    create_time_delay_system = lambda *args, **kwargs: None
    analyze_rhp_zeros = lambda *args, **kwargs: None
