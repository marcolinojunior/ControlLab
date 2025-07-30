#!/usr/bin/env python3
"""
Módulo de Validações Pedagógicas - ControlLab
============================================

Este módulo implementa validações educativas para sistemas de controle,
incluindo verificações de estabilidade, causalidade, cancelamentos polo-zero,
e outras propriedades importantes para o aprendizado.

Funcionalidades:
- Verificação de cancelamentos polo-zero
- Verificação de causalidade
- Verificação de estabilidade BIBO
- Comparação com métodos alternativos
- Detecção de casos especiais
"""

import warnings
from typing import Dict, List, Any, Optional, Union, Tuple

try:
    import sympy as sp
    from sympy import symbols, solve, Poly, simplify, expand, factor
    from sympy import I, pi, oo, zoo, nan
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False


class ValidationResult:
    """Classe para armazenar resultados de validação"""
    
    def __init__(self):
        self.is_valid = True
        self.warnings = []
        self.errors = []
        self.info = []
        self.properties = {}
    
    def add_warning(self, message: str):
        """Adiciona um aviso"""
        self.warnings.append(message)
    
    def add_error(self, message: str):
        """Adiciona um erro"""
        self.errors.append(message)
        self.is_valid = False
    
    def add_info(self, message: str):
        """Adiciona informação"""
        self.info.append(message)
    
    def set_property(self, name: str, value: Any):
        """Define uma propriedade"""
        self.properties[name] = value
    
    def get_summary(self) -> str:
        """Obtém resumo da validação"""
        lines = []
        lines.append("=== RESULTADO DA VALIDAÇÃO ===")
        lines.append(f"Status: {'✅ VÁLIDO' if self.is_valid else '❌ INVÁLIDO'}")
        
        if self.errors:
            lines.append("\n🚨 ERROS:")
            for error in self.errors:
                lines.append(f"  - {error}")
        
        if self.warnings:
            lines.append("\n⚠️ AVISOS:")
            for warning in self.warnings:
                lines.append(f"  - {warning}")
        
        if self.info:
            lines.append("\n📋 INFORMAÇÕES:")
            for info in self.info:
                lines.append(f"  - {info}")
        
        if self.properties:
            lines.append("\n📊 PROPRIEDADES:")
            for prop, value in self.properties.items():
                lines.append(f"  - {prop}: {value}")
        
        return "\n".join(lines)


def check_pole_zero_cancellation(numerator, denominator, variable='s', tolerance=1e-10):
    """
    Verifica cancelamentos polo-zero em uma função de transferência
    
    Args:
        numerator: Numerador da função de transferência
        denominator: Denominador da função de transferência
        variable: Variável (default 's')
        tolerance: Tolerância para detecção de cancelamentos
    
    Returns:
        ValidationResult com informações sobre cancelamentos
    """
    if not SYMPY_AVAILABLE:
        result = ValidationResult()
        result.add_error("SymPy necessário para verificação de cancelamentos")
        return result
    
    result = ValidationResult()
    
    try:
        s = symbols(variable)
        
        # Fatorar numerador e denominador
        num_factored = factor(numerator)
        den_factored = factor(denominator)
        
        result.add_info(f"Numerador fatorado: {num_factored}")
        result.add_info(f"Denominador fatorado: {den_factored}")
        
        # Encontrar zeros e pólos
        zeros = solve(numerator, s)
        poles = solve(denominator, s)
        
        result.set_property("zeros", zeros)
        result.set_property("poles", poles)
        
        # Verificar cancelamentos
        cancellations = []
        for zero in zeros:
            for pole in poles:
                # Verificar se são próximos (cancelamento)
                try:
                    diff = abs(complex(zero - pole))
                    if diff < tolerance:
                        cancellations.append((zero, pole))
                        result.add_warning(f"Cancelamento detectado: zero em {zero} com pólo em {pole}")
                except:
                    # Para valores simbólicos, verificar igualdade exata
                    if simplify(zero - pole) == 0:
                        cancellations.append((zero, pole))
                        result.add_warning(f"Cancelamento exato: zero = pólo = {zero}")
        
        result.set_property("cancellations", cancellations)
        
        if cancellations:
            result.add_warning("Cancelamentos polo-zero podem indicar problemas de modelagem")
            result.add_info("Verifique se os cancelamentos são físicamente justificados")
        else:
            result.add_info("Nenhum cancelamento polo-zero detectado")
    
    except Exception as e:
        result.add_error(f"Erro na verificação de cancelamentos: {e}")
    
    return result


def check_causality(transfer_function, variable='s'):
    """
    Verifica se um sistema é causal
    
    Args:
        transfer_function: Função de transferência
        variable: Variável (default 's')
    
    Returns:
        ValidationResult com informações sobre causalidade
    """
    if not SYMPY_AVAILABLE:
        result = ValidationResult()
        result.add_error("SymPy necessário para verificação de causalidade")
        return result
    
    result = ValidationResult()
    
    try:
        s = symbols(variable)
        
        # Expandir em série de Laurent para verificar causalidade
        # Um sistema é causal se não há termos com potências positivas de s no denominador
        # quando expandido em s → ∞
        
        # Método simples: verificar se grau do numerador <= grau do denominador
        num, den = sp.fraction(transfer_function)
        
        num_poly = Poly(num, s)
        den_poly = Poly(den, s)
        
        num_degree = num_poly.degree()
        den_degree = den_poly.degree()
        
        result.set_property("numerator_degree", num_degree)
        result.set_property("denominator_degree", den_degree)
        
        if num_degree <= den_degree:
            result.add_info("✅ Sistema é próprio (causal)")
            result.set_property("causal", True)
        elif num_degree == den_degree + 1:
            result.add_warning("⚠️ Sistema é semi-próprio (contém impulso)")
            result.set_property("causal", True)
            result.set_property("has_impulse", True)
        else:
            result.add_error("❌ Sistema é impróprio (não-causal)")
            result.set_property("causal", False)
            result.add_info("Sistemas impróprios não são fisicamente realizáveis")
    
    except Exception as e:
        result.add_error(f"Erro na verificação de causalidade: {e}")
    
    return result


def check_bibo_stability(transfer_function, variable='s'):
    """
    Verifica estabilidade BIBO (Bounded Input Bounded Output)
    
    Args:
        transfer_function: Função de transferência
        variable: Variável (default 's')
    
    Returns:
        ValidationResult com informações sobre estabilidade
    """
    if not SYMPY_AVAILABLE:
        result = ValidationResult()
        result.add_error("SymPy necessário para verificação de estabilidade")
        return result
    
    result = ValidationResult()
    
    try:
        s = symbols(variable)
        
        # Obter denominador (pólos)
        num, den = sp.fraction(transfer_function)
        poles = solve(den, s)
        
        result.set_property("poles", poles)
        
        stable = True
        marginal_poles = []
        unstable_poles = []
        
        for pole in poles:
            try:
                # Avaliar parte real do pólo
                if pole.is_real:
                    if pole > 0:
                        unstable_poles.append(pole)
                        stable = False
                        result.add_error(f"Pólo instável no semi-plano direito: {pole}")
                    elif pole == 0:
                        marginal_poles.append(pole)
                        result.add_warning(f"Pólo marginal na origem: {pole}")
                    else:
                        result.add_info(f"Pólo estável: {pole}")
                else:
                    # Pólo complexo - verificar parte real
                    real_part = sp.re(pole)
                    if real_part > 0:
                        unstable_poles.append(pole)
                        stable = False
                        result.add_error(f"Pólo complexo instável: {pole}")
                    elif real_part == 0:
                        marginal_poles.append(pole)
                        result.add_warning(f"Pólo complexo marginal: {pole}")
                    else:
                        result.add_info(f"Pólo complexo estável: {pole}")
            
            except:
                # Se não conseguir determinar, assumir potencialmente instável
                result.add_warning(f"Não foi possível determinar estabilidade do pólo: {pole}")
        
        result.set_property("stable_poles", [p for p in poles if p not in unstable_poles and p not in marginal_poles])
        result.set_property("marginal_poles", marginal_poles)
        result.set_property("unstable_poles", unstable_poles)
        
        if stable and not marginal_poles:
            result.add_info("✅ Sistema é BIBO estável")
            result.set_property("bibo_stable", True)
        elif stable and marginal_poles:
            result.add_warning("⚠️ Sistema é marginalmente estável")
            result.set_property("bibo_stable", False)
            result.set_property("marginally_stable", True)
        else:
            result.add_error("❌ Sistema é BIBO instável")
            result.set_property("bibo_stable", False)
    
    except Exception as e:
        result.add_error(f"Erro na verificação de estabilidade: {e}")
    
    return result


def check_minimum_phase(transfer_function, variable='s'):
    """
    Verifica se o sistema é de fase mínima
    
    Args:
        transfer_function: Função de transferência
        variable: Variável (default 's')
    
    Returns:
        ValidationResult com informações sobre fase
    """
    if not SYMPY_AVAILABLE:
        result = ValidationResult()
        result.add_error("SymPy necessário para verificação de fase")
        return result
    
    result = ValidationResult()
    
    try:
        s = symbols(variable)
        
        # Obter numerador (zeros)
        num, den = sp.fraction(transfer_function)
        zeros = solve(num, s)
        
        result.set_property("zeros", zeros)
        
        rhp_zeros = []  # Right Half Plane zeros
        
        for zero in zeros:
            try:
                if zero.is_real:
                    if zero > 0:
                        rhp_zeros.append(zero)
                        result.add_warning(f"Zero no semi-plano direito: {zero}")
                else:
                    # Zero complexo
                    real_part = sp.re(zero)
                    if real_part > 0:
                        rhp_zeros.append(zero)
                        result.add_warning(f"Zero complexo no semi-plano direito: {zero}")
            except:
                result.add_warning(f"Não foi possível determinar localização do zero: {zero}")
        
        result.set_property("rhp_zeros", rhp_zeros)
        
        if not rhp_zeros:
            result.add_info("✅ Sistema é de fase mínima")
            result.set_property("minimum_phase", True)
        else:
            result.add_warning("⚠️ Sistema é de fase não-mínima")
            result.set_property("minimum_phase", False)
            result.add_info("Sistemas de fase não-mínima têm resposta temporal mais lenta")
    
    except Exception as e:
        result.add_error(f"Erro na verificação de fase: {e}")
    
    return result


def validate_system_properties(transfer_function, variable='s', include_all=True):
    """
    Executa todas as validações em um sistema
    
    Args:
        transfer_function: Função de transferência
        variable: Variável (default 's')
        include_all: Se deve incluir todas as verificações
    
    Returns:
        Dict com todos os resultados de validação
    """
    results = {}
    
    if include_all:
        print("🔍 Executando validações do sistema...")
        
        # Cancelamentos polo-zero
        print("  Verificando cancelamentos polo-zero...")
        results['pole_zero_cancellation'] = check_pole_zero_cancellation(
            *sp.fraction(transfer_function), variable
        )
        
        # Causalidade
        print("  Verificando causalidade...")
        results['causality'] = check_causality(transfer_function, variable)
        
        # Estabilidade BIBO
        print("  Verificando estabilidade BIBO...")
        results['bibo_stability'] = check_bibo_stability(transfer_function, variable)
        
        # Fase mínima
        print("  Verificando fase mínima...")
        results['minimum_phase'] = check_minimum_phase(transfer_function, variable)
        
        print("✅ Validações concluídas!")
    
    return results


def print_validation_summary(validation_results):
    """
    Imprime resumo de todas as validações
    
    Args:
        validation_results: Resultados das validações
    """
    print("\n" + "="*60)
    print("RESUMO DAS VALIDAÇÕES DO SISTEMA")
    print("="*60)
    
    for validation_name, result in validation_results.items():
        print(f"\n🔬 {validation_name.upper().replace('_', ' ')}:")
        print(result.get_summary())
    
    print("\n" + "="*60)


def compare_with_alternative_method(original_result, alternative_result, method_name: str):
    """
    Compara resultado com método alternativo
    
    Args:
        original_result: Resultado do método original
        alternative_result: Resultado do método alternativo
        method_name: Nome do método alternativo
    
    Returns:
        ValidationResult com comparação
    """
    result = ValidationResult()
    
    try:
        if SYMPY_AVAILABLE:
            difference = simplify(original_result - alternative_result)
            
            if difference == 0:
                result.add_info(f"✅ Métodos concordam: original ≡ {method_name}")
                result.set_property("methods_agree", True)
            else:
                result.add_warning(f"⚠️ Diferença entre métodos: {difference}")
                result.set_property("methods_agree", False)
                result.set_property("difference", difference)
        else:
            result.add_error("SymPy necessário para comparação")
    
    except Exception as e:
        result.add_error(f"Erro na comparação: {e}")
    
    return result


# Classes de fallback
class FallbackValidation:
    """Classe de fallback quando SymPy não está disponível"""
    
    def __init__(self):
        warnings.warn("Validações limitadas - instale SymPy para funcionalidade completa")
    
    def check_pole_zero_cancellation(self, *args, **kwargs):
        result = ValidationResult()
        result.add_error("SymPy necessário para validações")
        return result
    
    def check_causality(self, *args, **kwargs):
        result = ValidationResult()
        result.add_error("SymPy necessário para validações")
        return result
    
    def check_bibo_stability(self, *args, **kwargs):
        result = ValidationResult()
        result.add_error("SymPy necessário para validações")
        return result


# Instanciar fallback se necessário
if not SYMPY_AVAILABLE:
    _fallback = FallbackValidation()
    check_pole_zero_cancellation = _fallback.check_pole_zero_cancellation
    check_causality = _fallback.check_causality
    check_bibo_stability = _fallback.check_bibo_stability
