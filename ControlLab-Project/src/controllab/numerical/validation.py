#!/usr/bin/env python3
"""
Validador Numérico - ControlLab Numerical
Valida conversões e garante consistência numérica
"""

import sympy as sp
import warnings
from typing import Dict, List, Any, Union, Optional

# Importações condicionais
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

from ..core.history import OperationHistory


class NumericalValidator:
    """
    Valida conversões simbólico-numéricas e garante consistência
    """
    
    def __init__(self, tolerance: float = 1e-12):
        self.tolerance = tolerance
        self.history = OperationHistory()
    
    def validate_numerical_substitutions(self, expression: sp.Expr, 
                                       substitutions: Dict[sp.Symbol, Any]) -> Dict[str, Any]:
        """
        Valida substituições numéricas para uma expressão
        
        Args:
            expression: Expressão simbólica
            substitutions: Dicionário de substituições
            
        Returns:
            Dict: Resultado da validação
        """
        self.history.add_step(
            "VALIDAÇÃO_SUBSTITUIÇÕES",
            "Validando substituições numéricas",
            str(expression),
            f"Substituições: {substitutions}"
        )
        
        try:
            # Verifica símbolos livres
            free_symbols = expression.free_symbols
            provided_symbols = set(substitutions.keys())
            
            # Símbolos não substituídos
            missing_symbols = free_symbols - provided_symbols
            extra_symbols = provided_symbols - free_symbols
            
            # Valida valores numéricos
            invalid_values = {}
            for symbol, value in substitutions.items():
                if not self._is_numerical(value):
                    invalid_values[symbol] = value
            
            # Testa avaliação
            evaluation_error = None
            try:
                result = expression.subs(substitutions)
                numeric_result = complex(result)
            except Exception as e:
                evaluation_error = str(e)
                numeric_result = None
            
            validation_result = {
                'valid': len(missing_symbols) == 0 and len(invalid_values) == 0 and evaluation_error is None,
                'missing_symbols': list(missing_symbols),
                'extra_symbols': list(extra_symbols),
                'invalid_values': invalid_values,
                'evaluation_error': evaluation_error,
                'numeric_result': numeric_result
            }
            
            self.history.add_step(
                "SUBSTITUIÇÕES_VALIDADAS",
                f"Validação {'bem-sucedida' if validation_result['valid'] else 'com problemas'}",
                f"Símbolos faltando: {validation_result['missing_symbols']}",
                f"Valores inválidos: {validation_result['invalid_values']}"
            )
            
            return validation_result
            
        except Exception as e:
            self.history.add_step(
                "ERRO_VALIDAÇÃO",
                f"Erro na validação: {str(e)}",
                str(expression),
                None
            )
            raise
    
    def ensure_numerical_values(self, substitutions: Dict[sp.Symbol, Any]) -> Dict[sp.Symbol, complex]:
        """
        Garante que todos os valores são numéricos
        
        Args:
            substitutions: Dicionário de substituições
            
        Returns:
            Dict: Substituições com valores numéricos garantidos
        """
        numerical_subs = {}
        
        for symbol, value in substitutions.items():
            try:
                if isinstance(value, (int, float, complex)):
                    numerical_subs[symbol] = complex(value)
                elif hasattr(value, 'evalf'):  # SymPy expression
                    numerical_subs[symbol] = complex(value.evalf())
                else:
                    # Tentar conversão direta
                    numerical_subs[symbol] = complex(value)
            except (ValueError, TypeError) as e:
                warnings.warn(f"Não foi possível converter {symbol}={value} para numérico: {e}")
                numerical_subs[symbol] = value  # Manter valor original
        
        return numerical_subs
    
    def check_stability_numerical(self, poles: List[complex]) -> Dict[str, Any]:
        """
        Verifica estabilidade baseada em polos numéricos
        
        Args:
            poles: Lista de polos (números complexos)
            
        Returns:
            Dict: Análise de estabilidade
        """
        self.history.add_step(
            "ANÁLISE_ESTABILIDADE_NUMÉRICA",
            "Verificando estabilidade através dos polos",
            f"Polos: {poles}",
            "Analisando partes reais..."
        )
        
        try:
            stable_poles = []
            unstable_poles = []
            marginal_poles = []
            
            for pole in poles:
                pole_complex = complex(pole)
                real_part = pole_complex.real
                
                if real_part < -self.tolerance:
                    stable_poles.append(pole_complex)
                elif real_part > self.tolerance:
                    unstable_poles.append(pole_complex)
                else:
                    marginal_poles.append(pole_complex)
            
            is_stable = len(unstable_poles) == 0 and len(marginal_poles) == 0
            is_marginally_stable = len(unstable_poles) == 0 and len(marginal_poles) > 0
            
            result = {
                'is_stable': is_stable,
                'is_marginally_stable': is_marginally_stable,
                'stable_poles': stable_poles,
                'unstable_poles': unstable_poles,
                'marginal_poles': marginal_poles,
                'dominant_pole': self._find_dominant_pole(stable_poles) if stable_poles else None
            }
            
            self.history.add_step(
                "ESTABILIDADE_ANALISADA",
                f"Sistema {'estável' if is_stable else 'instável'}",
                f"Polos instáveis: {len(unstable_poles)}, marginais: {len(marginal_poles)}",
                f"Polos estáveis: {len(stable_poles)}"
            )
            
            return result
            
        except Exception as e:
            self.history.add_step(
                "ERRO_ANÁLISE_ESTABILIDADE",
                f"Erro na análise: {str(e)}",
                f"Polos: {poles}",
                None
            )
            raise
    
    def validate_system_equivalence(self, symbolic_result: Any, 
                                   numerical_result: Any,
                                   test_points: Optional[List] = None) -> Dict[str, Any]:
        """
        Valida equivalência entre resultados simbólicos e numéricos
        
        Args:
            symbolic_result: Resultado simbólico
            numerical_result: Resultado numérico
            test_points: Pontos de teste para comparação
            
        Returns:
            Dict: Resultado da validação de equivalência
        """
        if test_points is None:
            test_points = [0, 1, -1, 1j, -1j]
        
        self.history.add_step(
            "VALIDAÇÃO_EQUIVALÊNCIA",
            "Validando equivalência simbólico-numérico",
            f"Simbólico: {symbolic_result}",
            f"Numérico: {numerical_result}"
        )
        
        try:
            differences = []
            max_error = 0
            
            for point in test_points:
                try:
                    # Avaliar resultado simbólico
                    if hasattr(symbolic_result, 'subs'):
                        sym_value = complex(symbolic_result.subs('s', point))
                    else:
                        sym_value = complex(symbolic_result)
                    
                    # Avaliar resultado numérico
                    if hasattr(numerical_result, '__call__'):
                        num_value = complex(numerical_result(point))
                    else:
                        num_value = complex(numerical_result)
                    
                    # Calcular diferença
                    diff = abs(sym_value - num_value)
                    differences.append({
                        'point': point,
                        'symbolic': sym_value,
                        'numerical': num_value,
                        'difference': diff
                    })
                    
                    max_error = max(max_error, diff)
                    
                except Exception as e:
                    differences.append({
                        'point': point,
                        'error': str(e)
                    })
            
            is_equivalent = max_error < self.tolerance
            
            result = {
                'is_equivalent': is_equivalent,
                'max_error': max_error,
                'tolerance': self.tolerance,
                'differences': differences,
                'test_points': test_points
            }
            
            self.history.add_step(
                "EQUIVALÊNCIA_VALIDADA",
                f"Equivalência {'confirmada' if is_equivalent else 'FALHOU'}",
                f"Erro máximo: {max_error:.2e}",
                f"Tolerância: {self.tolerance:.2e}"
            )
            
            return result
            
        except Exception as e:
            self.history.add_step(
                "ERRO_VALIDAÇÃO_EQUIVALÊNCIA",
                f"Erro na validação: {str(e)}",
                f"Simbólico: {symbolic_result}",
                f"Numérico: {numerical_result}"
            )
            raise
    
    def _is_numerical(self, value: Any) -> bool:
        """Verifica se um valor é numérico"""
        try:
            complex(value)
            return True
        except (ValueError, TypeError):
            return False
    
    def _find_dominant_pole(self, poles: List[complex]) -> Optional[complex]:
        """Encontra o polo dominante (com menor parte real em módulo)"""
        if not poles:
            return None
        
        return min(poles, key=lambda p: abs(p.real))


# Funções de conveniência
def validate_substitutions(expression: sp.Expr, substitutions: Dict) -> Dict[str, Any]:
    """Função de conveniência para validar substituições"""
    validator = NumericalValidator()
    return validator.validate_numerical_substitutions(expression, substitutions)


def check_stability(poles: List[complex]) -> Dict[str, Any]:
    """Função de conveniência para verificar estabilidade"""
    validator = NumericalValidator()
    return validator.check_stability_numerical(poles)
