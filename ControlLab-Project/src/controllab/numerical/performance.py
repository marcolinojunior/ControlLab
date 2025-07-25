#!/usr/bin/env python3
"""
Análise de Desempenho - ControlLab Numerical
Especificações de desempenho temporal e steady-state error
"""

import sympy as sp
import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Any
import warnings

# Importações condicionais
try:
    import control
    CONTROL_AVAILABLE = True
except ImportError:
    CONTROL_AVAILABLE = False
    control = None

from ..core.symbolic_tf import SymbolicTransferFunction
from ..core.history import OperationHistory


class PerformanceAnalyzer:
    """
    Analisador de desempenho de sistemas de controle
    """
    
    def __init__(self):
        self.history = OperationHistory()
    
    def analyze_steady_state_error(self, tf_system, input_type='step', substitutions=None):
        """
        Calcula constantes de erro estático (Kp, Kv, Ka) e erro em regime permanente
        
        Args:
            tf_system: SymbolicTransferFunction
            input_type: Tipo de entrada ('step', 'ramp', 'parabolic')
            substitutions: Substituições de símbolos
            
        Returns:
            dict: Análise completa de erro steady-state
        """
        if not isinstance(tf_system, SymbolicTransferFunction):
            raise TypeError("tf_system deve ser SymbolicTransferFunction")
        
        self.history.add_step(
            "ANÁLISE_ERRO_STEADY_STATE",
            f"Analisando erro steady-state para entrada {input_type}",
            str(tf_system),
            "Calculando constantes de erro..."
        )
        
        try:
            # Determinar tipo do sistema (Tipo 0, 1, 2, etc.)
            system_type = self._determine_system_type(tf_system, substitutions)
            
            # Calcular constantes de erro
            Kp = self._calculate_position_constant(tf_system, substitutions)
            Kv = self._calculate_velocity_constant(tf_system, substitutions)
            Ka = self._calculate_acceleration_constant(tf_system, substitutions)
            
            # Calcular erro para o tipo de entrada
            error_steady_state = self._calculate_steady_state_error(
                input_type, system_type, Kp, Kv, Ka
            )
            
            result = {
                'system_type': system_type,
                'position_constant_Kp': Kp,
                'velocity_constant_Kv': Kv,
                'acceleration_constant_Ka': Ka,
                'steady_state_error': error_steady_state,
                'input_type': input_type,
                'error_coefficient': self._get_error_coefficient(input_type, system_type)
            }
            
            self.history.add_step(
                "ERRO_CALCULADO",
                f"Erro steady-state calculado: {error_steady_state}",
                f"Tipo {system_type}, Kp={Kp}, Kv={Kv}, Ka={Ka}",
                result
            )
            
            return result
            
        except Exception as e:
            error_msg = f"Erro na análise steady-state: {e}"
            self.history.add_step("ERRO_ANÁLISE", error_msg, "", str(e))
            raise ValueError(error_msg) from e
    
    def _determine_system_type(self, tf_system, substitutions=None):
        """Determina o tipo do sistema (número de polos na origem)"""
        s = tf_system.variable
        denominator = tf_system.denominator
        
        if substitutions:
            denominator = denominator.subs(substitutions)
        
        # Expandir denominador
        expanded_den = sp.expand(denominator)
        
        # Contar fatores s (polos na origem)
        type_count = 0
        
        # Método mais robusto: fatorar o denominador
        try:
            # Tentar fatorar
            factored = sp.factor(expanded_den)
            
            # Se é um produto, analisar cada fator
            if factored.is_Mul:
                for factor in factored.args:
                    if factor == s:
                        type_count += 1
                    elif factor.is_Pow and factor.base == s:
                        type_count += int(factor.exp)
            elif factored == s:
                type_count = 1
            elif factored.is_Pow and factored.base == s:
                type_count = int(factored.exp)
                
        except:
            # Método alternativo: substituição s→0
            temp_den = expanded_den
            while type_count < 10:  # Limite de segurança
                try:
                    # Calcular limite quando s→0
                    limit_val = sp.limit(temp_den, s, 0)
                    if limit_val == 0:
                        # Se é zero, dividir por s e continuar
                        temp_den = sp.simplify(temp_den / s)
                        type_count += 1
                    else:
                        # Se não é zero, parar
                        break
                except:
                    break
        
        return type_count
    
    def _calculate_position_constant(self, tf_system, substitutions=None):
        """Calcula constante de erro de posição Kp"""
        s = tf_system.variable
        tf_expr = tf_system.numerator / tf_system.denominator
        
        if substitutions:
            tf_expr = tf_expr.subs(substitutions)
        
        try:
            # Kp = lim(s→0) G(s)
            Kp = sp.limit(tf_expr, s, 0)
            return float(Kp) if Kp.is_real else complex(Kp)
        except:
            return float('inf')
    
    def _calculate_velocity_constant(self, tf_system, substitutions=None):
        """Calcula constante de erro de velocidade Kv"""
        s = tf_system.variable
        tf_expr = tf_system.numerator / tf_system.denominator
        
        if substitutions:
            tf_expr = tf_expr.subs(substitutions)
        
        try:
            # Kv = lim(s→0) s*G(s)
            Kv = sp.limit(s * tf_expr, s, 0)
            return float(Kv) if Kv.is_real else complex(Kv)
        except:
            return 0.0
    
    def _calculate_acceleration_constant(self, tf_system, substitutions=None):
        """Calcula constante de erro de aceleração Ka"""
        s = tf_system.variable
        tf_expr = tf_system.numerator / tf_system.denominator
        
        if substitutions:
            tf_expr = tf_expr.subs(substitutions)
        
        try:
            # Ka = lim(s→0) s²*G(s)
            Ka = sp.limit(s**2 * tf_expr, s, 0)
            return float(Ka) if Ka.is_real else complex(Ka)
        except:
            return 0.0
    
    def _calculate_steady_state_error(self, input_type, system_type, Kp, Kv, Ka):
        """Calcula erro steady-state baseado no tipo de sistema e entrada"""
        if input_type == 'step':
            if system_type >= 1:
                return 0.0
            else:
                return 1.0 / (1.0 + Kp) if Kp != float('inf') else 0.0
                
        elif input_type == 'ramp':
            if system_type >= 2:
                return 0.0
            elif system_type == 1:
                return 1.0 / Kv if Kv != 0 else float('inf')
            else:
                return float('inf')
                
        elif input_type == 'parabolic':
            if system_type >= 3:
                return 0.0
            elif system_type == 2:
                return 1.0 / Ka if Ka != 0 else float('inf')
            else:
                return float('inf')
        
        return None
    
    def _get_error_coefficient(self, input_type, system_type):
        """Retorna coeficiente de erro para o tipo de entrada e sistema"""
        error_table = {
            ('step', 0): '1/(1+Kp)',
            ('step', 1): '0',
            ('step', 2): '0',
            ('ramp', 0): '∞',
            ('ramp', 1): '1/Kv',
            ('ramp', 2): '0',
            ('parabolic', 0): '∞',
            ('parabolic', 1): '∞',
            ('parabolic', 2): '1/Ka'
        }
        
        key = (input_type, min(system_type, 2))
        return error_table.get(key, 'N/A')
    
    def analyze_time_response_specs(self, time_response, time_vector, input_type='step'):
        """
        Analisa especificações de resposta temporal
        
        Args:
            time_response: Array da resposta temporal
            time_vector: Array do tempo
            input_type: Tipo de entrada
            
        Returns:
            dict: Especificações de desempenho
        """
        try:
            if not hasattr(time_response, '__len__') or not hasattr(time_vector, '__len__'):
                # Converter para numpy arrays se necessário
                time_response = np.array(time_response)
                time_vector = np.array(time_vector)
            
            # Valor final
            final_value = time_response[-1]
            
            # Overshoot
            max_value = np.max(time_response)
            overshoot_percent = ((max_value - final_value) / final_value) * 100 if final_value != 0 else 0
            
            # Rise time (10% a 90% do valor final)
            rise_time = self._calculate_rise_time(time_response, time_vector, final_value)
            
            # Settling time (2% ou 5% do valor final)
            settling_time_2 = self._calculate_settling_time(time_response, time_vector, final_value, 0.02)
            settling_time_5 = self._calculate_settling_time(time_response, time_vector, final_value, 0.05)
            
            # Peak time
            peak_time = time_vector[np.argmax(time_response)]
            
            # Delay time (50% do valor final)
            delay_time = self._calculate_delay_time(time_response, time_vector, final_value)
            
            specs = {
                'final_value': final_value,
                'max_value': max_value,
                'overshoot_percent': overshoot_percent,
                'rise_time': rise_time,
                'settling_time_2_percent': settling_time_2,
                'settling_time_5_percent': settling_time_5,
                'peak_time': peak_time,
                'delay_time': delay_time,
                'input_type': input_type
            }
            
            self.history.add_step(
                "SPECS_TEMPORAIS",
                f"Especificações calculadas para {input_type}",
                f"Overshoot: {overshoot_percent:.1f}%, Settling: {settling_time_2:.3f}s",
                specs
            )
            
            return specs
            
        except Exception as e:
            error_msg = f"Erro no cálculo de especificações: {e}"
            self.history.add_step("ERRO_SPECS", error_msg, "", str(e))
            raise ValueError(error_msg) from e
    
    def _calculate_rise_time(self, response, time, final_value):
        """Calcula rise time (10% a 90%)"""
        try:
            val_10 = 0.1 * final_value
            val_90 = 0.9 * final_value
            
            # Encontrar índices onde a resposta cruza 10% e 90%
            idx_10 = np.where(response >= val_10)[0]
            idx_90 = np.where(response >= val_90)[0]
            
            if len(idx_10) > 0 and len(idx_90) > 0:
                t_10 = time[idx_10[0]]
                t_90 = time[idx_90[0]]
                return t_90 - t_10
            return None
        except:
            return None
    
    def _calculate_settling_time(self, response, time, final_value, tolerance):
        """Calcula settling time para tolerância dada"""
        try:
            upper_bound = final_value * (1 + tolerance)
            lower_bound = final_value * (1 - tolerance)
            
            # Procurar o último ponto que sai da banda de tolerância
            outside_band = (response > upper_bound) | (response < lower_bound)
            
            if np.any(outside_band):
                last_outside_idx = np.where(outside_band)[0][-1]
                return time[last_outside_idx]
            return time[0]  # Se sempre dentro da banda
        except:
            return None
    
    def _calculate_delay_time(self, response, time, final_value):
        """Calcula delay time (50% do valor final)"""
        try:
            val_50 = 0.5 * final_value
            idx_50 = np.where(response >= val_50)[0]
            
            if len(idx_50) > 0:
                return time[idx_50[0]]
            return None
        except:
            return None
    
    def analyze_second_order_parameters(self, tf_system, substitutions=None):
        """
        Analisa parâmetros de sistema de segunda ordem (ωn, ζ)
        
        Args:
            tf_system: SymbolicTransferFunction
            substitutions: Substituições de símbolos
            
        Returns:
            dict: Parâmetros do sistema de segunda ordem
        """
        if not isinstance(tf_system, SymbolicTransferFunction):
            raise TypeError("tf_system deve ser SymbolicTransferFunction")
        
        try:
            s = tf_system.variable
            denominator = tf_system.denominator
            
            if substitutions:
                denominator = denominator.subs(substitutions)
            
            # Expandir denominador
            expanded_den = sp.expand(denominator)
            
            # Extrair coeficientes do polinômio s² + 2ζωn*s + ωn²
            coeffs = sp.Poly(expanded_den, s).all_coeffs()
            
            if len(coeffs) == 3:  # Sistema de segunda ordem
                a2, a1, a0 = [float(c) for c in coeffs]
                
                # ωn² = a0/a2, 2ζωn = a1/a2
                wn_squared = a0 / a2
                wn = np.sqrt(abs(wn_squared))
                
                if wn > 0:
                    zeta = (a1 / a2) / (2 * wn)
                else:
                    zeta = 0
                
                # Classificar tipo de resposta
                if zeta < 0:
                    response_type = "Instável"
                elif zeta == 0:
                    response_type = "Não amortecido"
                elif 0 < zeta < 1:
                    response_type = "Subamortecido"
                elif zeta == 1:
                    response_type = "Criticamente amortecido"
                else:
                    response_type = "Superamortecido"
                
                # Calcular frequências características
                wd = wn * np.sqrt(1 - zeta**2) if zeta < 1 else 0  # Frequência amortecida
                
                # Calcular especificações teóricas para sistema subamortecido
                if 0 < zeta < 1:
                    overshoot_theory = 100 * np.exp(-zeta * np.pi / np.sqrt(1 - zeta**2))
                    settling_time_theory = 4 / (zeta * wn)  # Critério 2%
                    peak_time_theory = np.pi / wd
                    rise_time_theory = (np.pi - np.arctan(np.sqrt(1 - zeta**2) / zeta)) / wd
                else:
                    overshoot_theory = 0
                    settling_time_theory = None
                    peak_time_theory = None
                    rise_time_theory = None
                
                result = {
                    'order': 2,
                    'natural_frequency_wn': wn,
                    'damping_ratio_zeta': zeta,
                    'damped_frequency_wd': wd,
                    'response_type': response_type,
                    'theoretical_overshoot': overshoot_theory,
                    'theoretical_settling_time': settling_time_theory,
                    'theoretical_peak_time': peak_time_theory,
                    'theoretical_rise_time': rise_time_theory,
                    'poles': [-zeta * wn + 1j * wd, -zeta * wn - 1j * wd] if zeta < 1 else [-wn, -wn]
                }
                
                self.history.add_step(
                    "ANÁLISE_2ª_ORDEM",
                    f"Sistema analisado: {response_type}",
                    f"ωn={wn:.3f}, ζ={zeta:.3f}, overshoot={overshoot_theory:.1f}%",
                    result
                )
                
                return result
            else:
                return {
                    'order': len(coeffs) - 1,
                    'message': f"Sistema de {len(coeffs)-1}ª ordem - análise de 2ª ordem não aplicável"
                }
                
        except Exception as e:
            error_msg = f"Erro na análise de 2ª ordem: {e}"
            self.history.add_step("ERRO_2ª_ORDEM", error_msg, "", str(e))
            raise ValueError(error_msg) from e
    
    def get_performance_summary(self, tf_system, substitutions=None, input_type='step'):
        """
        Gera resumo completo de desempenho do sistema
        
        Args:
            tf_system: SymbolicTransferFunction
            substitutions: Substituições de símbolos
            input_type: Tipo de entrada para análise
            
        Returns:
            dict: Resumo completo de desempenho
        """
        summary = {
            'system': str(tf_system),
            'substitutions': substitutions,
            'analysis_date': 'Real-time analysis'
        }
        
        try:
            # Análise de erro steady-state
            steady_state = self.analyze_steady_state_error(tf_system, input_type, substitutions)
            summary['steady_state_analysis'] = steady_state
            
            # Análise de 2ª ordem se aplicável
            second_order = self.analyze_second_order_parameters(tf_system, substitutions)
            summary['second_order_analysis'] = second_order
            
            # Adicionar recomendações
            summary['recommendations'] = self._generate_recommendations(steady_state, second_order)
            
            self.history.add_step(
                "RESUMO_DESEMPENHO",
                "Análise completa de desempenho gerada",
                f"Tipo {steady_state['system_type']}, Erro: {steady_state['steady_state_error']}",
                summary
            )
            
            return summary
            
        except Exception as e:
            error_msg = f"Erro no resumo de desempenho: {e}"
            self.history.add_step("ERRO_RESUMO", error_msg, "", str(e))
            summary['error'] = error_msg
            return summary
    
    def _generate_recommendations(self, steady_state, second_order):
        """Gera recomendações baseadas na análise"""
        recommendations = []
        
        # Recomendações de erro steady-state
        if steady_state['steady_state_error'] == float('inf'):
            recommendations.append("⚠️ Erro infinito - considere aumentar o tipo do sistema")
        elif steady_state['steady_state_error'] > 0.1:
            recommendations.append("⚠️ Erro elevado - considere aumentar o ganho ou tipo do sistema")
        
        # Recomendações de 2ª ordem
        if second_order.get('order') == 2:
            zeta = second_order.get('damping_ratio_zeta', 0)
            
            if zeta < 0.4:
                recommendations.append("⚠️ Sistema subamortecido - overshoot elevado")
            elif zeta > 1.2:
                recommendations.append("⚠️ Sistema superamortecido - resposta lenta")
            elif 0.6 <= zeta <= 0.8:
                recommendations.append("✅ Amortecimento adequado para controle")
        
        return recommendations if recommendations else ["✅ Sistema com características adequadas"]
