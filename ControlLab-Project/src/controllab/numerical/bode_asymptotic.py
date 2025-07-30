#!/usr/bin/env python3
"""
Aproximações Assintóticas para Bode - ControlLab Numerical
Geração automática de aproximações assintóticas para diagramas de Bode
"""

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
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


class BodeAsymptoticAnalyzer:
    """
    Analisador de aproximações assintóticas para diagramas de Bode
    """
    
    def __init__(self):
        self.history = OperationHistory()
    
    def analyze_tf_factors(self, tf_system, substitutions=None):
        """
        Analisa fatores da função de transferência para construir Bode
        
        Args:
            tf_system: SymbolicTransferFunction
            substitutions: Substituições de símbolos
            
        Returns:
            dict: Análise dos fatores da TF
        """
        if not isinstance(tf_system, SymbolicTransferFunction):
            raise TypeError("tf_system deve ser SymbolicTransferFunction")
        
        self.history.add_step(
            "ANÁLISE_FATORES_TF",
            "Analisando fatores para aproximação assintótica",
            str(tf_system),
            "Extraindo polos e zeros..."
        )
        
        try:
            s = tf_system.variable
            numerator = tf_system.numerator
            denominator = tf_system.denominator
            
            if substitutions:
                numerator = numerator.subs(substitutions)
                denominator = denominator.subs(substitutions)
            
            # Calcular ganho estático K
            s_zero = 0
            K = float((numerator / denominator).subs(s, s_zero)) if denominator.subs(s, s_zero) != 0 else 1
            
            # Analisar polos e zeros
            zeros_analysis = self._analyze_poles_zeros(numerator, s, "zeros")
            poles_analysis = self._analyze_poles_zeros(denominator, s, "poles")
            
            # Determinar tipo do sistema (integrador/derivador)
            system_type = self._determine_system_type_from_factors(denominator, s)
            
            # Construir fatores Bode
            bode_factors = []
            
            # Adicionar ganho constante
            if K != 0:
                bode_factors.append({
                    'type': 'constant',
                    'K': K,
                    'magnitude_db': 20 * np.log10(abs(K)),
                    'phase_deg': 180 if K < 0 else 0
                })
            
            # Adicionar fatores integradores/derivadores
            if system_type['integrator_order'] > 0:
                bode_factors.append({
                    'type': 'integrator',
                    'order': system_type['integrator_order'],
                    'slope_db_decade': -20 * system_type['integrator_order'],
                    'phase_deg': -90 * system_type['integrator_order']
                })
            elif system_type['differentiator_order'] > 0:
                bode_factors.append({
                    'type': 'differentiator',
                    'order': system_type['differentiator_order'],
                    'slope_db_decade': 20 * system_type['differentiator_order'],
                    'phase_deg': 90 * system_type['differentiator_order']
                })
            
            # Adicionar zeros
            for zero_info in zeros_analysis:
                bode_factors.append(self._create_bode_factor_from_root(zero_info, 'zero'))
            
            # Adicionar polos (excluindo polos na origem já tratados)
            for pole_info in poles_analysis:
                if abs(pole_info['root']) > 1e-10:  # Não incluir polos em s=0
                    bode_factors.append(self._create_bode_factor_from_root(pole_info, 'pole'))
            
            result = {
                'original_tf': tf_system,
                'static_gain': K,
                'system_type': system_type,
                'zeros_analysis': zeros_analysis,
                'poles_analysis': poles_analysis,
                'bode_factors': bode_factors,
                'total_factors': len(bode_factors)
            }
            
            self.history.add_step(
                "FATORES_ANALISADOS",
                f"Identificados {len(bode_factors)} fatores Bode",
                f"K={K}, Tipo={system_type['integrator_order']}, Polos={len(poles_analysis)}, Zeros={len(zeros_analysis)}",
                result
            )
            
            return result
            
        except Exception as e:
            error_msg = f"Erro na análise de fatores: {e}"
            self.history.add_step("ERRO_FATORES", error_msg, "", str(e))
            raise ValueError(error_msg) from e
    
    def _analyze_poles_zeros(self, polynomial, variable, root_type):
        """Analisa polos ou zeros de um polinômio"""
        try:
            # Encontrar raízes
            roots = sp.solve(polynomial, variable)
            
            roots_info = []
            for root in roots:
                # Verificar se é real ou complexo
                if root.is_real:
                    root_val = float(root)
                    roots_info.append({
                        'root': root_val,
                        'type': 'real',
                        'frequency': abs(root_val),
                        'multiplicity': 1,  # Simplificação
                        'root_type': root_type
                    })
                else:
                    # Raiz complexa
                    root_complex = complex(root)
                    real_part = root_complex.real
                    imag_part = root_complex.imag
                    
                    if imag_part > 0:  # Considerar apenas uma das raízes conjugadas
                        natural_freq = abs(root_complex)
                        damping_ratio = -real_part / natural_freq if natural_freq > 0 else 0
                        
                        roots_info.append({
                            'root': root_complex,
                            'type': 'complex',
                            'natural_frequency': natural_freq,
                            'damping_ratio': damping_ratio,
                            'frequency': natural_freq,
                            'multiplicity': 1,
                            'root_type': root_type
                        })
            
            return roots_info
            
        except Exception as e:
            warnings.warn(f"Erro na análise de {root_type}: {e}")
            return []
    
    def _determine_system_type_from_factors(self, denominator, variable):
        """Determina tipo do sistema baseado em fatores s no denominador"""
        # Contar fatores s no denominador
        integrator_order = 0
        differentiator_order = 0
        
        temp_den = sp.expand(denominator)
        
        # Contar fatores s (polos na origem)
        while True:
            divided = sp.simplify(temp_den / variable)
            limit_val = sp.limit(divided, variable, 0)
            
            if limit_val == sp.oo or limit_val == -sp.oo:
                break
            
            temp_den = divided
            integrator_order += 1
            
            if integrator_order > 10:  # Limite de segurança
                break
        
        return {
            'integrator_order': integrator_order,
            'differentiator_order': differentiator_order,
            'system_type': integrator_order
        }
    
    def _create_bode_factor_from_root(self, root_info, factor_type):
        """Cria fator Bode a partir de informação de raiz"""
        if root_info['type'] == 'real':
            # Fator real: (s ± a)
            frequency = abs(root_info['root'])
            
            if factor_type == 'zero':
                # Zero: (s + a) -> slope +20 dB/dec, phase +90°
                return {
                    'type': 'real_zero',
                    'corner_frequency': frequency,
                    'slope_db_decade': 20,
                    'phase_change_deg': 90,
                    'root_value': root_info['root']
                }
            else:  # pole
                # Polo: 1/(s + a) -> slope -20 dB/dec, phase -90°
                return {
                    'type': 'real_pole',
                    'corner_frequency': frequency,
                    'slope_db_decade': -20,
                    'phase_change_deg': -90,
                    'root_value': root_info['root']
                }
        
        else:  # complex
            # Fator complexo: (s² + 2ζωn*s + ωn²)
            wn = root_info['natural_frequency']
            zeta = root_info['damping_ratio']
            
            if factor_type == 'zero':
                return {
                    'type': 'complex_zero',
                    'corner_frequency': wn,
                    'damping_ratio': zeta,
                    'slope_db_decade': 40,  # +40 dB/dec após corner
                    'phase_change_deg': 180,  # +180° total
                    'root_value': root_info['root']
                }
            else:  # pole
                return {
                    'type': 'complex_pole',
                    'corner_frequency': wn,
                    'damping_ratio': zeta,
                    'slope_db_decade': -40,  # -40 dB/dec após corner
                    'phase_change_deg': -180,  # -180° total
                    'root_value': root_info['root']
                }
    
    def generate_asymptotic_bode(self, tf_system, substitutions=None, frequency_range=None):
        """
        Gera aproximação assintótica do diagrama de Bode
        
        Args:
            tf_system: SymbolicTransferFunction
            substitutions: Substituições de símbolos
            frequency_range: Tupla (freq_min, freq_max) em rad/s
            
        Returns:
            dict: Dados da aproximação assintótica
        """
        if frequency_range is None:
            frequency_range = (0.01, 1000)
        
        try:
            # Analisar fatores
            factors_analysis = self.analyze_tf_factors(tf_system, substitutions)
            
            # Gerar pontos de frequência
            freq_min, freq_max = frequency_range
            
            # Identificar frequências importantes (corner frequencies)
            corner_frequencies = []
            for factor in factors_analysis['bode_factors']:
                if 'corner_frequency' in factor:
                    corner_frequencies.append(factor['corner_frequency'])
            
            # Criar vetor de frequências incluindo corners
            if corner_frequencies:
                freq_min = min(freq_min, min(corner_frequencies) / 10)
                freq_max = max(freq_max, max(corner_frequencies) * 10)
            
            # Frequências logarítmicas
            num_points = 1000
            frequencies = np.logspace(np.log10(freq_min), np.log10(freq_max), num_points)
            
            # Calcular magnitude e fase assintóticas
            magnitude_db = np.zeros_like(frequencies)
            phase_deg = np.zeros_like(frequencies)
            
            # Processar cada fator
            for factor in factors_analysis['bode_factors']:
                mag_factor, phase_factor = self._calculate_factor_response(factor, frequencies)
                magnitude_db += mag_factor
                phase_deg += phase_factor
            
            # Identificar breakpoints (mudanças de slope)
            breakpoints = self._identify_breakpoints(factors_analysis['bode_factors'])
            
            result = {
                'frequencies': frequencies,
                'magnitude_db': magnitude_db,
                'phase_deg': phase_deg,
                'corner_frequencies': sorted(corner_frequencies),
                'breakpoints': breakpoints,
                'factors_analysis': factors_analysis,
                'frequency_range': frequency_range
            }
            
            self.history.add_step(
                "BODE_ASSINTÓTICO_GERADO",
                f"Aproximação assintótica calculada para {len(frequencies)} pontos",
                f"Corners: {len(corner_frequencies)}, Range: {freq_min:.3f}-{freq_max:.3f} rad/s",
                result
            )
            
            return result
            
        except Exception as e:
            error_msg = f"Erro na geração de Bode assintótico: {e}"
            self.history.add_step("ERRO_BODE_ASSINTÓTICO", error_msg, "", str(e))
            raise ValueError(error_msg) from e
    
    def _calculate_factor_response(self, factor, frequencies):
        """Calcula resposta de um fator individual"""
        magnitude_db = np.zeros_like(frequencies)
        phase_deg = np.zeros_like(frequencies)
        
        if factor['type'] == 'constant':
            # Ganho constante
            magnitude_db.fill(factor['magnitude_db'])
            phase_deg.fill(factor['phase_deg'])
        
        elif factor['type'] == 'integrator':
            # Integrador: 1/s^n
            slope = factor['slope_db_decade']
            phase = factor['phase_deg']
            
            magnitude_db = slope * np.log10(frequencies)
            phase_deg.fill(phase)
        
        elif factor['type'] == 'differentiator':
            # Derivador: s^n
            slope = factor['slope_db_decade']
            phase = factor['phase_deg']
            
            magnitude_db = slope * np.log10(frequencies)
            phase_deg.fill(phase)
        
        elif factor['type'] in ['real_zero', 'real_pole']:
            # Fator real
            corner_freq = factor['corner_frequency']
            slope = factor['slope_db_decade']
            phase_change = factor['phase_change_deg']
            
            # Magnitude assintótica
            magnitude_db = np.where(
                frequencies >= corner_freq,
                slope * np.log10(frequencies / corner_freq),
                0
            )
            
            # Fase assintótica (transição aproximada)
            phase_deg = np.where(
                frequencies >= corner_freq * 10,
                phase_change,
                np.where(
                    frequencies <= corner_freq / 10,
                    0,
                    phase_change * np.log10(frequencies / (corner_freq / 10)) / np.log10(100)
                )
            )
        
        elif factor['type'] in ['complex_zero', 'complex_pole']:
            # Fator complexo
            corner_freq = factor['corner_frequency']
            slope = factor['slope_db_decade']
            phase_change = factor['phase_change_deg']
            
            # Magnitude assintótica (similar ao real, mas slope dobrado)
            magnitude_db = np.where(
                frequencies >= corner_freq,
                slope * np.log10(frequencies / corner_freq),
                0
            )
            
            # Fase assintótica
            phase_deg = np.where(
                frequencies >= corner_freq * 10,
                phase_change,
                np.where(
                    frequencies <= corner_freq / 10,
                    0,
                    phase_change * np.log10(frequencies / (corner_freq / 10)) / np.log10(100)
                )
            )
        
        return magnitude_db, phase_deg
    
    def _identify_breakpoints(self, bode_factors):
        """Identifica pontos de quebra (mudança de slope)"""
        breakpoints = []
        
        for factor in bode_factors:
            if 'corner_frequency' in factor:
                breakpoint = {
                    'frequency': factor['corner_frequency'],
                    'type': factor['type'],
                    'slope_change_db_decade': factor.get('slope_db_decade', 0),
                    'phase_change_deg': factor.get('phase_change_deg', 0)
                }
                breakpoints.append(breakpoint)
        
        # Ordenar por frequência
        breakpoints.sort(key=lambda x: x['frequency'])
        
        return breakpoints
    
    def plot_asymptotic_bode(self, bode_data, show_exact=False, tf_system=None, substitutions=None):
        """
        Plota diagrama de Bode assintótico
        
        Args:
            bode_data: Dados da aproximação assintótica
            show_exact: Se True, sobrepõe resposta exata
            tf_system: TF para resposta exata (se show_exact=True)
            substitutions: Substituições para resposta exata
            
        Returns:
            matplotlib.figure.Figure: Figura do gráfico
        """
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            frequencies = bode_data['frequencies']
            magnitude_db = bode_data['magnitude_db']
            phase_deg = bode_data['phase_deg']
            
            # Plot magnitude
            ax1.semilogx(frequencies, magnitude_db, 'b-', linewidth=2, label='Assintótica')
            ax1.set_ylabel('Magnitude (dB)')
            ax1.grid(True, alpha=0.3)
            ax1.set_title('Diagrama de Bode - Aproximação Assintótica')
            
            # Marcar corner frequencies
            for corner_freq in bode_data['corner_frequencies']:
                if corner_freq > 0:
                    ax1.axvline(corner_freq, color='r', linestyle='--', alpha=0.7)
            
            # Plot fase
            ax2.semilogx(frequencies, phase_deg, 'b-', linewidth=2, label='Assintótica')
            ax2.set_ylabel('Fase (graus)')
            ax2.set_xlabel('Frequência (rad/s)')
            ax2.grid(True, alpha=0.3)
            
            # Marcar corner frequencies
            for corner_freq in bode_data['corner_frequencies']:
                if corner_freq > 0:
                    ax2.axvline(corner_freq, color='r', linestyle='--', alpha=0.7)
            
            # Adicionar resposta exata se solicitado
            if show_exact and tf_system and CONTROL_AVAILABLE:
                try:
                    from .interface import NumericalInterface
                    interface = NumericalInterface()
                    
                    # Calcular resposta exata
                    freq_response = interface.compute_frequency_response(
                        tf_system, substitutions, frequencies
                    )
                    
                    exact_mag_db = 20 * np.log10(np.abs(freq_response['frequency_response']))
                    exact_phase_deg = np.angle(freq_response['frequency_response'], deg=True)
                    
                    ax1.semilogx(frequencies, exact_mag_db, 'g--', linewidth=1.5, 
                               label='Exata', alpha=0.8)
                    ax2.semilogx(frequencies, exact_phase_deg, 'g--', linewidth=1.5, 
                               label='Exata', alpha=0.8)
                    
                except Exception as e:
                    warnings.warn(f"Erro ao calcular resposta exata: {e}")
            
            ax1.legend()
            ax2.legend()
            
            plt.tight_layout()
            
            self.history.add_step(
                "PLOT_BODE_GERADO",
                "Gráfico de Bode assintótico criado",
                f"Corners: {len(bode_data['corner_frequencies'])}, Exata: {show_exact}",
                "Plot concluído"
            )
            
            return fig
            
        except Exception as e:
            error_msg = f"Erro na criação do gráfico: {e}"
            self.history.add_step("ERRO_PLOT", error_msg, "", str(e))
            raise ValueError(error_msg) from e
    
    def get_bode_rules_summary(self, bode_data):
        """
        Gera resumo das regras de esboço de Bode aplicadas
        
        Args:
            bode_data: Dados da aproximação assintótica
            
        Returns:
            dict: Resumo das regras aplicadas
        """
        try:
            factors = bode_data['factors_analysis']['bode_factors']
            breakpoints = bode_data['breakpoints']
            
            # Construir slopes acumulativos
            cumulative_slope = 0
            slope_changes = []
            
            # Slope inicial (ganho + integrador/derivador)
            for factor in factors:
                if factor['type'] in ['constant', 'integrator', 'differentiator']:
                    if 'slope_db_decade' in factor:
                        cumulative_slope += factor['slope_db_decade']
            
            initial_slope = cumulative_slope
            
            # Adicionar mudanças nos breakpoints
            for bp in breakpoints:
                slope_change = bp.get('slope_change_db_decade', 0)
                cumulative_slope += slope_change
                slope_changes.append({
                    'frequency': bp['frequency'],
                    'slope_change': slope_change,
                    'new_slope': cumulative_slope,
                    'factor_type': bp['type']
                })
            
            # Resumir regras
            rules_applied = []
            
            # Regra 1: Slope inicial
            rules_applied.append(f"Slope inicial: {initial_slope} dB/década")
            
            # Regra 2: Corner frequencies
            if breakpoints:
                rules_applied.append(f"Corner frequencies: {[f'{bp['frequency']:.3f}' for bp in breakpoints[:5]]}")
            
            # Regra 3: Mudanças de slope
            for change in slope_changes[:5]:  # Limitar para primeiras 5
                rules_applied.append(
                    f"f={change['frequency']:.3f}: slope {change['slope_change']:+d} → {change['new_slope']} dB/década"
                )
            
            summary = {
                'initial_slope_db_decade': initial_slope,
                'final_slope_db_decade': cumulative_slope,
                'total_breakpoints': len(breakpoints),
                'slope_changes': slope_changes,
                'rules_applied': rules_applied,
                'bode_construction_steps': self._generate_construction_steps(factors, breakpoints)
            }
            
            self.history.add_step(
                "RESUMO_REGRAS_BODE",
                f"Regras de Bode aplicadas: {len(rules_applied)} regras",
                f"Slope inicial: {initial_slope}, Final: {cumulative_slope} dB/década",
                summary
            )
            
            return summary
            
        except Exception as e:
            error_msg = f"Erro no resumo de regras: {e}"
            self.history.add_step("ERRO_RESUMO_REGRAS", error_msg, "", str(e))
            raise ValueError(error_msg) from e
    
    def _generate_construction_steps(self, factors, breakpoints):
        """Gera passos de construção do diagrama"""
        steps = []
        
        # Passo 1: Ganho
        gain_factors = [f for f in factors if f['type'] == 'constant']
        if gain_factors:
            K = gain_factors[0]['K']
            steps.append(f"1. Traçar linha horizontal em {20*np.log10(abs(K)):.1f} dB (ganho K={K})")
        
        # Passo 2: Integradores/Derivadores
        int_factors = [f for f in factors if f['type'] in ['integrator', 'differentiator']]
        if int_factors:
            for factor in int_factors:
                if factor['type'] == 'integrator':
                    steps.append(f"2. Adicionar slope {factor['slope_db_decade']} dB/década (integrador ordem {factor['order']})")
                else:
                    steps.append(f"2. Adicionar slope +{factor['slope_db_decade']} dB/década (derivador ordem {factor['order']})")
        
        # Passo 3: Breakpoints
        if breakpoints:
            steps.append("3. Adicionar breakpoints:")
            for i, bp in enumerate(breakpoints[:5]):
                factor_name = "zero" if "zero" in bp['type'] else "polo"
                slope_change = bp.get('slope_change_db_decade', 0)
                steps.append(f"   f={bp['frequency']:.3f} rad/s: {factor_name} ({slope_change:+d} dB/década)")
        
        return steps
