"""
Módulo de análise de estabilidade para sistemas de controle
"""

import sympy as sp
import numpy as np
from typing import List, Dict, Union, Tuple
from .symbolic_tf import SymbolicTransferFunction
from .history import OperationHistory


class RouthHurwitzAnalyzer:
    """Análise de estabilidade usando critério de Routh-Hurwitz"""
    
    def __init__(self):
        self.history = OperationHistory()
    
    def analyze(self, characteristic_poly: sp.Expr, variable: sp.Symbol) -> dict:
        """
        Executa análise de Routh-Hurwitz
        
        Args:
            characteristic_poly: Polinômio característico
            variable: Variável do polinômio
            
        Returns:
            dict: Resultado da análise
        """
        self.history.add_step(
            "INÍCIO_ROUTH_HURWITZ",
            "Iniciando análise de Routh-Hurwitz",
            str(characteristic_poly),
            "Construindo tabela de Routh"
        )
        
        try:
            # Extrai coeficientes
            poly = sp.Poly(characteristic_poly, variable)
            coeffs = poly.all_coeffs()
            
            # Constrói tabela de Routh
            routh_table = self._build_routh_table(coeffs)
            
            # Analisa estabilidade
            stability_result = self._analyze_stability(routh_table)
            
            result = {
                'coefficients': coeffs,
                'routh_table': routh_table,
                'is_stable': stability_result['is_stable'],
                'poles_right_half_plane': stability_result['poles_rhp'],
                'poles_on_imaginary_axis': stability_result['poles_jw'],
                'critical_gain': stability_result.get('critical_gain'),
                'analysis_steps': self.history.get_formatted_history()
            }
            
            self.history.add_step(
                "RESULTADO_ROUTH_HURWITZ",
                f"Sistema {'estável' if result['is_stable'] else 'instável'}",
                f"Polos no semiplano direito: {result['poles_right_half_plane']}",
                f"Polos no eixo imaginário: {result['poles_on_imaginary_axis']}"
            )
            
            return result
            
        except Exception as e:
            self.history.add_step(
                "ERRO_ROUTH_HURWITZ",
                f"Erro na análise: {str(e)}",
                str(characteristic_poly),
                None
            )
            return {'error': str(e)}
    
    def _build_routh_table(self, coeffs: List) -> List[List]:
        """Constrói tabela de Routh"""
        n = len(coeffs)
        table = []
        
        # Primeira linha (s^n)
        first_row = [coeffs[i] if i < len(coeffs) else 0 for i in range(0, len(coeffs), 2)]
        table.append(first_row)
        
        # Segunda linha (s^(n-1))
        second_row = [coeffs[i] if i < len(coeffs) else 0 for i in range(1, len(coeffs), 2)]
        table.append(second_row)
        
        # Linhas subsequentes
        for row_idx in range(2, n):
            new_row = []
            prev_row = table[row_idx - 1]
            prev_prev_row = table[row_idx - 2]
            
            for col_idx in range(len(prev_prev_row) - 1):
                if prev_row[0] == 0:
                    # Caso especial: primeiro elemento é zero
                    new_row.append(float('inf'))
                else:
                    try:
                        element = (prev_row[0] * prev_prev_row[col_idx + 1] - 
                                 prev_prev_row[0] * prev_row[col_idx + 1]) / prev_row[0]
                        new_row.append(sp.simplify(element))
                    except:
                        new_row.append(0)
            
            if new_row:
                table.append(new_row)
            else:
                break
        
        self.history.add_step(
            "TABELA_ROUTH",
            "Tabela de Routh construída",
            f"Coeficientes: {coeffs}",
            f"Tabela: {table}"
        )
        
        return table
    
    def _analyze_stability(self, routh_table: List[List]) -> dict:
        """Analisa estabilidade baseada na tabela de Routh"""
        first_column = [row[0] for row in routh_table if row]
        
        # Conta mudanças de sinal na primeira coluna
        sign_changes = 0
        prev_sign = None
        
        for element in first_column:
            try:
                # Tenta avaliar numericamente
                numeric_val = complex(element).real
                current_sign = 1 if numeric_val > 0 else -1 if numeric_val < 0 else 0
                
                if prev_sign is not None and current_sign != 0 and current_sign != prev_sign:
                    sign_changes += 1
                
                if current_sign != 0:
                    prev_sign = current_sign
                    
            except:
                # Se não conseguir avaliar, considera como mudança de sinal
                sign_changes += 1
        
        return {
            'is_stable': sign_changes == 0,
            'poles_rhp': sign_changes,
            'poles_jw': 0,  # Seria necessário análise mais detalhada
            'first_column': first_column
        }


class NyquistAnalyzer:
    """Análise de estabilidade usando critério de Nyquist"""
    
    def __init__(self):
        self.history = OperationHistory()
    
    def analyze(self, open_loop_tf: SymbolicTransferFunction, 
                frequency_range: Tuple[float, float] = (-3, 3)) -> dict:
        """
        Executa análise de Nyquist
        
        Args:
            open_loop_tf: Função de transferência de malha aberta
            frequency_range: Faixa de frequências (log10)
            
        Returns:
            dict: Resultado da análise
        """
        self.history.add_step(
            "INÍCIO_NYQUIST",
            "Iniciando análise de Nyquist",
            str(open_loop_tf),
            f"Faixa de frequência: 10^{frequency_range[0]} a 10^{frequency_range[1]}"
        )
        
        try:
            # Substitui s por jω
            omega = sp.Symbol('omega', real=True)
            jw_expr = open_loop_tf.numerator / open_loop_tf.denominator
            jw_expr = jw_expr.subs(open_loop_tf.variable, sp.I * omega)
            
            # Análise de polos de malha aberta no semiplano direito
            poles = open_loop_tf.poles()
            rhp_poles = self._count_rhp_poles(poles)
            
            result = {
                'frequency_response': jw_expr,
                'frequency_var': omega,
                'rhp_poles_open_loop': rhp_poles,
                'nyquist_expression': jw_expr,
                'analysis_steps': self.history.get_formatted_history()
            }
            
            self.history.add_step(
                "RESULTADO_NYQUIST",
                f"Polos de malha aberta no RHP: {rhp_poles}",
                str(jw_expr),
                "Use plotagem para análise visual completa"
            )
            
            return result
            
        except Exception as e:
            self.history.add_step(
                "ERRO_NYQUIST",
                f"Erro na análise: {str(e)}",
                str(open_loop_tf),
                None
            )
            return {'error': str(e)}
    
    def _count_rhp_poles(self, poles: List[sp.Expr]) -> int:
        """Conta polos no semiplano direito"""
        rhp_count = 0
        
        for pole in poles:
            try:
                # Tenta avaliar numericamente
                numeric_pole = complex(pole)
                if numeric_pole.real > 0:
                    rhp_count += 1
            except:
                # Se não conseguir avaliar, assume conservadoramente
                rhp_count += 1
        
        return rhp_count


class BodeAnalyzer:
    """Análise de resposta em frequência usando diagramas de Bode"""
    
    def __init__(self):
        self.history = OperationHistory()
    
    def analyze(self, transfer_function: SymbolicTransferFunction) -> dict:
        """
        Executa análise de Bode
        
        Args:
            transfer_function: Função de transferência para análise
            
        Returns:
            dict: Resultado da análise
        """
        self.history.add_step(
            "INÍCIO_BODE",
            "Iniciando análise de Bode",
            str(transfer_function),
            "Calculando magnitude e fase"
        )
        
        try:
            # Substitui s por jω
            omega = sp.Symbol('omega', real=True, positive=True)
            jw_expr = transfer_function.numerator / transfer_function.denominator
            jw_expr = jw_expr.subs(transfer_function.variable, sp.I * omega)
            
            # Magnitude em dB
            magnitude_db = 20 * sp.log(sp.Abs(jw_expr), 10)
            
            # Fase em graus
            phase_rad = sp.arg(jw_expr)
            phase_deg = phase_rad * 180 / sp.pi
            
            # Análise de margens
            margins = self._calculate_margins(jw_expr, omega)
            
            result = {
                'magnitude_db': magnitude_db,
                'phase_deg': phase_deg,
                'frequency_var': omega,
                'gain_margin': margins.get('gain_margin'),
                'phase_margin': margins.get('phase_margin'),
                'gain_crossover': margins.get('gain_crossover'),
                'phase_crossover': margins.get('phase_crossover'),
                'analysis_steps': self.history.get_formatted_history()
            }
            
            self.history.add_step(
                "RESULTADO_BODE",
                "Análise de Bode concluída",
                f"Magnitude: {magnitude_db}",
                f"Fase: {phase_deg}"
            )
            
            return result
            
        except Exception as e:
            self.history.add_step(
                "ERRO_BODE",
                f"Erro na análise: {str(e)}",
                str(transfer_function),
                None
            )
            return {'error': str(e)}
    
    def _calculate_margins(self, jw_expr: sp.Expr, omega: sp.Symbol) -> dict:
        """Calcula margens de ganho e fase"""
        try:
            # Para margens precisas, seria necessário resolver equações
            # Aqui fornecemos as expressões simbólicas
            magnitude = sp.Abs(jw_expr)
            phase = sp.arg(jw_expr)
            
            return {
                'magnitude_expr': magnitude,
                'phase_expr': phase,
                'note': 'Resolva |G(jω)| = 1 para frequência de cruzamento de ganho',
                'note2': 'Resolva ∠G(jω) = -180° para frequência de cruzamento de fase'
            }
        except:
            return {}


class RootLocusAnalyzer:
    """Análise do lugar das raízes"""
    
    def __init__(self):
        self.history = OperationHistory()
    
    def analyze(self, open_loop_tf: SymbolicTransferFunction, 
                gain_symbol: sp.Symbol = None) -> dict:
        """
        Executa análise do lugar das raízes
        
        Args:
            open_loop_tf: Função de transferência de malha aberta
            gain_symbol: Símbolo para o ganho variável
            
        Returns:
            dict: Resultado da análise
        """
        if gain_symbol is None:
            gain_symbol = sp.Symbol('K', real=True, positive=True)
        
        self.history.add_step(
            "INÍCIO_ROOT_LOCUS",
            "Iniciando análise do lugar das raízes",
            str(open_loop_tf),
            f"Ganho variável: {gain_symbol}"
        )
        
        try:
            # Equação característica: 1 + K*G(s) = 0
            char_eq = open_loop_tf.denominator + gain_symbol * open_loop_tf.numerator
            
            # Pontos de partida (polos de malha aberta)
            poles_open = open_loop_tf.poles()
            
            # Pontos de chegada (zeros de malha aberta)
            zeros_open = open_loop_tf.zeros()
            
            # Assintotas
            n_poles = len(poles_open)
            n_zeros = len(zeros_open)
            n_asymptotes = n_poles - n_zeros
            
            result = {
                'characteristic_equation': char_eq,
                'gain_symbol': gain_symbol,
                'open_loop_poles': poles_open,
                'open_loop_zeros': zeros_open,
                'num_asymptotes': n_asymptotes,
                'analysis_steps': self.history.get_formatted_history()
            }
            
            if n_asymptotes > 0:
                # Centroide das assintotas
                sum_poles = sum(poles_open)
                sum_zeros = sum(zeros_open) if zeros_open else 0
                centroid = (sum_poles - sum_zeros) / n_asymptotes
                result['asymptote_centroid'] = centroid
                
                # Ângulos das assintotas
                asymptote_angles = [(2*k + 1) * sp.pi / n_asymptotes 
                                  for k in range(n_asymptotes)]
                result['asymptote_angles'] = asymptote_angles
            
            self.history.add_step(
                "RESULTADO_ROOT_LOCUS",
                f"Lugar das raízes para {gain_symbol}*{open_loop_tf}",
                f"Polos: {poles_open}",
                f"Zeros: {zeros_open}"
            )
            
            return result
            
        except Exception as e:
            self.history.add_step(
                "ERRO_ROOT_LOCUS",
                f"Erro na análise: {str(e)}",
                str(open_loop_tf),
                None
            )
            return {'error': str(e)}
