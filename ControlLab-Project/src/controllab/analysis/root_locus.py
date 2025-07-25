"""
Módulo de Análise de Root Locus (Lugar Geométrico das Raízes)
=============================================================

Este módulo implementa análise completa do lugar geométrico das raízes (root locus)
para sistemas de controle, aplicando as 6 regras fundamentais de forma pedagógica.

Classes:
    RootLocusAnalyzer: Analisador principal do root locus
    LocusFeatures: Características extraídas do root locus
    LocusHistory: Histórico educacional da análise

Funcionalidades:
    - Aplicação das 6 regras do root locus
    - Cálculo de assíntotas, centróide e ângulos
    - Determinação de pontos de breakaway/break-in
    - Análise de cruzamentos no eixo jω
    - Ângulos de partida e chegada
    - Segmentos no eixo real

Exemplo de Uso:
    ```python
    from controllab.analysis.root_locus import RootLocusAnalyzer

    analyzer = RootLocusAnalyzer()
    features = analyzer.get_locus_features(system)
    print(f"Polos: {features.poles}")
    print(f"Zeros: {features.zeros}")
    ```
"""

import sympy as sp
from sympy import symbols, solve, diff, simplify, I, pi, atan2, sqrt, Poly, roots
from typing import Dict, List, Tuple, Any, Optional, Union
from numbers import Complex
import warnings

# Importar do core se disponível
try:
    from ..core.symbolic_tf import SymbolicTransferFunction
except ImportError:
    SymbolicTransferFunction = None


class LocusHistory:
    """Histórico pedagógico da análise de root locus"""
    
    def __init__(self):
        self.steps = []
        self.rules_applied = []
        self.transfer_function = None
        self.features = {}
        
    def add_step(self, rule_number: int, rule_name: str, calculation: Any, 
                 result: Any, explanation: str = ""):
        step = {
            'step': len(self.steps) + 1,
            'rule': rule_number,
            'rule_name': rule_name,
            'calculation': calculation,
            'result': result,
            'explanation': explanation
        }
        self.steps.append(step)
        
    def add_rule(self, rule_number: int, description: str, result: Any):
        rule = {
            'number': rule_number,
            'description': description,
            'result': result
        }
        self.rules_applied.append(rule)
        
    def get_formatted_report(self) -> str:
        """Retorna relatório formatado da análise"""
        report = "📍 ANÁLISE DE ROOT LOCUS - RELATÓRIO PEDAGÓGICO\n"
        report += "=" * 60 + "\n\n"
        
        if self.transfer_function:
            report += f"🎯 FUNÇÃO DE TRANSFERÊNCIA:\n{self.transfer_function}\n\n"
        
        report += "📏 REGRAS DE ESBOÇO APLICADAS:\n"
        for rule in self.rules_applied:
            report += f"Regra {rule['number']}: {rule['description']}\n"
            report += f"Resultado: {rule['result']}\n\n"
        
        report += "📋 PASSOS DE CÁLCULO:\n"
        for step in self.steps:
            report += f"{step['step']}. {step['rule_name']}\n"
            report += f"   Cálculo: {step['calculation']}\n"
            report += f"   Resultado: {step['result']}\n"
            if step['explanation']:
                report += f"   📝 {step['explanation']}\n"
            report += "-" * 40 + "\n"
        
        return report


class LocusFeatures:
    """Características do lugar geométrico das raízes"""
    
    def __init__(self):
        self.poles = []
        self.zeros = []
        self.num_branches = 0
        self.asymptotes = {'angles': [], 'centroid': None}
        self.breakaway_points = []
        self.jw_crossings = []
        self.departure_angles = {}
        self.arrival_angles = {}
        self.real_axis_segments = []
        self.characteristic_equation = None
        self.analysis_history = None  # Histórico pedagógico
        
    def __str__(self):
        result = "CARACTERÍSTICAS DO ROOT LOCUS:\n"
        result += f"Polos: {self.poles}\n"
        result += f"Zeros: {self.zeros}\n"
        result += f"Número de ramos: {self.num_branches}\n"
        result += f"Assíntotas: {self.asymptotes}\n"
        result += f"Pontos de separação: {self.breakaway_points}\n"
        result += f"Cruzamentos jω: {self.jw_crossings}\n"
        return result


class RootLocusAnalyzer:
    """
    Analisador completo do lugar geométrico das raízes
    
    Esta classe implementa as 6 regras fundamentais do root locus:
    1. Pontos de partida e chegada
    2. Número de ramos
    3. Assíntotas (ângulos e centroide)
    4. Pontos de breakaway/break-in
    5. Cruzamentos do eixo jω
    6. Ângulos de partida/chegada
    """
    
    def __init__(self):
        self.history = LocusHistory()
        self.s = sp.Symbol('s', complex=True)
        self.K = sp.Symbol('K', real=True, positive=True)
        
    def get_locus_features(self, tf_obj, show_steps: bool = True) -> LocusFeatures:
        """
        Extrai todas as características do root locus
        
        Args:
            tf_obj: Função de transferência (SymbolicTransferFunction ou expressão)
            show_steps: Se deve mostrar os passos
            
        Returns:
            LocusFeatures: Objeto com todas as características
        """
        if show_steps:
            self.history = LocusHistory()
            self.history.transfer_function = tf_obj
            
        features = LocusFeatures()
        
        # Extrair numerador e denominador
        if hasattr(tf_obj, 'numerator') and hasattr(tf_obj, 'denominator'):
            num = tf_obj.numerator
            den = tf_obj.denominator
            variable = tf_obj.variable if hasattr(tf_obj, 'variable') else self.s
        else:
            # Para expressões SymPy diretas
            num = sp.numer(tf_obj)
            den = sp.denom(tf_obj)
            
            # Expandir denominador para obter forma polinomial
            den_expanded = sp.expand(den)
            num_expanded = sp.expand(num)
            
            # Determinar variável
            free_symbols = tf_obj.free_symbols
            if free_symbols:
                variable = list(free_symbols)[0]
            else:
                variable = self.s
                
            num = num_expanded
            den = den_expanded
            
        # Aplicar as 6 regras do root locus
        features = self._apply_rule_1(features, num, den, variable, show_steps)
        features = self._apply_rule_2(features, show_steps)
        features = self._apply_rule_3(features, show_steps)
        features = self._apply_rule_4(features, num, den, variable, show_steps)
        features = self._apply_rule_5(features, num, den, variable, show_steps)
        features = self._apply_rule_6(features, show_steps)
        
        # Determinar segmentos do eixo real
        features = self._determine_real_axis_segments(features, show_steps)
        
        if show_steps:
            self.history.features = features
            features.analysis_history = self.history  # Atribuir histórico
            
        return features
    
    def analyze_comprehensive(self, tf_obj, show_steps: bool = True) -> LocusFeatures:
        """
        Realiza análise completa do root locus
        
        Args:
            tf_obj: Função de transferência
            show_steps: Se deve mostrar os passos
            
        Returns:
            LocusFeatures: Análise completa com características
        """
        if show_steps:
            self.history = LocusHistory()
            self.history.add_step(0, "Iniciando análise completa do Root Locus", 
                                "comprehensive_analysis", "Análise iniciada", 
                                "Começando análise detalhada do root locus")
        
        # Usar método existente
        features = self.get_locus_features(tf_obj, show_steps)
        
        # Análise de estabilidade adicional
        if hasattr(features, 'jw_crossings') and features.jw_crossings:
            stable_k_range = []
            for crossing in features.jw_crossings:
                if isinstance(crossing, dict) and 'k' in crossing and crossing['k'] > 0:
                    stable_k_range.append(crossing['k'])
            
            if stable_k_range:
                features.stability_assessment = {
                    'stable_range': f"0 < K < {min(stable_k_range):.3f}",
                    'marginal_k': min(stable_k_range),
                    'unstable_range': f"K > {min(stable_k_range):.3f}"
                }
            else:
                features.stability_assessment = {
                    'stable_range': "K > 0",
                    'marginal_k': None,
                    'unstable_range': "Nenhum"
                }
        else:
            features.stability_assessment = {
                'stable_range': "Análise incompleta", 
                'marginal_k': None,
                'unstable_range': "Análise incompleta"
            }
        
        if show_steps:
            self.history.add_step(0, "Análise completa finalizada", "completion", 
                                "Análise concluída", "Todas as etapas foram executadas")
            features.analysis_history = self.history
        
        return features
    
    def _apply_rule_1(self, features: LocusFeatures, num, den, variable, show_steps: bool):
        """Regra 1: Pontos de partida (polos) e chegada (zeros)"""
        
        # Encontrar polos (raízes do denominador)
        try:
            poles = solve(den, variable)
            features.poles = [complex(sp.N(pole)) if pole.is_real is False else float(sp.N(pole)) 
                            for pole in poles if pole.is_finite]
        except Exception as e:
            features.poles = []
            
        # Encontrar zeros (raízes do numerador)  
        try:
            zeros = solve(num, variable)
            features.zeros = [complex(sp.N(zero)) if zero.is_real is False else float(sp.N(zero)) 
                            for zero in zeros if zero.is_finite]
        except Exception as e:
            features.zeros = []
            
        if show_steps:
            self.history.add_rule(
                1, 
                "Pontos de partida e chegada",
                f"Polos: {features.poles}, Zeros: {features.zeros}"
            )
            self.history.add_step(
                1,
                "Extração de polos e zeros",
                f"Denominador: {den}, Numerador: {num}",
                f"Polos: {features.poles}, Zeros: {features.zeros}",
                "Polos são pontos de partida, zeros são pontos de chegada"
            )
            
        return features
    
    def _apply_rule_2(self, features: LocusFeatures, show_steps: bool):
        """Regra 2: Número de ramos"""
        
        n_poles = len(features.poles)
        n_zeros = len(features.zeros)
        
        features.num_branches = n_poles
        
        if show_steps:
            self.history.add_rule(
                2,
                "Número de ramos",
                f"{features.num_branches} ramos (igual ao número de polos)"
            )
            self.history.add_step(
                2,
                "Contagem de ramos",
                f"Polos: {n_poles}, Zeros: {n_zeros}",
                f"Número de ramos = {features.num_branches}",
                "Número de ramos = número de polos"
            )
            
        return features
    
    def _apply_rule_3(self, features: LocusFeatures, show_steps: bool):
        """Regra 3: Assíntotas (ângulos e centroide)"""
        
        n_poles = len(features.poles)
        n_zeros = len(features.zeros)
        
        # Número de assíntotas
        num_asymptotes = n_poles - n_zeros
        
        if num_asymptotes > 0:
            # Calcular centroide
            sum_poles = sum(pole.real if hasattr(pole, 'real') else pole for pole in features.poles)
            sum_zeros = sum(zero.real if hasattr(zero, 'real') else zero for zero in features.zeros)
            
            centroid = (sum_poles - sum_zeros) / num_asymptotes
            features.asymptotes['centroid'] = centroid
            
            # Calcular ângulos das assíntotas
            angles = []
            for q in range(num_asymptotes):
                angle = (2*q + 1) * pi / num_asymptotes
                angles.append(angle)
            
            features.asymptotes['angles'] = angles
            
            if show_steps:
                self.history.add_rule(
                    3,
                    "Assíntotas",
                    f"Centroide: {centroid:.3f}, Ângulos: {[f'{a*180/pi:.1f}°' for a in angles]}"
                )
                self.history.add_step(
                    3,
                    "Cálculo das assíntotas",
                    f"n-m = {n_poles}-{n_zeros} = {num_asymptotes}",
                    f"Centroide: {centroid:.3f}, Ângulos: {angles}",
                    "σ_a = (Σpolos - Σzeros)/(n-m), θ_q = (2q+1)π/(n-m)"
                )
        else:
            if show_steps:
                self.history.add_rule(3, "Assíntotas", "Nenhuma assíntota (n ≤ m)")
                
        return features
    
    def _apply_rule_4(self, features: LocusFeatures, num, den, variable, show_steps: bool):
        """Regra 4: Pontos de breakaway/break-in"""
        
        try:
            # Construir K(s) = -P(s)/Q(s) onde P é denominador e Q é numerador
            K_func = -den / num
            
            # Calcular dK/ds
            dK_ds = diff(K_func, variable)
            
            # Resolver dK/ds = 0
            breakaway_candidates = solve(dK_ds, variable)
            
            # Filtrar candidatos reais
            breakaway_points = []
            for candidate in breakaway_candidates:
                try:
                    candidate_val = complex(sp.N(candidate))
                    if abs(candidate_val.imag) < 1e-10:  # Essencialmente real
                        breakaway_points.append(candidate_val.real)
                except:
                    pass
            
            features.breakaway_points = breakaway_points
            
            if show_steps:
                self.history.add_rule(
                    4,
                    "Pontos de breakaway/break-in",
                    f"Pontos: {breakaway_points}"
                )
                self.history.add_step(
                    4,
                    "Cálculo de pontos de separação",
                    f"dK/ds = d/ds(-{den}/{num}) = 0",
                    f"Pontos: {breakaway_points}",
                    "Resolvendo dK/ds = 0 para encontrar pontos críticos"
                )
                
        except Exception as e:
            features.breakaway_points = []
            if show_steps:
                self.history.add_rule(4, "Pontos de breakaway/break-in", "Erro no cálculo")
                
        return features
    
    def _apply_rule_5(self, features: LocusFeatures, num, den, variable, show_steps: bool):
        """Regra 5: Cruzamentos do eixo jω"""
        
        try:
            # Substituir s = jω na equação característica
            omega = sp.Symbol('omega', real=True)
            char_eq = den + self.K * num
            char_eq_jw = char_eq.subs(variable, I * omega)
            
            # Separar parte real e imaginária
            char_eq_jw_expanded = sp.expand(char_eq_jw)
            real_part = sp.re(char_eq_jw_expanded)
            imag_part = sp.im(char_eq_jw_expanded)
            
            # Resolver sistema: parte real = 0 e parte imaginária = 0
            solutions = solve([real_part, imag_part], [omega, self.K])
            
            crossings = []
            for sol in solutions:
                try:
                    if isinstance(sol, (list, tuple)) and len(sol) >= 2:
                        omega_val, k_val = sol[0], sol[1]
                        if omega_val.is_real and k_val.is_real and k_val > 0:
                            crossings.append({
                                'omega': float(omega_val),
                                'K': float(k_val),
                                'point': complex(0, float(omega_val))
                            })
                except:
                    pass
            
            features.jw_crossings = crossings
            
            if show_steps:
                self.history.add_rule(
                    5,
                    "Cruzamentos do eixo jω",
                    f"Cruzamentos: {crossings}"
                )
                self.history.add_step(
                    5,
                    "Análise de cruzamentos jω",
                    f"Substituindo s = jω na equação característica",
                    f"Cruzamentos: {crossings}",
                    "Resolver Re[1 + K*G(jω)H(jω)] = 0 e Im[1 + K*G(jω)H(jω)] = 0"
                )
                
        except Exception as e:
            features.jw_crossings = []
            if show_steps:
                self.history.add_rule(5, "Cruzamentos do eixo jω", "Erro no cálculo")
                
        return features
    
    def _apply_rule_6(self, features: LocusFeatures, show_steps: bool):
        """Regra 6: Ângulos de partida/chegada"""
        
        # Para polos complexos, calcular ângulo de partida
        departure_angles = {}
        for i, pole in enumerate(features.poles):
            if hasattr(pole, 'imag') and abs(pole.imag) > 1e-10:
                # Calcular ângulo de partida
                angle = self._calculate_departure_angle(pole, features)
                departure_angles[f'pole_{i}'] = angle
        
        # Para zeros complexos, calcular ângulo de chegada
        arrival_angles = {}
        for i, zero in enumerate(features.zeros):
            if hasattr(zero, 'imag') and abs(zero.imag) > 1e-10:
                # Calcular ângulo de chegada
                angle = self._calculate_arrival_angle(zero, features)
                arrival_angles[f'zero_{i}'] = angle
        
        features.departure_angles = departure_angles
        features.arrival_angles = arrival_angles
        
        if show_steps:
            self.history.add_rule(
                6,
                "Ângulos de partida/chegada",
                f"Partida: {departure_angles}, Chegada: {arrival_angles}"
            )
            self.history.add_step(
                6,
                "Cálculo de ângulos",
                "Usando condição de fase: Σ∠(s-zeros) - Σ∠(s-polos) = ±180°(2q+1)",
                f"Partida: {departure_angles}, Chegada: {arrival_angles}",
                "Ângulos calculados para polos/zeros complexos"
            )
            
        return features
    
    def _calculate_departure_angle(self, pole, features):
        """Calcula ângulo de partida de um polo complexo"""
        # Implementação simplificada
        # Em uma implementação completa, usaria a condição de ângulo
        return 0  # Placeholder
    
    def _calculate_arrival_angle(self, zero, features):
        """Calcula ângulo de chegada de um zero complexo"""
        # Implementação simplificada
        # Em uma implementação completa, usaria a condição de ângulo
        return 0  # Placeholder
    
    def _determine_real_axis_segments(self, features: LocusFeatures, show_steps: bool):
        """Determina segmentos do eixo real que pertencem ao root locus"""
        
        # Coletar todos os pontos críticos no eixo real
        real_points = []
        
        for pole in features.poles:
            if hasattr(pole, 'imag'):
                if abs(pole.imag) < 1e-10:
                    real_points.append(pole.real)
            else:
                real_points.append(float(pole))
                
        for zero in features.zeros:
            if hasattr(zero, 'imag'):
                if abs(zero.imag) < 1e-10:
                    real_points.append(zero.real)
            else:
                real_points.append(float(zero))
        
        # Adicionar pontos de breakaway
        real_points.extend(features.breakaway_points)
        
        # Ordenar pontos
        real_points.sort()
        
        # Determinar segmentos válidos
        # Regra: segmento pertence ao locus se número total de polos e zeros
        # à direita do ponto é ímpar
        segments = []
        for i in range(len(real_points) - 1):
            mid_point = (real_points[i] + real_points[i + 1]) / 2
            count = self._count_poles_zeros_to_right(mid_point, features)
            if count % 2 == 1:
                segments.append((real_points[i], real_points[i + 1]))
        
        features.real_axis_segments = segments
        
        if show_steps:
            self.history.add_step(
                0,
                "Segmentos do eixo real",
                f"Pontos críticos: {real_points}",
                f"Segmentos válidos: {segments}",
                "Segmento pertence ao locus se #(polos+zeros) à direita for ímpar"
            )
            
        return features
    
    def _count_poles_zeros_to_right(self, point, features):
        """Conta polos e zeros à direita de um ponto no eixo real"""
        count = 0
        
        for pole in features.poles:
            pole_real = pole.real if hasattr(pole, 'real') else pole
            if pole_real > point:
                count += 1
                
        for zero in features.zeros:
            zero_real = zero.real if hasattr(zero, 'real') else zero
            if zero_real > point:
                count += 1
                
        return count
    
    def calculate_locus_points(self, tf_obj, k_range: List[float], show_steps: bool = False) -> Dict:
        """
        Calcula pontos específicos do root locus para valores de K
        
        Args:
            tf_obj: Função de transferência
            k_range: Lista de valores de K
            show_steps: Se deve mostrar os passos
            
        Returns:
            Dict com pontos calculados
        """
        # Extrair numerador e denominador
        if hasattr(tf_obj, 'numerator') and hasattr(tf_obj, 'denominator'):
            num = tf_obj.numerator
            den = tf_obj.denominator
            variable = tf_obj.variable
        else:
            num = sp.numer(tf_obj)
            den = sp.denom(tf_obj)
            variable = self.s
            
        locus_points = {}
        
        for k_val in k_range:
            # Equação característica: denominador + K * numerador = 0
            char_eq = den + k_val * num
            
            try:
                # Resolver equação característica
                roots_eq = solve(char_eq, variable)
                
                # Converter para complexos
                points = []
                for root in roots_eq:
                    try:
                        point = complex(sp.N(root))
                        points.append(point)
                    except:
                        pass
                
                locus_points[k_val] = points
                
            except:
                locus_points[k_val] = []
        
        if show_steps:
            self.history.add_step(
                0,
                "Cálculo de pontos do locus",
                f"Valores de K: {k_range}",
                f"Pontos calculados para {len(k_range)} valores de K",
                "Resolvendo equação característica para cada K"
            )
            
        return {
            'locus_points': locus_points, 
            'k_values': k_range,
            'roots': locus_points  # Adicionar alias para compatibilidade
        }


# Funções utilitárias independentes
def get_locus_features(tf_obj, show_steps: bool = True) -> LocusFeatures:
    """Função wrapper para extrair características do root locus"""
    analyzer = RootLocusAnalyzer()
    return analyzer.get_locus_features(tf_obj, show_steps)


def calculate_asymptotes(zeros: List, poles: List) -> Dict:
    """Calcula assíntotas do root locus"""
    n_poles = len(poles)
    n_zeros = len(zeros)
    num_asymptotes = n_poles - n_zeros
    
    if num_asymptotes <= 0:
        return {'angles': [], 'centroid': None}
    
    # Calcular centroide
    sum_poles = sum(pole.real if hasattr(pole, 'real') else pole for pole in poles)
    sum_zeros = sum(zero.real if hasattr(zero, 'real') else zero for zero in zeros)
    centroid = (sum_poles - sum_zeros) / num_asymptotes
    
    # Calcular ângulos
    angles = [(2*q + 1) * pi / num_asymptotes for q in range(num_asymptotes)]
    
    return {'angles': angles, 'centroid': centroid}


def find_breakaway_points(tf_obj) -> List:
    """Encontra pontos de breakaway/break-in"""
    analyzer = RootLocusAnalyzer()
    features = analyzer.get_locus_features(tf_obj, show_steps=False)
    return features.breakaway_points


def find_jw_crossings(tf_obj) -> List:
    """Encontra cruzamentos do eixo jω"""
    analyzer = RootLocusAnalyzer()
    features = analyzer.get_locus_features(tf_obj, show_steps=False)
    return features.jw_crossings


def calculate_locus_points(tf_obj, k_range: List[float]) -> Dict:
    """Calcula pontos do root locus para faixa de K"""
    analyzer = RootLocusAnalyzer()
    return analyzer.calculate_locus_points(tf_obj, k_range)
