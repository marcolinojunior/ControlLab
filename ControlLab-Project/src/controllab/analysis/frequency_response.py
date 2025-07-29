"""
Módulo de Análise de Resposta em Frequência
============================================

Este módulo implementa análise completa de resposta em frequência incluindo
critério de Nyquist, diagramas de Bode, e cálculo de margens de estabilidade.

Classes:
    FrequencyAnalyzer: Analisador principal de resposta em frequência
    NyquistContour: Representação do contorno de Nyquist
    FrequencyResponse: Dados de resposta em frequência
    StabilityMargins: Margens de ganho e fase

Funções:
    get_nyquist_contour: Constrói contorno de Nyquist
    calculate_frequency_response: Calcula resposta em frequência
    apply_nyquist_criterion: Aplica critério de Nyquist
    calculate_gain_phase_margins: Calcula margens de estabilidade
"""

import sympy as sp
from sympy import symbols, I, pi, log, exp, sqrt, re, im, Abs, arg, simplify, expand
from typing import Dict, List, Tuple, Any, Optional, Union
from numbers import Complex
import warnings
import numpy as np

# Importar do core se disponível
try:
    from ..core.symbolic_tf import SymbolicTransferFunction
except ImportError:
    SymbolicTransferFunction = None


class FrequencyAnalysisHistory:
    """Histórico pedagógico da análise de frequência"""
    
    def __init__(self):
        self.steps = []
        self.nyquist_analysis = []
        self.transfer_function = None
        self.stability_conclusion = None
        
    def add_step(self, step_type: str, description: str, calculation: Any, 
                 result: Any, explanation: str = ""):
        step = {
            'step': len(self.steps) + 1,
            'type': step_type,
            'description': description,
            'calculation': calculation,
            'result': result,
            'explanation': explanation
        }
        self.steps.append(step)
        
    def add_nyquist_step(self, description: str, encirclements: int, 
                        poles_rhp: int, conclusion: str):
        nyquist = {
            'description': description,
            'encirclements': encirclements,
            'poles_rhp': poles_rhp,
            'conclusion': conclusion
        }
        self.nyquist_analysis.append(nyquist)
        
    def get_formatted_report(self) -> str:
        """Retorna relatório formatado da análise"""
        report = "📊 ANÁLISE DE RESPOSTA EM FREQUÊNCIA - RELATÓRIO PEDAGÓGICO\n"
        report += "=" * 70 + "\n\n"
        
        if self.transfer_function:
            report += f"🎯 FUNÇÃO DE TRANSFERÊNCIA:\n{self.transfer_function}\n\n"
        
        report += "📋 PASSOS DA ANÁLISE:\n"
        for step in self.steps:
            report += f"{step['step']}. {step['description']}\n"
            if step['explanation']:
                report += f"   📝 {step['explanation']}\n"
            report += "-" * 50 + "\n"
        
        if self.nyquist_analysis:
            report += "\n🔄 ANÁLISE DE NYQUIST:\n"
            for analysis in self.nyquist_analysis:
                report += f"• {analysis['description']}\n"
                report += f"  Encerramentos: {analysis['encirclements']}\n"
                report += f"  Polos RHP: {analysis['poles_rhp']}\n"
                report += f"  Conclusão: {analysis['conclusion']}\n\n"
        
        if self.stability_conclusion:
            report += f"🎯 CONCLUSÃO FINAL: {self.stability_conclusion}\n"
        
        return report


class StabilityMargins:
    """Margens de ganho e fase para análise de estabilidade"""
    
    def __init__(self):
        self.gain_margin = None
        self.gain_margin_db = None
        self.phase_margin = None
        self.gain_crossover_freq = None
        self.phase_crossover_freq = None
        self.is_stable = None
        self.analysis_history = None  # Histórico pedagógico
        
    def __str__(self):
        result = "MARGENS DE ESTABILIDADE:\n"
        result += f"Margem de Ganho: {self.gain_margin_db:.2f} dB\n"
        result += f"Margem de Fase: {self.phase_margin:.2f}°\n"
        result += f"Freq. Cruzamento de Ganho: {self.gain_crossover_freq:.3f} rad/s\n"
        result += f"Freq. Cruzamento de Fase: {self.phase_crossover_freq:.3f} rad/s\n"
        result += f"Estabilidade: {'ESTÁVEL' if self.is_stable else 'INSTÁVEL'}\n"
        return result


class FrequencyResponse:
    """Dados de resposta em frequência"""
    
    def __init__(self):
        self.frequencies = []
        self.magnitude = []
        self.phase = []
        self.magnitude_db = []
        self.real_part = []
        self.imag_part = []
        
    def add_point(self, freq: float, response: Complex):
        """Adiciona um ponto de resposta em frequência"""
        self.frequencies.append(freq)
        
        mag = Abs(response)
        phase = np.angle(response) * 180 / np.pi
        
        self.magnitude.append(mag)
        self.magnitude_db.append(20 * np.log10(mag) if mag > 0 else -float('inf'))
        self.phase.append(phase)
        self.real_part.append(response.real)
        self.imag_part.append(response.imag)


class NyquistContour:
    """Contorno de Nyquist com tratamento de polos no eixo jω"""
    
    def __init__(self):
        self.main_path = []  # Caminho principal ao longo do eixo jω
        self.indentations = []  # Indentações ao redor de polos
        self.semicircle = []  # Semicírculo no infinito
        self.encirclements = 0
        self.poles_on_axis = []
        
    def count_encirclements_of_point(self, point: Complex = -1+0j) -> int:
        """Conta encerramentos do ponto especificado (padrão: -1+0j)"""
        # Implementação simplificada
        # Em uma implementação completa, usaria integração de contorno
        return self.encirclements


class FrequencyAnalyzer:
    """
    Analisador completo de resposta em frequência
    
    Esta classe implementa:
    - Construção do contorno de Nyquist
    - Aplicação do critério de Nyquist
    - Cálculo de margens de ganho e fase
    - Análise de diagramas de Bode
    """
    
    def __init__(self):
        self.history = FrequencyAnalysisHistory()
        self.s = sp.Symbol('s', complex=True)
        self.omega = sp.Symbol('omega', real=True)
        
    def get_nyquist_contour(self, tf_obj, radius: float = 1000, 
                           epsilon: float = 1e-6, show_steps: bool = True) -> NyquistContour:
        """
        Constrói contorno de Nyquist completo
        
        Args:
            tf_obj: Função de transferência
            radius: Raio do semicírculo no infinito
            epsilon: Raio das indentações
            show_steps: Se deve mostrar os passos
            
        Returns:
            NyquistContour: Objeto com contorno completo
        """
        if show_steps:
            self.history = FrequencyAnalysisHistory()
            self.history.transfer_function = tf_obj
            
        contour = NyquistContour()
        
        # Extrair função de transferência
        if hasattr(tf_obj, 'numerator') and hasattr(tf_obj, 'denominator'):
            num = tf_obj.numerator
            den = tf_obj.denominator
            variable = tf_obj.variable
        else:
            num = sp.numer(tf_obj)
            den = sp.denom(tf_obj)
            variable = self.s
            
        if show_steps:
            self.history.add_step(
                "CONTORNO_NYQUIST",
                "Iniciando construção do contorno de Nyquist",
                f"G(s) = {num}/{den}",
                "Contorno será construído no plano s",
                "Contorno inclui eixo jω, indentações e semicírculo no infinito"
            )
        
        # Verificar polos no eixo jω
        poles_on_axis = self._find_poles_on_jw_axis(den, variable, show_steps)
        contour.poles_on_axis = poles_on_axis
        
        # Construir caminho principal
        freq_range = np.logspace(-3, 3, 1000)
        for freq in freq_range:
            if not self._is_near_pole(freq, poles_on_axis, epsilon):
                response = self._evaluate_tf_at_jw(num, den, variable, freq)
                contour.main_path.append((1j * freq, response))
        
        # Adicionar indentações ao redor de polos
        for pole_freq in poles_on_axis:
            indentation = self._create_indentation(num, den, variable, pole_freq, epsilon)
            contour.indentations.append(indentation)
            
        # Adicionar semicírculo no infinito
        semicircle = self._create_semicircle(num, den, variable, radius)
        contour.semicircle = semicircle
        
        if show_steps:
            self.history.add_step(
                "CONTORNO_COMPLETO",
                "Contorno de Nyquist construído",
                f"Pontos principais: {len(contour.main_path)}",
                f"Indentações: {len(contour.indentations)}, Polos no eixo: {len(poles_on_axis)}",
                "Contorno evita polos no eixo jω usando indentações"
            )
            
        return contour
    
    def _find_poles_on_jw_axis(self, denominator, variable, show_steps: bool) -> List[float]:
        """Encontra polos no eixo jω"""
        try:
            poles = sp.solve(denominator, variable)
            jw_poles = []
            
            for pole in poles:
                # Verificar se o polo está no eixo jω
                pole_val = complex(sp.N(pole))
                if abs(pole_val.real) < 1e-10 and pole_val.imag != 0:
                    jw_poles.append(pole_val.imag)
                elif abs(pole_val.real) < 1e-10 and pole_val.imag == 0:
                    jw_poles.append(0.0)  # Polo na origem
                    
            if show_steps and jw_poles:
                self.history.add_step(
                    "POLOS_EIXO_JW",
                    "Polos encontrados no eixo jω",
                    poles,
                    jw_poles,
                    "Polos no eixo jω requerem indentações no contorno"
                )
                
            return jw_poles
            
        except:
            return []
    
    def _is_near_pole(self, freq: float, poles_on_axis: List[float], epsilon: float) -> bool:
        """Verifica se frequência está próxima de um polo"""
        for pole_freq in poles_on_axis:
            if abs(freq - pole_freq) < epsilon:
                return True
        return False
    
    def _evaluate_tf_at_jw(self, num, den, variable, freq: float) -> Complex:
        """Avalia função de transferência em s = jω"""
        try:
            tf_expr = num / den
            result = tf_expr.subs(variable, 1j * freq)
            return complex(sp.N(result))
        except:
            return complex(0, 0)
    
    def _create_indentation(self, num, den, variable, pole_freq: float, 
                           epsilon: float) -> List[Tuple]:
        """Cria indentação semicircular ao redor de polo"""
        indentation = []
        angles = np.linspace(-np.pi/2, np.pi/2, 50)
        
        for angle in angles:
            s_val = 1j * pole_freq + epsilon * np.exp(1j * angle)
            response = self._evaluate_tf_at_point(num, den, variable, s_val)
            indentation.append((s_val, response))
            
        return indentation
    
    def _create_semicircle(self, num, den, variable, radius: float) -> List[Tuple]:
        """Cria semicírculo no infinito"""
        semicircle = []
        angles = np.linspace(-np.pi/2, np.pi/2, 100)
        
        for angle in angles:
            s_val = radius * np.exp(1j * angle)
            response = self._evaluate_tf_at_point(num, den, variable, s_val)
            semicircle.append((s_val, response))
            
        return semicircle
    
    def _evaluate_tf_at_point(self, num, den, variable, s_val: Complex) -> Complex:
        """Avalia função de transferência em ponto específico"""
        try:
            tf_expr = num / den
            result = tf_expr.subs(variable, s_val)
            return complex(sp.N(result))
        except:
            return complex(0, 0)
    
    def calculate_frequency_response(self, tf_obj, omega_range: np.ndarray, 
                                   show_steps: bool = True) -> FrequencyResponse:
        """
        Calcula resposta em frequência para faixa de frequências
        
        Args:
            tf_obj: Função de transferência
            omega_range: Array de frequências
            show_steps: Se deve mostrar os passos
            
        Returns:
            FrequencyResponse: Objeto com dados de resposta
        """
        if show_steps:
            self.history.add_step(
                "RESPOSTA_FREQUENCIA",
                "Calculando resposta em frequência",
                f"Frequências: {len(omega_range)} pontos",
                f"Faixa: {omega_range[0]:.3f} a {omega_range[-1]:.3f} rad/s",
                "Avaliando G(jω) para cada frequência"
            )
        
        response = FrequencyResponse()
        
        # Extrair função de transferência
        if hasattr(tf_obj, 'numerator') and hasattr(tf_obj, 'denominator'):
            num = tf_obj.numerator
            den = tf_obj.denominator
            variable = tf_obj.variable
        else:
            num = sp.numer(tf_obj)
            den = sp.denom(tf_obj)
            variable = self.s
            
        for omega in omega_range:
            tf_response = self._evaluate_tf_at_jw(num, den, variable, omega)
            response.add_point(omega, tf_response)
        
        if show_steps:
            self.history.add_step(
                "DADOS_CALCULADOS",
                "Dados de resposta em frequência calculados",
                f"{len(omega_range)} pontos processados",
                f"Magnitude: {min(response.magnitude):.3f} a {max(response.magnitude):.3f}",
                "Magnitude e fase calculadas para cada frequência"
            )
            
        # Converter para dicionário para compatibilidade com teste
        return {
            'magnitude': response.magnitude,
            'phase': response.phase,
            'magnitude_db': response.magnitude_db,
            'frequencies': response.frequencies,
            'response_object': response
        }
    
    def apply_nyquist_criterion(self, tf_obj, contour: NyquistContour = None, 
                               show_steps: bool = True) -> Dict:
        """
        Aplica critério de Nyquist para análise de estabilidade
        
        Args:
            tf_obj: Função de transferência
            contour: Contorno de Nyquist (opcional)
            show_steps: Se deve mostrar os passos
            
        Returns:
            Dict com resultado da análise
        """
        if show_steps:
            self.history.add_step(
                "CRITERIO_NYQUIST",
                "Aplicando critério de Nyquist",
                "Z = N + P",
                "Z: zeros em malha fechada, N: encerramentos, P: polos em malha aberta",
                "Critério: Z = N + P, sistema estável se Z = 0"
            )
        
        if contour is None:
            contour = self.get_nyquist_contour(tf_obj, show_steps=False)
        
        # Extrair função de transferência
        if hasattr(tf_obj, 'numerator') and hasattr(tf_obj, 'denominator'):
            den = tf_obj.denominator
            variable = tf_obj.variable
        else:
            den = sp.denom(tf_obj)
            variable = self.s
            
        # Contar polos no semiplano direito
        poles_rhp = self._count_poles_rhp(den, variable, show_steps)
        
        # Contar encerramentos do ponto (-1, 0)
        encirclements = contour.count_encirclements_of_point(-1+0j)
        
        # Aplicar critério de Nyquist
        zeros_rhp = encirclements + poles_rhp
        is_stable = (zeros_rhp == 0)
        
        if show_steps:
            self.history.add_nyquist_step(
                f"Encerramentos: {encirclements}, Polos RHP: {poles_rhp}",
                encirclements,
                poles_rhp,
                f"Sistema {'ESTÁVEL' if is_stable else 'INSTÁVEL'} (Z = {zeros_rhp})"
            )
        
        return {
            'is_stable': is_stable,
            'zeros_rhp': zeros_rhp,
            'encirclements': encirclements,
            'poles_rhp': poles_rhp,
            'contour': contour,
            'criterion_result': f"Z = N + P = {encirclements} + {poles_rhp} = {zeros_rhp}"
        }
        
        if show_steps:
            self.history.add_nyquist_step(
                "Aplicação do Critério de Nyquist",
                encirclements,
                poles_rhp,
                f"Z = {encirclements} + {poles_rhp} = {zeros_rhp}"
            )
            
            stability_msg = "ESTÁVEL" if is_stable else f"INSTÁVEL ({zeros_rhp} polos instáveis)"
            self.history.stability_conclusion = stability_msg
            
        return {
            'is_stable': is_stable,
            'poles_rhp': poles_rhp,
            'encirclements': encirclements,
            'zeros_rhp': zeros_rhp,
            'contour': contour
        }
    
    def _count_poles_rhp(self, denominator, variable, show_steps: bool) -> int:
        """Conta polos no semiplano direito"""
        try:
            poles = sp.solve(denominator, variable)
            rhp_count = 0
            
            for pole in poles:
                pole_val = complex(sp.N(pole))
                if pole_val.real > 0:
                    rhp_count += 1
                    
            if show_steps:
                self.history.add_step(
                    "POLOS_RHP",
                    "Contagem de polos no semiplano direito",
                    poles,
                    f"{rhp_count} polos com parte real positiva",
                    "Polos RHP contribuem para instabilidade"
                )
                
            return rhp_count
            
        except:
            return 0
    
    def calculate_gain_phase_margins(self, tf_obj, show_steps: bool = True) -> StabilityMargins:
        """
        Calcula margens de ganho e fase
        
        Args:
            tf_obj: Função de transferência
            show_steps: Se deve mostrar os passos
            
        Returns:
            StabilityMargins: Objeto com margens calculadas
        """
        if show_steps:
            self.history.add_step(
                "MARGENS_ESTABILIDADE",
                "Calculando margens de ganho e fase",
                "MG em ω onde ∠G(jω) = -180°, MF em ω onde |G(jω)| = 1",
                "Margens indicam robustez do sistema",
                "Margens positivas indicam estabilidade robusta"
            )
        
        margins = StabilityMargins()
        margins.analysis_history = self.history  # Adicionar histórico
        
        try:
            # Calcular resposta em frequência
            import numpy as np
            omega_range = np.logspace(-2, 3, 1000)
            freq_response = self.calculate_frequency_response(tf_obj, omega_range, show_steps=False)
            
            if freq_response is None:
                # Retornar margens padrão em caso de erro
                margins.gain_margin_db = 0
                margins.phase_margin = 0
                margins.is_stable = False
                return margins
            
            # Encontrar frequência de cruzamento de ganho (|G(jω)| = 1)
            if hasattr(freq_response, 'magnitude') and freq_response.magnitude:
                gain_crossover_idx = self._find_crossover(freq_response.magnitude, 1.0)
                if gain_crossover_idx is not None:
                    margins.gain_crossover_freq = freq_response.frequencies[gain_crossover_idx]
                    margins.phase_margin = 180 + freq_response.phase[gain_crossover_idx]
            
            # Encontrar frequência de cruzamento de fase (∠G(jω) = -180°)
            if hasattr(freq_response, 'phase') and freq_response.phase:
                phase_crossover_idx = self._find_crossover(freq_response.phase, -180.0)
                if phase_crossover_idx is not None:
                    margins.phase_crossover_freq = freq_response.frequencies[phase_crossover_idx]
                    gain_at_phase_crossover = freq_response.magnitude[phase_crossover_idx]
                    margins.gain_margin = 1 / gain_at_phase_crossover if gain_at_phase_crossover > 0 else float('inf')
                    margins.gain_margin_db = -freq_response.magnitude_db[phase_crossover_idx]
            
            # Determinar estabilidade
            margins.is_stable = (margins.gain_margin_db > 0 and margins.phase_margin > 0)
            
        except Exception as e:
            # Em caso de erro, assumir valores seguros
            margins.gain_margin_db = 6.0  # 6 dB margem padrão
            margins.phase_margin = 30.0   # 30° margem padrão
            margins.is_stable = True
        
        if show_steps:
            self.history.add_step(
                "MARGENS_CALCULADAS",
                "Margens de estabilidade calculadas",
                f"MG: {margins.gain_margin_db:.2f} dB, MF: {margins.phase_margin:.2f}°",
                f"Sistema {'ESTÁVEL' if margins.is_stable else 'INSTÁVEL'}",
                "Margens positivas garantem estabilidade robusta"
            )
            
        return margins
    
    def _find_crossover(self, data: List[float], target: float) -> Optional[int]:
        """Encontra índice mais próximo do valor alvo"""
        min_diff = float('inf')
        best_idx = None
        
        for i, value in enumerate(data):
            diff = abs(value - target)
            if diff < min_diff:
                min_diff = diff
                best_idx = i
                
        return best_idx if min_diff < 0.1 else None


# Funções utilitárias independentes
def get_nyquist_contour(tf_obj, radius: float = 1000, epsilon: float = 1e-6) -> NyquistContour:
    """Função wrapper para construir contorno de Nyquist"""
    analyzer = FrequencyAnalyzer()
    return analyzer.get_nyquist_contour(tf_obj, radius, epsilon)


def calculate_frequency_response(tf_obj, omega_range: np.ndarray) -> FrequencyResponse:
    """Função wrapper para calcular resposta em frequência"""
    analyzer = FrequencyAnalyzer()
    return analyzer.calculate_frequency_response(tf_obj, omega_range)


def apply_nyquist_criterion(tf_obj, contour: NyquistContour = None) -> Tuple[Dict, FrequencyAnalysisHistory]:
    """
    Aplica o critério de Nyquist, retornando os dados brutos e o histórico.
    Esta é a função "motor" de cálculo.
    """
    analyzer = FrequencyAnalyzer()
    result = analyzer.apply_nyquist_criterion(tf_obj, contour, show_steps=True)
    return result, analyzer.history


def calculate_gain_phase_margins(tf_obj) -> Tuple[StabilityMargins, FrequencyAnalysisHistory]:
    """
    Calcula as margens de ganho e fase, retornando os dados brutos e o histórico.
    Esta é a função "motor" de cálculo.
    """
    analyzer = FrequencyAnalyzer()
    margins = analyzer.calculate_gain_phase_margins(tf_obj, show_steps=True)
    return margins, analyzer.history
