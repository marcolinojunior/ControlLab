"""
ControlLab - Resposta em Frequência de Sistemas Discretos
========================================================

Este módulo implementa análise de resposta em frequência para sistemas discretos,
incluindo diagramas de Bode discretos e análise de estabilidade.

Características:
- Diagramas de Bode para sistemas discretos
- Transformação de frequências s->z
- Margens de estabilidade em frequência
- Análise de aliasing e efeitos de amostragem
"""

import sympy as sp
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass, field
from ..core.symbolic_tf import SymbolicTransferFunction
from ..core.history import OperationHistory, OperationStep

@dataclass
class DiscreteFrequencyResult:
    """
    Resultado da análise de frequência discreta
    
    Atributos:
        system: Sistema analisado
        frequency_range: Faixa de frequências analisada
        magnitude_response: Resposta em magnitude
        phase_response: Resposta em fase
        gain_margin: Margem de ganho
        phase_margin: Margem de fase
        crossover_frequencies: Frequências de cruzamento
        stability_analysis: Análise de estabilidade
        nyquist_frequency: Frequência de Nyquist
    """
    system: SymbolicTransferFunction = None
    frequency_range: List[float] = field(default_factory=list)
    magnitude_response: List[float] = field(default_factory=list)
    phase_response: List[float] = field(default_factory=list)
    gain_margin: Optional[float] = None
    phase_margin: Optional[float] = None
    crossover_frequencies: Dict[str, float] = field(default_factory=dict)
    stability_analysis: Dict[str, Any] = field(default_factory=dict)
    nyquist_frequency: float = 0.0
    sampling_time: float = 0.1
    history: OperationHistory = field(default_factory=OperationHistory)

class DiscreteFrequencyAnalyzer:
    """
    Analisador de resposta em frequência para sistemas discretos
    
    Implementa análise completa de frequência:
    - Diagramas de Bode discretos
    - Cálculo de margens de estabilidade
    - Análise de efeitos de amostragem
    """
    
    def __init__(self, sampling_time: float = 0.1):
        """
        Inicializa o analisador de frequência
        
        Args:
            sampling_time: Período de amostragem
        """
        self.T = sampling_time
        self.z = sp.Symbol('z')
        self.w = sp.Symbol('w', real=True)  # Frequência digital
        self.history = OperationHistory()
        
        # Frequência de Nyquist
        self.w_nyquist = np.pi / self.T
    
    def bode_analysis(self, discrete_tf: SymbolicTransferFunction,
                     frequency_range: Optional[Tuple[float, float]] = None,
                     num_points: int = 100,
                     show_steps: bool = True) -> DiscreteFrequencyResult:
        """
        Análise de Bode para sistemas discretos
        
        Args:
            discrete_tf: Função de transferência discreta
            frequency_range: Faixa de frequências (rad/s)
            num_points: Número de pontos a calcular
            show_steps: Se deve mostrar os passos
        
        Returns:
            DiscreteFrequencyResult: Resultado da análise
        """
        if show_steps:
            print("🔄 DIAGRAMA DE BODE - SISTEMA DISCRETO")
            print("=" * 40)
            print(f"📊 Sistema: H(z) = {discrete_tf.num}/{discrete_tf.den}")
            print(f"⏱️  Período de amostragem: T = {self.T}")
            print(f"🎯 Frequência de Nyquist: {self.w_nyquist:.2f} rad/s")
        
        result = DiscreteFrequencyResult()
        result.system = discrete_tf
        result.sampling_time = self.T
        result.nyquist_frequency = self.w_nyquist
        
        try:
            # Definir faixa de frequências
            if frequency_range is None:
                # Faixa padrão: 0.01 a 0.9 * frequência de Nyquist
                w_min = 0.01
                w_max = 0.9 * self.w_nyquist
            else:
                w_min, w_max = frequency_range
                w_max = min(w_max, 0.99 * self.w_nyquist)  # Limitar à Nyquist
            
            if show_steps:
                print(f"   📈 Faixa de análise: {w_min:.3f} a {w_max:.2f} rad/s")
            
            # Gerar frequências
            frequencies = np.logspace(np.log10(w_min), np.log10(w_max), num_points)
            result.frequency_range = frequencies.tolist()
            
            # Calcular resposta em frequência
            # H(e^(jwT)) = H(z)|_{z=e^(jwT)}
            magnitude_db = []
            phase_deg = []
            
            if show_steps:
                print(f"   🧮 Calculando {num_points} pontos da resposta...")
            
            for w_freq in frequencies:
                try:
                    # Substituir z = e^(jwT)
                    z_val = sp.exp(sp.I * w_freq * self.T)
                    
                    # Calcular H(z) no ponto
                    H_num = discrete_tf.num.subs(self.z, z_val)
                    H_den = discrete_tf.den.subs(self.z, z_val)
                    
                    if H_den != 0:
                        H_val = H_num / H_den
                        
                        # Converter para complexo
                        H_complex = complex(H_val)
                        
                        # Magnitude em dB
                        magnitude = abs(H_complex)
                        magnitude_db.append(20 * np.log10(magnitude) if magnitude > 0 else -np.inf)
                        
                        # Fase em graus
                        phase = np.angle(H_complex) * 180 / np.pi
                        phase_deg.append(phase)
                    
                    else:
                        magnitude_db.append(-np.inf)
                        phase_deg.append(0)
                
                except Exception as e:
                    magnitude_db.append(-np.inf)
                    phase_deg.append(0)
                    if show_steps and len(magnitude_db) < 5:
                        print(f"   ⚠️  Erro em w={w_freq:.3f}: {e}")
            
            result.magnitude_response = magnitude_db
            result.phase_response = phase_deg
            
            # Calcular margens de estabilidade
            margins = self._calculate_stability_margins(frequencies, magnitude_db, phase_deg)
            result.gain_margin = margins.get('gain_margin')
            result.phase_margin = margins.get('phase_margin')
            result.crossover_frequencies = margins.get('crossover_frequencies', {})
            
            if show_steps:
                print(f"   📊 Resposta calculada:")
                print(f"       Margem de ganho: {result.gain_margin:.2f} dB" if result.gain_margin else "       Margem de ganho: Não definida")
                print(f"       Margem de fase: {result.phase_margin:.1f}°" if result.phase_margin else "       Margem de fase: Não definida")
                print("   ✅ Análise de Bode concluída!")
            
            # Análise de estabilidade
            result.stability_analysis = self._analyze_frequency_stability(result)
            
            # Adicionar ao histórico
            step = OperationStep(
                operation="bode_discreto",
                input_expr=f"{discrete_tf.num}/{discrete_tf.den}",
                output_expr=f"Margens: GM={result.gain_margin:.1f}dB, PM={result.phase_margin:.1f}°" if result.gain_margin and result.phase_margin else "Margens calculadas",
                explanation=f"Bode discreto, T={self.T}"
            )
            result.history.add_step(step)
            
        except Exception as e:
            error_msg = f"Erro na análise de Bode: {e}"
            if show_steps:
                print(f"❌ {error_msg}")
        
        return result
    
    def _calculate_stability_margins(self, frequencies: np.ndarray,
                                   magnitude_db: List[float],
                                   phase_deg: List[float]) -> Dict[str, Any]:
        """Calcula margens de ganho e fase"""
        
        margins = {
            'gain_margin': None,
            'phase_margin': None,
            'crossover_frequencies': {}
        }
        
        try:
            # Encontrar frequência de cruzamento de ganho (magnitude = 0 dB)
            gain_crossover_idx = None
            for i, mag in enumerate(magnitude_db):
                if mag is not None and not np.isinf(mag):
                    if i > 0 and magnitude_db[i-1] is not None:
                        # Procurar cruzamento por zero
                        if (magnitude_db[i-1] > 0 and mag < 0) or (magnitude_db[i-1] < 0 and mag > 0):
                            gain_crossover_idx = i
                            break
            
            if gain_crossover_idx is not None:
                w_gc = frequencies[gain_crossover_idx]
                margins['crossover_frequencies']['gain'] = w_gc
                
                # Margem de fase = 180° + fase na frequência de cruzamento de ganho
                phase_at_gc = phase_deg[gain_crossover_idx]
                margins['phase_margin'] = 180 + phase_at_gc
            
            # Encontrar frequência de cruzamento de fase (fase = -180°)
            phase_crossover_idx = None
            for i, phase in enumerate(phase_deg):
                if phase is not None:
                    if i > 0 and phase_deg[i-1] is not None:
                        # Procurar cruzamento por -180°
                        if (phase_deg[i-1] > -180 and phase < -180) or (phase_deg[i-1] < -180 and phase > -180):
                            phase_crossover_idx = i
                            break
            
            if phase_crossover_idx is not None:
                w_pc = frequencies[phase_crossover_idx]
                margins['crossover_frequencies']['phase'] = w_pc
                
                # Margem de ganho = -magnitude na frequência de cruzamento de fase
                mag_at_pc = magnitude_db[phase_crossover_idx]
                if mag_at_pc is not None and not np.isinf(mag_at_pc):
                    margins['gain_margin'] = -mag_at_pc
        
        except Exception:
            pass
        
        return margins
    
    def _analyze_frequency_stability(self, result: DiscreteFrequencyResult) -> Dict[str, Any]:
        """Analisa estabilidade baseada na resposta em frequência"""
        
        analysis = {
            'is_stable': False,
            'stability_confidence': 'Baixa',
            'recommendations': []
        }
        
        try:
            # Critérios básicos de estabilidade
            gm = result.gain_margin
            pm = result.phase_margin
            
            if gm is not None and pm is not None:
                # Sistema estável se GM > 0 dB e PM > 0°
                if gm > 0 and pm > 0:
                    analysis['is_stable'] = True
                    
                    # Avaliar qualidade das margens
                    if gm >= 6 and pm >= 30:
                        analysis['stability_confidence'] = 'Alta'
                        analysis['recommendations'].append("Margens adequadas - sistema bem projetado")
                    elif gm >= 3 and pm >= 20:
                        analysis['stability_confidence'] = 'Média'
                        analysis['recommendations'].append("Margens aceitáveis - considerar melhorias")
                    else:
                        analysis['stability_confidence'] = 'Baixa'
                        analysis['recommendations'].append("Margens baixas - sistema próximo à instabilidade")
                
                else:
                    analysis['is_stable'] = False
                    analysis['recommendations'].append("Sistema instável - margens negativas")
            
            else:
                analysis['recommendations'].append("Margens não calculáveis - verificar sistema")
            
            # Verificar efeitos de amostragem
            max_freq = max(result.frequency_range) if result.frequency_range else 0
            if max_freq > 0.5 * result.nyquist_frequency:
                analysis['recommendations'].append("Cuidado: análise próxima à frequência de Nyquist")
        
        except Exception:
            analysis['recommendations'].append("Erro na análise de estabilidade")
        
        return analysis
    
    def nyquist_analysis(self, discrete_tf: SymbolicTransferFunction,
                        show_steps: bool = True) -> Dict[str, Any]:
        """
        Análise de Nyquist para sistemas discretos
        
        Args:
            discrete_tf: Função de transferência discreta
            show_steps: Se deve mostrar os passos
        
        Returns:
            Dict com análise de Nyquist
        """
        if show_steps:
            print("🔄 DIAGRAMA DE NYQUIST - SISTEMA DISCRETO")
            print("=" * 42)
            print(f"📊 Sistema: H(z) = {discrete_tf.num}/{discrete_tf.den}")
            print(f"🎯 Contorno: Círculo unitário |z| = 1")
        
        nyquist_result = {
            'system': discrete_tf,
            'stability_assessment': 'Indeterminado',
            'encirclements': 0,
            'critical_point_analysis': {},
            'recommendations': []
        }
        
        try:
            # Para sistemas discretos, o critério de Nyquist usa o círculo unitário
            # em vez do semiplano direito
            
            if show_steps:
                print(f"   🧮 Calculando contorno no círculo unitário...")
            
            # Gerar pontos no círculo unitário
            angles = np.linspace(0, 2*np.pi, 360)
            nyquist_points = []
            
            for angle in angles:
                try:
                    z_val = sp.exp(sp.I * angle)
                    
                    H_num = discrete_tf.num.subs(self.z, z_val)
                    H_den = discrete_tf.den.subs(self.z, z_val)
                    
                    if H_den != 0:
                        H_val = H_num / H_den
                        H_complex = complex(H_val)
                        nyquist_points.append(H_complex)
                
                except Exception:
                    continue
            
            if nyquist_points:
                # Analisar envolvimentos do ponto (-1, 0)
                # Simplificação: verificar se curva passa próxima a (-1, 0)
                critical_distances = [abs(point + 1) for point in nyquist_points]
                min_distance = min(critical_distances)
                
                nyquist_result['critical_point_analysis'] = {
                    'min_distance_to_critical': min_distance,
                    'num_points_calculated': len(nyquist_points)
                }
                
                if show_steps:
                    print(f"   📊 {len(nyquist_points)} pontos calculados")
                    print(f"   📍 Distância mínima ao ponto crítico: {min_distance:.3f}")
                
                # Análise de estabilidade simplificada
                if min_distance > 0.5:
                    nyquist_result['stability_assessment'] = 'Provavelmente estável'
                    nyquist_result['recommendations'].append("Sistema aparenta ser estável")
                elif min_distance > 0.1:
                    nyquist_result['stability_assessment'] = 'Marginal'
                    nyquist_result['recommendations'].append("Sistema próximo à instabilidade")
                else:
                    nyquist_result['stability_assessment'] = 'Provavelmente instável'
                    nyquist_result['recommendations'].append("Sistema aparenta ser instável")
                
                if show_steps:
                    print(f"   📊 Avaliação: {nyquist_result['stability_assessment']}")
            
            else:
                nyquist_result['recommendations'].append("Erro no cálculo dos pontos de Nyquist")
        
        except Exception as e:
            error_msg = f"Erro na análise de Nyquist: {e}"
            nyquist_result['recommendations'].append(error_msg)
            if show_steps:
                print(f"❌ {error_msg}")
        
        return nyquist_result
    
    def aliasing_analysis(self, continuous_tf: SymbolicTransferFunction,
                         max_frequency: float = 100.0,
                         show_steps: bool = True) -> Dict[str, Any]:
        """
        Analisa efeitos de aliasing na discretização
        
        Args:
            continuous_tf: Sistema contínuo original
            max_frequency: Frequência máxima de interesse
            show_steps: Se deve mostrar os passos
        
        Returns:
            Dict com análise de aliasing
        """
        if show_steps:
            print("🔄 ANÁLISE DE ALIASING")
            print("=" * 22)
            print(f"📊 Sistema contínuo: H(s) = {continuous_tf.num}/{continuous_tf.den}")
            print(f"⏱️  Período de amostragem: T = {self.T}")
            print(f"🎯 Frequência de Nyquist: {self.w_nyquist:.2f} rad/s")
        
        aliasing_result = {
            'sampling_time': self.T,
            'nyquist_frequency': self.w_nyquist,
            'max_analyzed_frequency': max_frequency,
            'aliasing_risk': 'Baixo',
            'recommendations': []
        }
        
        try:
            # Verificar se frequência máxima excede Nyquist
            if max_frequency > self.w_nyquist:
                aliasing_result['aliasing_risk'] = 'Alto'
                aliasing_result['recommendations'].append(
                    f"Frequência máxima ({max_frequency:.1f} rad/s) excede Nyquist ({self.w_nyquist:.2f} rad/s)")
                aliasing_result['recommendations'].append("Considerar filtro anti-aliasing ou aumentar frequência de amostragem")
            
            elif max_frequency > 0.8 * self.w_nyquist:
                aliasing_result['aliasing_risk'] = 'Médio'
                aliasing_result['recommendations'].append("Frequência máxima próxima à Nyquist - monitorar aliasing")
            
            else:
                aliasing_result['aliasing_risk'] = 'Baixo'
                aliasing_result['recommendations'].append("Frequência de amostragem adequada")
            
            # Calcular frequência de amostragem recomendada
            recommended_fs = 2.5 * max_frequency  # Fator de segurança
            recommended_T = 2 * np.pi / recommended_fs
            
            aliasing_result['recommended_sampling_time'] = recommended_T
            aliasing_result['recommended_sampling_frequency'] = recommended_fs
            
            if show_steps:
                print(f"   📊 Análise de aliasing:")
                print(f"       Risco: {aliasing_result['aliasing_risk']}")
                print(f"       T recomendado: {recommended_T:.4f}s")
                print(f"       fs recomendada: {recommended_fs:.1f} rad/s")
            
            # Verificar resposta do sistema contínuo em altas frequências
            s = sp.Symbol('s')
            try:
                # Avaliar magnitude em frequência de Nyquist
                H_at_nyquist = continuous_tf.num.subs(s, sp.I * self.w_nyquist) / continuous_tf.den.subs(s, sp.I * self.w_nyquist)
                magnitude_at_nyquist = abs(complex(H_at_nyquist))
                
                if magnitude_at_nyquist > 0.01:  # -40 dB
                    aliasing_result['recommendations'].append("Sistema tem ganho significativo na frequência de Nyquist")
                
                aliasing_result['magnitude_at_nyquist'] = magnitude_at_nyquist
                
                if show_steps:
                    print(f"       |H(jωn)|: {magnitude_at_nyquist:.4f}")
            
            except Exception:
                aliasing_result['recommendations'].append("Não foi possível avaliar resposta em alta frequência")
        
        except Exception as e:
            error_msg = f"Erro na análise de aliasing: {e}"
            aliasing_result['recommendations'].append(error_msg)
            if show_steps:
                print(f"❌ {error_msg}")
        
        return aliasing_result

def analyze_discrete_frequency_response(discrete_tf: SymbolicTransferFunction,
                                      sampling_time: float = 0.1,
                                      frequency_range: Optional[Tuple[float, float]] = None,
                                      show_steps: bool = True) -> DiscreteFrequencyResult:
    """
    Função de conveniência para análise de frequência discreta
    
    Args:
        discrete_tf: Função de transferência discreta
        sampling_time: Período de amostragem
        frequency_range: Faixa de frequências (rad/s)
        show_steps: Se deve mostrar os passos
    
    Returns:
        DiscreteFrequencyResult: Resultado completo
    """
    analyzer = DiscreteFrequencyAnalyzer(sampling_time)
    return analyzer.bode_analysis(discrete_tf, frequency_range, 100, show_steps)

def compare_continuous_discrete_frequency(continuous_tf: SymbolicTransferFunction,
                                        discrete_tf: SymbolicTransferFunction,
                                        sampling_time: float = 0.1,
                                        show_steps: bool = True) -> Dict[str, Any]:
    """
    Compara resposta em frequência de sistemas contínuo e discreto
    
    Args:
        continuous_tf: Sistema contínuo
        discrete_tf: Sistema discreto
        sampling_time: Período de amostragem
        show_steps: Se deve mostrar os passos
    
    Returns:
        Dict com comparação
    """
    if show_steps:
        print("🔄 COMPARAÇÃO CONTÍNUO vs DISCRETO")
        print("=" * 37)
        print(f"📊 Contínuo: H(s) = {continuous_tf.num}/{continuous_tf.den}")
        print(f"📊 Discreto: H(z) = {discrete_tf.num}/{discrete_tf.den}")
        print(f"⏱️  T = {sampling_time}")
    
    analyzer = DiscreteFrequencyAnalyzer(sampling_time)
    
    # Análise do sistema discreto
    discrete_result = analyzer.bode_analysis(discrete_tf, None, 100, False)
    
    # Análise de aliasing
    aliasing_result = analyzer.aliasing_analysis(continuous_tf, 100.0, False)
    
    comparison = {
        'continuous_system': continuous_tf,
        'discrete_system': discrete_tf,
        'discrete_analysis': discrete_result,
        'aliasing_analysis': aliasing_result,
        'recommendations': []
    }
    
    # Recomendações baseadas na comparação
    if discrete_result.gain_margin and discrete_result.phase_margin:
        if discrete_result.gain_margin < 6 or discrete_result.phase_margin < 30:
            comparison['recommendations'].append("Margens de estabilidade baixas no sistema discreto")
    
    if aliasing_result['aliasing_risk'] != 'Baixo':
        comparison['recommendations'].append("Risco de aliasing detectado")
    
    if show_steps:
        print(f"\n📊 RESUMO DA COMPARAÇÃO:")
        if discrete_result.gain_margin and discrete_result.phase_margin:
            print(f"   Margens discretas - GM: {discrete_result.gain_margin:.1f}dB, PM: {discrete_result.phase_margin:.1f}°")
        print(f"   Risco de aliasing: {aliasing_result['aliasing_risk']}")
        
        if comparison['recommendations']:
            print("   💡 Recomendações:")
            for rec in comparison['recommendations']:
                print(f"       • {rec}")
    
    return comparison
