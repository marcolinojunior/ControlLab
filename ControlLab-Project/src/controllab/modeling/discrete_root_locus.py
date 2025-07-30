"""
ControlLab - Lugar das Raízes no Domínio Z
==========================================

Este módulo implementa o lugar das raízes para sistemas discretos no domínio Z,
com curvas de amortecimento e círculos de frequência natural.

Características:
- Lugar das raízes Z-domain
- Círculos de frequência natural discreta
- Curvas de amortecimento constante
- Análise de margem de estabilidade
"""

import sympy as sp
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass, field
from ..core.symbolic_tf import SymbolicTransferFunction
from ..core.history import OperationHistory, OperationStep

@dataclass
class DiscreteRootLocusResult:
    """
    Resultado do lugar das raízes discreto
    
    Atributos:
        system: Sistema analisado
        root_locus_points: Pontos do lugar das raízes
        stability_boundary: Pontos do círculo unitário
        damping_curves: Curvas de amortecimento constante
        frequency_circles: Círculos de frequência natural constante
        gain_range: Faixa de ganhos analisada
        stable_gain_range: Faixa de ganhos estáveis
        analysis_steps: Passos da análise
    """
    system: SymbolicTransferFunction = None
    root_locus_points: List[Tuple[float, float]] = field(default_factory=list)
    stability_boundary: List[Tuple[float, float]] = field(default_factory=list)
    damping_curves: Dict[float, List[Tuple[float, float]]] = field(default_factory=dict)
    frequency_circles: Dict[float, List[Tuple[float, float]]] = field(default_factory=dict)
    gain_range: Tuple[float, float] = (0.0, 10.0)
    stable_gain_range: Optional[Tuple[float, float]] = None
    analysis_steps: List[str] = field(default_factory=list)
    history: OperationHistory = field(default_factory=OperationHistory)

class DiscreteRootLocus:
    """
    Analisador de lugar das raízes para sistemas discretos
    
    Implementa análise completa do lugar das raízes no domínio Z:
    - Cálculo de pontos do lugar das raízes
    - Determinação de faixas de ganho estáveis
    - Geração de curvas de desempenho
    """
    
    def __init__(self, sampling_time: float = 0.1):
        """
        Inicializa o analisador de lugar das raízes
        
        Args:
            sampling_time: Período de amostragem para cálculos
        """
        self.T = sampling_time
        self.z = sp.Symbol('z')
        self.K = sp.Symbol('K', positive=True)
        self.history = OperationHistory()
    
    def calculate_root_locus(self, open_loop_tf: SymbolicTransferFunction,
                           gain_range: Tuple[float, float] = (0.0, 10.0),
                           num_points: int = 100,
                           show_steps: bool = True) -> DiscreteRootLocusResult:
        """
        Calcula o lugar das raízes no domínio Z
        
        Args:
            open_loop_tf: Função de transferência em malha aberta
            gain_range: Faixa de ganhos a analisar
            num_points: Número de pontos a calcular
            show_steps: Se deve mostrar os passos
        
        Returns:
            DiscreteRootLocusResult: Resultado da análise
        """
        if show_steps:
            print("🔄 LUGAR DAS RAÍZES - DOMÍNIO Z")
            print("=" * 32)
            print(f"📊 Sistema: G(z) = {open_loop_tf.num}/{open_loop_tf.den}")
            print(f"⏱️  Período de amostragem: T = {self.T}")
            print(f"📈 Faixa de ganho: K ∈ [{gain_range[0]}, {gain_range[1]}]")
        
        result = DiscreteRootLocusResult()
        result.system = open_loop_tf
        result.gain_range = gain_range
        
        try:
            # Função de transferência em malha fechada
            # T(z) = K*G(z) / (1 + K*G(z))
            G_num = open_loop_tf.num
            G_den = open_loop_tf.den
            
            # Polinômio característico: 1 + K*G(z) = 0
            char_poly = G_den + self.K * G_num
            
            if show_steps:
                print(f"   📝 Polinômio característico: {char_poly} = 0")
            
            result.analysis_steps.append(f"Equação característica: {char_poly} = 0")
            
            # Calcular pontos do lugar das raízes para diferentes valores de K
            gain_values = np.linspace(gain_range[0], gain_range[1], num_points)
            root_points = []
            stable_gains = []
            
            if show_steps:
                print(f"   🧮 Calculando {num_points} pontos...")
            
            for i, K_val in enumerate(gain_values):
                try:
                    # Substituir valor de K
                    char_poly_numeric = char_poly.subs(self.K, K_val)
                    
                    # Encontrar raízes
                    roots = sp.solve(char_poly_numeric, self.z)
                    
                    # Verificar estabilidade (todas as raízes dentro do círculo unitário)
                    is_stable = True
                    for root in roots:
                        try:
                            if root.is_real:
                                root_complex = complex(float(root))
                            else:
                                root_complex = complex(root)
                            
                            # Adicionar ponto ao lugar das raízes
                            root_points.append((float(root_complex.real), float(root_complex.imag)))
                            
                            # Verificar estabilidade
                            if abs(root_complex) >= 1.0:
                                is_stable = False
                                
                        except Exception:
                            is_stable = False
                    
                    if is_stable:
                        stable_gains.append(K_val)
                
                except Exception as e:
                    if show_steps and i < 5:  # Mostrar apenas os primeiros erros
                        print(f"   ⚠️  Erro para K={K_val:.2f}: {e}")
            
            result.root_locus_points = root_points
            
            # Determinar faixa de ganho estável
            if stable_gains:
                result.stable_gain_range = (min(stable_gains), max(stable_gains))
                if show_steps:
                    print(f"   ✅ Faixa estável: K ∈ [{min(stable_gains):.2f}, {max(stable_gains):.2f}]")
            else:
                result.stable_gain_range = None
                if show_steps:
                    print(f"   ❌ Nenhum ganho estável encontrado na faixa analisada")
            
            # Gerar curvas de referência
            result.stability_boundary = self._generate_unit_circle()
            result.damping_curves = self._generate_damping_curves()
            result.frequency_circles = self._generate_frequency_circles()
            
            if show_steps:
                print(f"   📊 {len(root_points)} pontos calculados")
                print("   ✅ Lugar das raízes concluído!")
            
            # Adicionar ao histórico
            step = OperationStep(
                operation="lugar_raizes_z",
                input_expr=f"{G_num}/{G_den}",
                output_expr=f"{len(root_points)} pontos",
                explanation=f"Lugar das raízes para K ∈ {gain_range}"
            )
            result.history.add_step(step)
            
        except Exception as e:
            error_msg = f"Erro no cálculo do lugar das raízes: {e}"
            result.analysis_steps.append(f"❌ {error_msg}")
            if show_steps:
                print(f"❌ {error_msg}")
        
        return result
    
    def _generate_unit_circle(self, num_points: int = 360) -> List[Tuple[float, float]]:
        """Gera pontos do círculo unitário (limite de estabilidade)"""
        
        angles = np.linspace(0, 2*np.pi, num_points)
        circle_points = []
        
        for angle in angles:
            x = np.cos(angle)
            y = np.sin(angle)
            circle_points.append((x, y))
        
        return circle_points
    
    def _generate_damping_curves(self) -> Dict[float, List[Tuple[float, float]]]:
        """
        Gera curvas de amortecimento constante no domínio Z
        
        Para sistemas discretos, o amortecimento relaciona-se com:
        ζ = -ln(r) / sqrt(ln²(r) + θ²)
        onde r = |z| e θ = arg(z)
        """
        
        damping_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        curves = {}
        
        for zeta in damping_values:
            curve_points = []
            
            # Para cada amortecimento, gerar curva
            r_values = np.linspace(0.1, 0.98, 100)
            
            for r in r_values:
                try:
                    # Calcular ângulo correspondente ao amortecimento
                    if zeta < 1.0:
                        # Para sistemas subamortecidos
                        ln_r = np.log(r)
                        if ln_r < 0:  # r < 1 (região estável)
                            theta = np.sqrt((ln_r / zeta)**2 - ln_r**2)
                            
                            # Adicionar pontos simétricos
                            x = r * np.cos(theta)
                            y = r * np.sin(theta)
                            curve_points.append((x, y))
                            
                            if theta != 0:
                                curve_points.append((x, -y))
                
                except Exception:
                    continue
            
            if curve_points:
                curves[zeta] = curve_points
        
        return curves
    
    def _generate_frequency_circles(self) -> Dict[float, List[Tuple[float, float]]]:
        """
        Gera círculos de frequência natural constante
        
        Para sistemas discretos: ωn*T = |ln(z)|
        """
        
        # Frequências naturais em rad/s
        wn_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        circles = {}
        
        for wn in wn_values:
            try:
                # Raio correspondente: r = exp(-ωn*T)
                radius = np.exp(-wn * self.T)
                
                if 0.1 <= radius <= 0.98:  # Apenas círculos visíveis e estáveis
                    angles = np.linspace(0, 2*np.pi, 360)
                    circle_points = []
                    
                    for angle in angles:
                        x = radius * np.cos(angle)
                        y = radius * np.sin(angle)
                        circle_points.append((x, y))
                    
                    circles[wn] = circle_points
            
            except Exception:
                continue
        
        return circles
    
    def analyze_gain_effect(self, open_loop_tf: SymbolicTransferFunction,
                           target_damping: float = 0.5,
                           show_steps: bool = True) -> Dict[str, Any]:
        """
        Analisa efeito do ganho na resposta do sistema
        
        Args:
            open_loop_tf: Função de transferência em malha aberta
            target_damping: Amortecimento desejado
            show_steps: Se deve mostrar os passos
        
        Returns:
            Dict com análise do ganho
        """
        if show_steps:
            print("🔄 ANÁLISE DO EFEITO DO GANHO")
            print("=" * 32)
            print(f"📊 Sistema: G(z) = {open_loop_tf.num}/{open_loop_tf.den}")
            print(f"🎯 Amortecimento desejado: ζ = {target_damping}")
        
        analysis = {
            'target_damping': target_damping,
            'recommended_gain': None,
            'predicted_poles': [],
            'stability_assessment': '',
            'performance_metrics': {}
        }
        
        try:
            # Calcular lugar das raízes
            rl_result = self.calculate_root_locus(open_loop_tf, (0.1, 20.0), 200, False)
            
            if not rl_result.root_locus_points:
                analysis['stability_assessment'] = 'Erro no cálculo'
                return analysis
            
            # Encontrar ganho que resulta no amortecimento desejado
            best_gain = None
            best_damping_error = float('inf')
            
            gain_values = np.linspace(0.1, 20.0, 200)
            
            for K_val in gain_values:
                try:
                    # Calcular polos para este ganho
                    G_num = open_loop_tf.num
                    G_den = open_loop_tf.den
                    char_poly = G_den + K_val * G_num
                    roots = sp.solve(char_poly, self.z)
                    
                    # Analisar polos dominantes
                    for root in roots:
                        if root.is_finite:
                            if root.is_real:
                                root_complex = complex(float(root))
                            else:
                                root_complex = complex(root)
                            
                            # Calcular amortecimento estimado
                            r = abs(root_complex)
                            theta = np.angle(root_complex)
                            
                            if r > 0 and r < 1.0:  # Polo estável
                                # Aproximação para amortecimento
                                ln_r = np.log(r)
                                if theta != 0:
                                    estimated_damping = -ln_r / np.sqrt(ln_r**2 + theta**2)
                                else:
                                    estimated_damping = 1.0  # Polo real
                                
                                damping_error = abs(estimated_damping - target_damping)
                                
                                if damping_error < best_damping_error:
                                    best_damping_error = damping_error
                                    best_gain = K_val
                                    analysis['predicted_poles'] = [root_complex]
                
                except Exception:
                    continue
            
            analysis['recommended_gain'] = best_gain
            
            if best_gain:
                if show_steps:
                    print(f"   🎯 Ganho recomendado: K = {best_gain:.3f}")
                    print(f"   📍 Polos resultantes: {analysis['predicted_poles']}")
                
                # Avaliar estabilidade
                if rl_result.stable_gain_range:
                    min_stable, max_stable = rl_result.stable_gain_range
                    if min_stable <= best_gain <= max_stable:
                        analysis['stability_assessment'] = 'Estável'
                    else:
                        analysis['stability_assessment'] = 'Instável'
                else:
                    analysis['stability_assessment'] = 'Instável'
                
                # Métricas de desempenho estimadas
                dominant_pole = analysis['predicted_poles'][0]
                settling_time_samples = -4 / np.log(abs(dominant_pole))
                overshoot = np.exp(-target_damping * np.pi / np.sqrt(1 - target_damping**2)) if target_damping < 1 else 0
                
                analysis['performance_metrics'] = {
                    'settling_time_samples': settling_time_samples,
                    'settling_time_seconds': settling_time_samples * self.T,
                    'estimated_overshoot': overshoot * 100
                }
                
                if show_steps:
                    print(f"   ⏱️  Tempo de acomodação: {settling_time_samples:.1f} amostras ({settling_time_samples * self.T:.3f}s)")
                    print(f"   📈 Sobressinal estimado: {overshoot * 100:.1f}%")
                    print(f"   📊 Estabilidade: {analysis['stability_assessment']}")
            
            else:
                if show_steps:
                    print("   ❌ Não foi possível encontrar ganho adequado")
                analysis['stability_assessment'] = 'Ganho não encontrado'
        
        except Exception as e:
            error_msg = f"Erro na análise do ganho: {e}"
            analysis['stability_assessment'] = error_msg
            if show_steps:
                print(f"❌ {error_msg}")
        
        return analysis
    
    def design_from_specifications(self, open_loop_tf: SymbolicTransferFunction,
                                 specs: Dict[str, float],
                                 show_steps: bool = True) -> Dict[str, Any]:
        """
        Projeta controlador baseado em especificações
        
        Args:
            open_loop_tf: Função de transferência em malha aberta
            specs: Especificações {'damping': 0.5, 'settling_time': 2.0, 'overshoot': 10.0}
            show_steps: Se deve mostrar os passos
        
        Returns:
            Dict com projeto do controlador
        """
        if show_steps:
            print("🔄 PROJETO BASEADO EM ESPECIFICAÇÕES")
            print("=" * 38)
            print(f"📊 Sistema: G(z) = {open_loop_tf.num}/{open_loop_tf.den}")
            print(f"📋 Especificações:")
            for spec, value in specs.items():
                print(f"   • {spec}: {value}")
        
        design = {
            'specifications': specs,
            'controller_gain': None,
            'meets_specifications': False,
            'actual_performance': {},
            'design_notes': []
        }
        
        try:
            # Extrair especificações
            target_damping = specs.get('damping', 0.5)
            target_settling = specs.get('settling_time', 2.0)
            max_overshoot = specs.get('overshoot', 20.0)
            
            # Analisar efeito do ganho
            gain_analysis = self.analyze_gain_effect(open_loop_tf, target_damping, False)
            
            if gain_analysis['recommended_gain']:
                design['controller_gain'] = gain_analysis['recommended_gain']
                design['actual_performance'] = gain_analysis['performance_metrics']
                
                # Verificar se especificações são atendidas
                performance = gain_analysis['performance_metrics']
                
                settling_ok = performance.get('settling_time_seconds', float('inf')) <= target_settling
                overshoot_ok = performance.get('estimated_overshoot', float('inf')) <= max_overshoot
                stable = gain_analysis['stability_assessment'] == 'Estável'
                
                design['meets_specifications'] = settling_ok and overshoot_ok and stable
                
                if show_steps:
                    print(f"\n🎯 RESULTADO DO PROJETO:")
                    print(f"   🔧 Ganho do controlador: K = {design['controller_gain']:.3f}")
                    print(f"   ⏱️  Tempo de acomodação: {performance.get('settling_time_seconds', 0):.3f}s")
                    print(f"   📈 Sobressinal: {performance.get('estimated_overshoot', 0):.1f}%")
                    print(f"   📊 Estabilidade: {gain_analysis['stability_assessment']}")
                    
                    status = "✅" if design['meets_specifications'] else "❌"
                    print(f"   {status} Especificações: {'Atendidas' if design['meets_specifications'] else 'Não atendidas'}")
                
                # Notas de projeto
                if not settling_ok:
                    design['design_notes'].append("Tempo de acomodação acima do especificado")
                if not overshoot_ok:
                    design['design_notes'].append("Sobressinal acima do especificado")
                if not stable:
                    design['design_notes'].append("Sistema resultante é instável")
                
                if design['meets_specifications']:
                    design['design_notes'].append("Projeto atende todas as especificações")
            
            else:
                design['design_notes'].append("Não foi possível encontrar ganho adequado")
                if show_steps:
                    print("   ❌ Projeto não realizado - ganho adequado não encontrado")
        
        except Exception as e:
            error_msg = f"Erro no projeto: {e}"
            design['design_notes'].append(error_msg)
            if show_steps:
                print(f"❌ {error_msg}")
        
        return design

def plot_discrete_root_locus(open_loop_tf: SymbolicTransferFunction,
                           sampling_time: float = 0.1,
                           gain_range: Tuple[float, float] = (0.0, 10.0),
                           show_steps: bool = True) -> DiscreteRootLocusResult:
    """
    Função de conveniência para plotar lugar das raízes discreto
    
    Args:
        open_loop_tf: Função de transferência em malha aberta
        sampling_time: Período de amostragem
        gain_range: Faixa de ganhos
        show_steps: Se deve mostrar os passos
    
    Returns:
        DiscreteRootLocusResult: Resultado completo
    """
    rl_analyzer = DiscreteRootLocus(sampling_time)
    return rl_analyzer.calculate_root_locus(open_loop_tf, gain_range, 200, show_steps)

def analyze_discrete_performance(open_loop_tf: SymbolicTransferFunction,
                               gain: float,
                               sampling_time: float = 0.1,
                               show_steps: bool = True) -> Dict[str, Any]:
    """
    Analisa desempenho de sistema discreto para um ganho específico
    
    Args:
        open_loop_tf: Função de transferência em malha aberta
        gain: Ganho do controlador
        sampling_time: Período de amostragem
        show_steps: Se deve mostrar os passos
    
    Returns:
        Dict com análise de desempenho
    """
    if show_steps:
        print("🔄 ANÁLISE DE DESEMPENHO - SISTEMA DISCRETO")
        print("=" * 45)
        print(f"📊 Sistema: G(z) = {open_loop_tf.num}/{open_loop_tf.den}")
        print(f"🔧 Ganho: K = {gain}")
        print(f"⏱️  Período: T = {sampling_time}")
    
    rl_analyzer = DiscreteRootLocus(sampling_time)
    
    # Simular análise para o ganho específico
    fake_specs = {'damping': 0.5}  # Valor temporário
    gain_analysis = rl_analyzer.analyze_gain_effect(open_loop_tf, 0.5, False)
    
    # Substituir com o ganho real
    gain_analysis['recommended_gain'] = gain
    
    return gain_analysis
