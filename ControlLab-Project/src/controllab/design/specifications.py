"""
ControlLab - Especificações de Desempenho
========================================

Este módulo implementa especificações de desempenho para projeto de controladores,
incluindo mapeamento entre especificações de tempo e frequência.

Características:
- Mapeamento especificações ↔ localizações de polos
- Validação automática de especificações
- Conversão entre domínios tempo/frequência
- Análise de trade-offs de projeto
"""

import sympy as sp
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass, field
from ..core.symbolic_tf import SymbolicTransferFunction
from ..core.symbolic_ss import SymbolicStateSpace
from .design_utils import ControllerResult, DesignSpecifications

@dataclass
class PerformanceSpec:
    """
    Classe avançada para especificações de desempenho
    
    Atributos:
        time_domain: Especificações no domínio do tempo
        frequency_domain: Especificações no domínio da frequência
        robustness: Especificações de robustez
        control_effort: Limitações de esforço de controle
    """
    # Domínio do tempo
    overshoot: Optional[float] = None  # %
    settling_time: Optional[float] = None  # s
    rise_time: Optional[float] = None  # s
    peak_time: Optional[float] = None  # s
    steady_state_error: Optional[float] = None  # %
    
    # Domínio da frequência
    phase_margin: Optional[float] = None  # graus
    gain_margin: Optional[float] = None  # dB
    bandwidth: Optional[float] = None  # rad/s
    gain_crossover_freq: Optional[float] = None  # rad/s
    phase_crossover_freq: Optional[float] = None  # rad/s
    
    # Parâmetros de segunda ordem
    damping_ratio: Optional[float] = None
    natural_frequency: Optional[float] = None
    
    # Robustez
    sensitivity_peak: Optional[float] = None
    complementary_sensitivity_peak: Optional[float] = None
    disturbance_rejection: Optional[float] = None
    
    # Esforço de controle
    max_control_effort: Optional[float] = None
    control_bandwidth: Optional[float] = None
    
    def to_second_order_params(self) -> Tuple[Optional[float], Optional[float]]:
        """
        Converte especificações para parâmetros de segunda ordem (ζ, ωn)
        
        Returns:
            Tuple[ζ, ωn]: Coeficiente de amortecimento e frequência natural
        """
        zeta, wn = None, None
        
        if self.overshoot is not None:
            # Mp = exp(-ζπ/√(1-ζ²)) → ζ = √(ln²(Mp)/(π² + ln²(Mp)))
            Mp = self.overshoot / 100.0
            if Mp > 0:
                ln_Mp = np.log(Mp)
                zeta = np.sqrt(ln_Mp**2 / (np.pi**2 + ln_Mp**2))
        
        if self.settling_time is not None and zeta is not None:
            # ts ≈ 4/(ζωn) para critério de 2%
            wn = 4 / (zeta * self.settling_time)
        
        if self.damping_ratio is not None:
            zeta = self.damping_ratio
            
        if self.natural_frequency is not None:
            wn = self.natural_frequency
            
        return zeta, wn
    
    def to_frequency_specs(self) -> Dict[str, float]:
        """
        Converte especificações de tempo para frequência
        
        Returns:
            Dict: Especificações equivalentes em frequência
        """
        freq_specs = {}
        
        zeta, wn = self.to_second_order_params()
        
        if zeta is not None and wn is not None:
            # Margem de fase: PM ≈ arctan(2ζ/√(√(1+4ζ⁴)-2ζ²))
            if zeta > 0:
                sqrt_term = np.sqrt(np.sqrt(1 + 4*zeta**4) - 2*zeta**2)
                freq_specs['phase_margin'] = np.degrees(np.arctan(2*zeta/sqrt_term))
            
            # Frequência de cruzamento de ganho
            freq_specs['gain_crossover_freq'] = wn * np.sqrt(np.sqrt(1 + 4*zeta**4) - 2*zeta**2)
            
            # Largura de banda
            freq_specs['bandwidth'] = wn * np.sqrt(1 - 2*zeta**2 + np.sqrt(2 - 4*zeta**2 + 4*zeta**4))
        
        return freq_specs
    
    def validate_consistency(self) -> Tuple[bool, List[str]]:
        """
        Valida consistência entre especificações
        
        Returns:
            Tuple[bool, List[str]]: (válido, lista_de_inconsistências)
        """
        errors = []
        
        # Validar limites físicos
        if self.overshoot is not None and (self.overshoot < 0 or self.overshoot > 100):
            errors.append("Sobressinal deve estar entre 0% e 100%")
        
        if self.damping_ratio is not None and self.damping_ratio < 0:
            errors.append("Coeficiente de amortecimento deve ser não-negativo")
        
        if self.phase_margin is not None and (self.phase_margin < 0 or self.phase_margin >= 180):
            errors.append("Margem de fase deve estar entre 0° e 180°")
        
        # Validar consistência entre especificações
        zeta, wn = self.to_second_order_params()
        freq_specs = self.to_frequency_specs()
        
        if self.phase_margin is not None and 'phase_margin' in freq_specs:
            diff = abs(self.phase_margin - freq_specs['phase_margin'])
            if diff > 5:  # Tolerância de 5 graus
                errors.append(f"Inconsistência: PM especificada={self.phase_margin}°, "
                            f"PM calculada={freq_specs['phase_margin']:.1f}°")
        
        return len(errors) == 0, errors

def design_by_specs(plant: SymbolicTransferFunction, 
                   performance_spec: PerformanceSpec,
                   method: str = 'auto') -> ControllerResult:
    """
    Projeta controlador baseado em especificações avançadas
    
    Args:
        plant: Planta do sistema
        performance_spec: Especificações de desempenho
        method: Método de projeto ('pid', 'lead', 'lag', 'auto')
    
    Returns:
        ControllerResult: Resultado do projeto
    """
    result = ControllerResult(controller=None)
    
    # Validar especificações
    is_valid, errors = performance_spec.validate_consistency()
    if not is_valid:
        for error in errors:
            result.add_step(f"❌ ERRO: {error}")
        raise ValueError(f"Especificações inconsistentes: {errors}")
    
    result.add_step("✅ Especificações validadas como consistentes")
    
    # Converter para parâmetros de segunda ordem
    zeta, wn = performance_spec.to_second_order_params()
    if zeta is not None and wn is not None:
        result.add_step(f"📊 Parâmetros equivalentes: ζ={zeta:.3f}, ωn={wn:.3f} rad/s")
    
    # Converter para especificações de frequência
    freq_specs = performance_spec.to_frequency_specs()
    if freq_specs:
        result.add_step("🔄 Especificações equivalentes em frequência:")
        for spec, value in freq_specs.items():
            result.add_step(f"   📐 {spec}: {value:.2f}")
    
    # Selecionar método automaticamente
    if method == 'auto':
        if performance_spec.overshoot is not None or performance_spec.settling_time is not None:
            method = 'lead'
        elif performance_spec.steady_state_error is not None:
            method = 'lag'
        else:
            method = 'pid'
        result.add_step(f"🔧 Método selecionado automaticamente: {method}")
    
    # Projeto do controlador (implementação simplificada)
    s = sp.Symbol('s')
    if method == 'pid':
        Kp, Ki, Kd = sp.symbols('Kp Ki Kd', real=True)
        controller = SymbolicTransferFunction(Kd*s**2 + Kp*s + Ki, s)
        result.add_step("🎯 Controlador PID projetado")
    elif method == 'lead':
        K, z, p = sp.symbols('K z p', real=True, positive=True)
        controller = SymbolicTransferFunction(K*(s + z), s + p)
        result.add_step("🎯 Compensador Lead projetado")
    else:  # lag
        K, z, p = sp.symbols('K z p', real=True, positive=True)
        controller = SymbolicTransferFunction(K*(s + z), s + p)
        result.add_step("🎯 Compensador Lag projetado")
    
    result.controller = controller
    
    # Adicionar notas educacionais
    result.add_educational_note("📚 Mapeamento especificações tempo ↔ frequência:")
    result.add_educational_note("   • Sobressinal ↔ Margem de fase")
    result.add_educational_note("   • Tempo de acomodação ↔ Largura de banda")
    result.add_educational_note("   • Erro regime ↔ Ganho em baixas frequências")
    
    return result

def verify_specifications(closed_loop: SymbolicTransferFunction, 
                         specs: PerformanceSpec) -> Dict[str, bool]:
    """
    Verifica se sistema em malha fechada atende especificações
    
    Args:
        closed_loop: Sistema em malha fechada
        specs: Especificações de desempenho
    
    Returns:
        Dict[str, bool]: Resultado da verificação para cada especificação
    """
    verification = {}
    
    try:
        print(f"[DEBUG] Tipo de closed_loop recebido em verify_specifications: {type(closed_loop)}")
        # Se não for SymbolicTransferFunction mas tem numerator/denominator, reconstrua
        from controllab.core.symbolic_tf import SymbolicTransferFunction
        if not isinstance(closed_loop, SymbolicTransferFunction):
            if hasattr(closed_loop, 'numerator') and hasattr(closed_loop, 'denominator'):
                closed_loop = SymbolicTransferFunction(closed_loop.numerator, closed_loop.denominator)
        # Obter polos do sistema de forma robusta
        if hasattr(closed_loop, 'get_poles'):
            poles = closed_loop.get_poles()
        elif hasattr(closed_loop, 'poles'):
            poles = closed_loop.poles()
        else:
            raise AttributeError("O objeto passado não possui métodos 'get_poles' nem 'poles'.")

        if poles:
            # Encontrar polos dominantes
            dominant_poles = []
            for pole in poles:
                # Tentar converter CRootOf para float se possível
                try:
                    if hasattr(pole, 'is_real') and pole.is_real:
                        val = float(pole)
                        if val < 0:
                            dominant_poles.append(complex(val, 0))
                    else:
                        # Tentar obter parte real/imag
                        real_part = float(sp.re(pole))
                        imag_part = float(sp.im(pole))
                        if real_part < 0:
                            dominant_poles.append(complex(real_part, imag_part))
                except Exception as e:
                    print(f"[DEBUG] Erro ao processar polo {pole}: {e}")
                    continue

            if dominant_poles:
                # Polo dominante (menor |parte_real|)
                dominant_pole = min(dominant_poles, key=lambda p: abs(p.real))

                # Calcular parâmetros
                if dominant_pole.imag != 0:  # Polos complexos
                    wn = abs(dominant_pole)
                    zeta = -dominant_pole.real / wn
                    # Verificar sobressinal
                    if specs.overshoot is not None:
                        calculated_overshoot = 100 * np.exp(-np.pi * zeta / np.sqrt(1 - zeta**2))
                        verification['overshoot'] = calculated_overshoot <= specs.overshoot
                    
                    # Verificar tempo de acomodação
                    if specs.settling_time is not None:
                        calculated_ts = 4 / abs(dominant_pole.real)
                        verification['settling_time'] = calculated_ts <= specs.settling_time
                
                else:  # Polo real
                    # Verificar tempo de acomodação para primeira ordem
                    if specs.settling_time is not None:
                        calculated_ts = 4 / abs(dominant_pole.real)
                        verification['settling_time'] = calculated_ts <= specs.settling_time
    
    except Exception as e:
        verification['error'] = f"Erro na verificação: {str(e)}"
    
    return verification

def pole_placement_from_specs(specs: PerformanceSpec) -> List[complex]:
    """
    Calcula localizações de polos desejadas baseado em especificações
    
    Args:
        specs: Especificações de desempenho
    
    Returns:
        List[complex]: Polos desejados
    """
    zeta, wn = specs.to_second_order_params()
    
    if zeta is not None and wn is not None:
        if zeta < 1:  # Sistema subamortecido
            wd = wn * np.sqrt(1 - zeta**2)
            pole1 = complex(-zeta * wn, wd)
            pole2 = complex(-zeta * wn, -wd)
            return [pole1, pole2]
        elif zeta == 1:  # Criticamente amortecido
            return [complex(-wn, 0), complex(-wn, 0)]
        else:  # Superamortecido
            r1 = -wn * (zeta + np.sqrt(zeta**2 - 1))
            r2 = -wn * (zeta - np.sqrt(zeta**2 - 1))
            return [complex(r1, 0), complex(r2, 0)]
    
    return []

def analyze_design_tradeoffs(specs: PerformanceSpec) -> Dict[str, Any]:
    """
    Analisa trade-offs entre diferentes especificações
    
    Args:
        specs: Especificações de desempenho
    
    Returns:
        Dict: Análise de trade-offs
    """
    analysis = {
        'conflicts': [],
        'recommendations': [],
        'feasibility': True
    }
    
    zeta, wn = specs.to_second_order_params()
    
    if zeta is not None:
        if zeta < 0.4:
            analysis['conflicts'].append("Baixo amortecimento pode causar overshoot excessivo")
            analysis['recommendations'].append("Considere aumentar coeficiente de amortecimento")
        
        if zeta > 0.8:
            analysis['conflicts'].append("Alto amortecimento pode tornar resposta muito lenta")
            analysis['recommendations'].append("Considere diminuir coeficiente de amortecimento")
    
    if specs.phase_margin is not None and specs.phase_margin < 30:
        analysis['conflicts'].append("Margem de fase baixa indica sistema próximo à instabilidade")
        analysis['recommendations'].append("Aumente margem de fase para pelo menos 45°")
        analysis['feasibility'] = False
    
    if specs.bandwidth is not None and specs.control_bandwidth is not None:
        if specs.bandwidth > specs.control_bandwidth:
            analysis['conflicts'].append("Largura de banda maior que capacidade do atuador")
            analysis['recommendations'].append("Reduza largura de banda ou melhore atuador")
    
    return analysis

class SpecificationMapper:
    """
    Classe para mapeamento entre especificações e parâmetros de projeto
    """
    
    @staticmethod
    def time_to_frequency(overshoot: float, settling_time: float) -> Tuple[float, float]:
        """
        Mapeia especificações de tempo para frequência
        
        Args:
            overshoot: Sobressinal em %
            settling_time: Tempo de acomodação em s
        
        Returns:
            Tuple[float, float]: (margem_fase, freq_cruzamento)
        """
        # Converter sobressinal para coeficiente de amortecimento
        Mp = overshoot / 100.0
        if Mp <= 0:
            zeta = 1.0
        else:
            ln_Mp = np.log(Mp)
            zeta = np.sqrt(ln_Mp**2 / (np.pi**2 + ln_Mp**2))
        
        # Calcular frequência natural
        wn = 4 / (zeta * settling_time)
        
        # Calcular margem de fase
        if zeta > 0:
            sqrt_term = np.sqrt(np.sqrt(1 + 4*zeta**4) - 2*zeta**2)
            phase_margin = np.degrees(np.arctan(2*zeta/sqrt_term))
        else:
            phase_margin = 0
        
        # Calcular frequência de cruzamento
        gain_crossover = wn * np.sqrt(np.sqrt(1 + 4*zeta**4) - 2*zeta**2)
        
        return phase_margin, gain_crossover
    
    @staticmethod
    def frequency_to_time(phase_margin: float, gain_crossover: float) -> Tuple[float, float]:
        """
        Mapeia especificações de frequência para tempo
        
        Args:
            phase_margin: Margem de fase em graus
            gain_crossover: Frequência de cruzamento em rad/s
        
        Returns:
            Tuple[float, float]: (sobressinal, tempo_acomodacao)
        """
        # Converter margem de fase para coeficiente de amortecimento (aproximação)
        pm_rad = np.radians(phase_margin)
        zeta = pm_rad / np.pi  # Aproximação simples
        
        # Estimar frequência natural
        wn = gain_crossover / np.sqrt(1 - 2*zeta**2)
        
        # Calcular sobressinal
        if zeta < 1:
            overshoot = 100 * np.exp(-np.pi * zeta / np.sqrt(1 - zeta**2))
        else:
            overshoot = 0
        
        # Calcular tempo de acomodação
        settling_time = 4 / (zeta * wn)
        
        return overshoot, settling_time
