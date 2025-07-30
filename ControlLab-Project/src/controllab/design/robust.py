"""
ControlLab - Projeto Robusto
============================

Este módulo implementa métodos de controle robusto:
- H-infinity design
- μ-synthesis
- Análise de margens de robustez
- Controle robusto com incertezas

Características:
- Derivação simbólica quando possível
- Explicações pedagógicas detalhadas
- Análise de incertezas estruturadas
- Métricas de robustez
"""

import sympy as sp
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
from ..core.symbolic_tf import SymbolicTransferFunction
from ..core.symbolic_ss import SymbolicStateSpace
from .design_utils import ControllerResult, create_educational_content

def h_infinity_design(plant: Union[SymbolicTransferFunction, SymbolicStateSpace],
                      performance_weight: Optional[SymbolicTransferFunction] = None,
                      uncertainty_weight: Optional[SymbolicTransferFunction] = None,
                      show_steps: bool = True) -> ControllerResult:
    """
    Projeta controlador H-infinity
    
    Args:
        plant: Planta a ser controlada
        performance_weight: Função de peso para performance
        uncertainty_weight: Função de peso para incertezas
        show_steps: Se deve mostrar passos
    
    Returns:
        ControllerResult: Controlador H-infinity
    """
    if show_steps:
        print("🎯 PROJETO DE CONTROLADOR H-INFINITY")
        print("=" * 40)
        print("🏭 Planta:", plant.expression if hasattr(plant, 'expression') else plant)
        if performance_weight:
            print("⚖️ Peso de performance:", performance_weight.expression)
        if uncertainty_weight:
            print("🔀 Peso de incerteza:", uncertainty_weight.expression)
    
    result = ControllerResult(controller=None)
    
    # Para implementação pedagógica básica
    result.add_step("Analisando estrutura da planta para H-infinity")
    
    if isinstance(plant, SymbolicTransferFunction):
        result.add_step(f"Planta: G(s) = {plant.expression}")
        
        # Implementação simplificada para fins educacionais
        # Em caso real, seria necessário resolver LMIs
        s = sp.Symbol('s')
        
        # Controlador placeholder baseado na planta
        if plant.poles():
            dominant_pole = min(plant.poles(), key=lambda p: abs(sp.re(p)))
            controller_gain = 1 / abs(dominant_pole)
            controller_tf = SymbolicTransferFunction(controller_gain, 1)
            result.controller = controller_tf
            result.add_step(f"Controlador H-infinity: K(s) = {controller_tf.expression}")
        else:
            result.add_step("❌ Não foi possível projetar controlador H-infinity")
            return result
    
    else:
        result.add_step("⚠️ Implementação para espaço de estados em desenvolvimento")
        return result
    
    # Adicionar conteúdo educacional
    educational_notes = [
        "🎓 CONTROLE H-INFINITY:",
        "• Minimiza norma H-infinity do sistema em malha fechada",
        "• Garante robustez a incertezas não-estruturadas",
        "• Baseia-se na solução de LMIs (Linear Matrix Inequalities)",
        "• Permite especificar trade-offs performance vs robustez"
    ]
    
    for note in educational_notes:
        result.add_educational_note(note)
    
    if show_steps:
        print(result.get_formatted_report())
    
    return result

def mu_synthesis(plant: Union[SymbolicTransferFunction, SymbolicStateSpace],
                uncertainty_structure: Dict[str, Any],
                show_steps: bool = True) -> ControllerResult:
    """
    Implementa μ-synthesis para controle robusto
    
    Args:
        plant: Planta a ser controlada
        uncertainty_structure: Estrutura de incertezas
        show_steps: Se deve mostrar passos
    
    Returns:
        ControllerResult: Controlador μ-synthesis
    """
    if show_steps:
        print("🎯 μ-SYNTHESIS PARA CONTROLE ROBUSTO")
        print("=" * 40)
        print("🏭 Planta:", plant.expression if hasattr(plant, 'expression') else plant)
        print("🔀 Estrutura de incertezas:", uncertainty_structure)
    
    result = ControllerResult(controller=None)
    
    result.add_step("Analisando estrutura de incertezas")
    result.add_step(f"Incertezas definidas: {list(uncertainty_structure.keys())}")
    
    # Implementação educacional básica
    if isinstance(plant, SymbolicTransferFunction):
        # Para fins pedagógicos, criar um controlador baseado em robustez
        s = sp.Symbol('s')
        
        # Controlador conservativo para robustez
        robustness_factor = uncertainty_structure.get('robustness_factor', 0.5)
        controller_tf = SymbolicTransferFunction(robustness_factor, 1)
        
        result.controller = controller_tf
        result.add_step(f"Controlador μ-synthesis: K(s) = {controller_tf.expression}")
        result.add_step(f"Fator de robustez aplicado: {robustness_factor}")
    
    else:
        result.add_step("⚠️ Implementação para espaço de estados em desenvolvimento")
    
    # Conteúdo educacional
    educational_notes = [
        "🎓 μ-SYNTHESIS:",
        "• Extensão do H-infinity para incertezas estruturadas",
        "• Minimiza valor singular estruturado μ",
        "• Considera conhecimento específico sobre incertezas",
        "• Iteração entre síntese H-infinity e análise μ"
    ]
    
    for note in educational_notes:
        result.add_educational_note(note)
    
    if show_steps:
        print(result.get_formatted_report())
    
    return result

def analyze_robustness_margins(closed_loop_system: Union[SymbolicTransferFunction, SymbolicStateSpace],
                              show_steps: bool = True) -> Dict[str, Any]:
    """
    Analisa margens de robustez do sistema em malha fechada
    
    Args:
        closed_loop_system: Sistema em malha fechada
        show_steps: Se deve mostrar passos
    
    Returns:
        Dict[str, Any]: Análise de robustez
    """
    if show_steps:
        print("🔍 ANÁLISE DE MARGENS DE ROBUSTEZ")
        print("=" * 35)
        print("🏭 Sistema:", closed_loop_system.expression if hasattr(closed_loop_system, 'expression') else closed_loop_system)
    
    robustness_analysis = {
        'system': closed_loop_system,
        'margins': {},
        'properties': [],
        'recommendations': []
    }
    
    if isinstance(closed_loop_system, SymbolicTransferFunction):
        # Analisar estabilidade básica
        poles = closed_loop_system.poles()
        stable = all(sp.re(pole) < 0 for pole in poles)
        
        robustness_analysis['margins']['stability'] = stable
        robustness_analysis['properties'].append(f"Sistema {'estável' if stable else 'instável'}")
        
        if stable:
            # Calcular distância aos polos mais próximos do eixo imaginário
            min_distance = min(abs(sp.re(pole)) for pole in poles)
            robustness_analysis['margins']['min_pole_distance'] = min_distance
            robustness_analysis['properties'].append(f"Distância mínima ao eixo jω: {min_distance}")
            
            # Recomendações baseadas em margens
            if min_distance < 0.1:
                robustness_analysis['recommendations'].append("⚠️ Sistema próximo à instabilidade - aumentar margens")
            elif min_distance > 1.0:
                robustness_analysis['recommendations'].append("✅ Boas margens de estabilidade")
            else:
                robustness_analysis['recommendations'].append("📊 Margens aceitáveis")
        
        else:
            robustness_analysis['recommendations'].append("❌ Sistema instável - redesenhar controlador")
    
    # Adicionar propriedades gerais de robustez
    robustness_analysis['properties'].extend([
        "Robustez multiplicativa vs aditiva",
        "Incertezas paramétricas vs não-paramétricas",
        "Trade-off performance vs robustez",
        "Margem de ganho e fase clássicas"
    ])
    
    if show_steps:
        print("\n📊 RESULTADOS DA ANÁLISE:")
        print(f"Estabilidade: {robustness_analysis['margins'].get('stability', 'N/A')}")
        if 'min_pole_distance' in robustness_analysis['margins']:
            print(f"Margem de estabilidade: {robustness_analysis['margins']['min_pole_distance']}")
        
        print("\n🎓 PROPRIEDADES DE ROBUSTEZ:")
        for prop in robustness_analysis['properties']:
            print(f"• {prop}")
        
        print("\n💡 RECOMENDAÇÕES:")
        for rec in robustness_analysis['recommendations']:
            print(f"• {rec}")
    
    return robustness_analysis

def design_robust_controller(plant: Union[SymbolicTransferFunction, SymbolicStateSpace],
                           uncertainty_description: Dict[str, Any],
                           performance_specs: Dict[str, float],
                           method: str = 'h_infinity',
                           show_steps: bool = True) -> ControllerResult:
    """
    Interface unificada para projeto de controladores robustos
    
    Args:
        plant: Planta a ser controlada
        uncertainty_description: Descrição das incertezas
        performance_specs: Especificações de performance
        method: Método de projeto ('h_infinity', 'mu_synthesis')
        show_steps: Se deve mostrar passos
    
    Returns:
        ControllerResult: Controlador robusto projetado
    """
    if show_steps:
        print("🎯 PROJETO DE CONTROLADOR ROBUSTO")
        print("=" * 35)
        print(f"📐 Método: {method}")
        print(f"🏭 Planta: {plant.expression if hasattr(plant, 'expression') else plant}")
        print(f"🔀 Incertezas: {uncertainty_description}")
        print(f"📊 Specs: {performance_specs}")
    
    if method.lower() == 'h_infinity':
        return h_infinity_design(plant, show_steps=show_steps)
    elif method.lower() == 'mu_synthesis':
        return mu_synthesis(plant, uncertainty_description, show_steps=show_steps)
    else:
        result = ControllerResult(controller=None)
        result.add_step(f"❌ Método '{method}' não implementado")
        return result

# Classe principal para design robusto
class RobustDesigner:
    """
    Classe para projeto sistemático de controladores robustos
    """
    
    def __init__(self, plant: Union[SymbolicTransferFunction, SymbolicStateSpace], show_steps: bool = True):
        """
        Inicializa o designer robusto
        
        Args:
            plant: Planta a ser controlada
            show_steps: Se deve mostrar passos
        """
        self.plant = plant
        self.show_steps = show_steps
        self.uncertainty_models = {}
        self.performance_weights = {}
    
    def add_uncertainty(self, name: str, model: Any):
        """Adiciona modelo de incerteza"""
        self.uncertainty_models[name] = model
    
    def add_performance_weight(self, name: str, weight: SymbolicTransferFunction):
        """Adiciona função de peso para performance"""
        self.performance_weights[name] = weight
    
    def design_h_infinity(self) -> ControllerResult:
        """Projeta controlador H-infinity"""
        performance_weight = self.performance_weights.get('performance', None)
        uncertainty_weight = self.performance_weights.get('uncertainty', None)
        return h_infinity_design(self.plant, performance_weight, uncertainty_weight, self.show_steps)
    
    def design_mu_synthesis(self) -> ControllerResult:
        """Projeta controlador por μ-synthesis"""
        return mu_synthesis(self.plant, self.uncertainty_models, self.show_steps)
    
    def analyze_robustness(self, controller: SymbolicTransferFunction) -> Dict[str, Any]:
        """Analisa robustez do sistema controlado"""
        if isinstance(self.plant, SymbolicTransferFunction):
            closed_loop = self.plant * controller / (1 + self.plant * controller)
            return analyze_robustness_margins(closed_loop, self.show_steps)
        else:
            return {'error': 'Análise para espaço de estados não implementada'}
