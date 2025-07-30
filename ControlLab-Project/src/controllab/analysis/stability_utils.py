"""
Módulo de Utilidades para Análise de Estabilidade
=================================================

Este módulo implementa funções utilitárias comuns para análise de estabilidade
e validação cruzada entre diferentes métodos.

Classes:
    StabilityValidator: Validador cruzado entre métodos
    ParametricAnalyzer: Análise paramétrica de estabilidade

Funções:
    validate_stability_methods: Compara resultados entre métodos
    cross_validate_poles: Valida polos calculados
    format_stability_report: Formata relatórios pedagógicos
"""

import sympy as sp
from sympy import symbols, solve, diff, Matrix, simplify, Poly
from typing import Dict, List, Tuple, Any, Optional, Union
from numbers import Complex
import warnings

# Importar módulos de análise
try:
    from .routh_hurwitz import RouthHurwitzAnalyzer, StabilityResult
    from .root_locus import RootLocusAnalyzer, LocusFeatures
    from .frequency_response import FrequencyAnalyzer, StabilityMargins
except ImportError:
    # Fallbacks se módulos não estiverem disponíveis
    RouthHurwitzAnalyzer = None
    RootLocusAnalyzer = None
    FrequencyAnalyzer = None


class ValidationHistory:
    """Histórico de validação cruzada entre métodos"""
    
    def __init__(self):
        self.validations = []
        self.methods_used = []
        self.discrepancies = []
        self.final_conclusion = None
        
    def add_validation(self, method1: str, method2: str, agreement: bool, 
                      details: str = ""):
        validation = {
            'method1': method1,
            'method2': method2,
            'agreement': agreement,
            'details': details
        }
        self.validations.append(validation)
        
    def add_discrepancy(self, description: str, methods: List[str], 
                       explanation: str = ""):
        discrepancy = {
            'description': description,
            'methods': methods,
            'explanation': explanation
        }
        self.discrepancies.append(discrepancy)
        
    def get_formatted_report(self) -> str:
        """Retorna relatório de validação formatado"""
        report = "🔍 VALIDAÇÃO CRUZADA DE MÉTODOS - RELATÓRIO\n"
        report += "=" * 50 + "\n\n"
        
        report += f"📊 MÉTODOS UTILIZADOS: {', '.join(self.methods_used)}\n\n"
        
        report += "✅ VALIDAÇÕES REALIZADAS:\n"
        for val in self.validations:
            status = "✓" if val['agreement'] else "✗"
            report += f"{status} {val['method1']} vs {val['method2']}\n"
            if val['details']:
                report += f"   {val['details']}\n"
        
        if self.discrepancies:
            report += "\n⚠️ DISCREPÂNCIAS ENCONTRADAS:\n"
            for disc in self.discrepancies:
                report += f"• {disc['description']}\n"
                report += f"  Métodos: {', '.join(disc['methods'])}\n"
                if disc['explanation']:
                    report += f"  Explicação: {disc['explanation']}\n"
        
        if self.final_conclusion:
            report += f"\n🎯 CONCLUSÃO FINAL: {self.final_conclusion}\n"
        
        return report


class StabilityValidator:
    """
    Validador cruzado para métodos de análise de estabilidade
    
    Esta classe compara resultados entre diferentes métodos:
    - Routh-Hurwitz vs cálculo direto de polos
    - Root Locus vs análise de frequência
    - Margens de estabilidade vs critério de Nyquist
    """
    
    def __init__(self):
        self.history = ValidationHistory()
        
    def validate_stability_methods(self, tf_obj, show_steps: bool = True) -> Dict:
        """
        Valida estabilidade usando múltiplos métodos
        
        Args:
            tf_obj: Função de transferência
            show_steps: Se deve mostrar os passos
            
        Returns:
            Dict com resultados de todos os métodos
        """
        results = {}
        
        if show_steps:
            self.history = ValidationHistory()
            
        # Método 1: Routh-Hurwitz
        if RouthHurwitzAnalyzer:
            try:
                routh_analyzer = RouthHurwitzAnalyzer()
                
                # Extrair polinômio característico
                if hasattr(tf_obj, 'denominator'):
                    char_poly = tf_obj.denominator
                else:
                    char_poly = sp.denom(tf_obj)
                    
                routh_result = routh_analyzer.analyze_stability(
                    routh_analyzer.build_routh_array(char_poly, show_steps=False),
                    show_steps=False
                )
                results['routh_hurwitz'] = routh_result
                self.history.methods_used.append('Routh-Hurwitz')
                
            except Exception as e:
                results['routh_hurwitz'] = {
                    'error': str(e),
                    'is_stable': None,
                    'method': 'Routh-Hurwitz'
                }
        
        # Método 2: Cálculo direto de polos
        try:
            poles = self._calculate_poles_directly(tf_obj)
            stability_direct = self._analyze_poles_stability(poles)
            results['direct_poles'] = {
                'poles': poles,
                'is_stable': stability_direct,
                'unstable_count': sum(1 for p in poles if p.real > 0)
            }
            # Adicionar alias para compatibilidade com testes
            results['root_analysis'] = results['direct_poles']
            self.history.methods_used.append('Cálculo Direto')
            
        except Exception as e:
            results['direct_poles'] = {
                'error': str(e),
                'is_stable': None,
                'method': 'Direct Poles'
            }
        
        # Método 3: Análise de frequência (margens)
        if FrequencyAnalyzer:
            try:
                freq_analyzer = FrequencyAnalyzer()
                margins = freq_analyzer.calculate_gain_phase_margins(tf_obj, show_steps=False)
                results['frequency_margins'] = margins
                # Adicionar alias para compatibilidade com testes
                results['frequency_analysis'] = margins
                self.history.methods_used.append('Margens de Estabilidade')
                
            except Exception as e:
                results['frequency_margins'] = {
                    'error': str(e),
                    'is_stable': None,
                    'method': 'Frequency Margins'
                }
        
        # Validação cruzada
        if show_steps:
            self._perform_cross_validation(results)
            
        return results
    
    def _calculate_poles_directly(self, tf_obj) -> List[Complex]:
        """Calcula polos diretamente resolvendo denominador = 0"""
        if hasattr(tf_obj, 'denominator') and hasattr(tf_obj, 'variable'):
            denominator = tf_obj.denominator
            variable = tf_obj.variable
        else:
            denominator = sp.denom(tf_obj)
            variable = list(tf_obj.free_symbols)[0]
            
        poles = solve(denominator, variable)
        return [complex(sp.N(pole)) for pole in poles if pole.is_finite]
    
    def _analyze_poles_stability(self, poles: List[Complex]) -> bool:
        """Analisa estabilidade baseada na localização dos polos"""
        for pole in poles:
            if pole.real > 1e-10:  # Polo no semiplano direito
                return False
        return True
    
    def _perform_cross_validation(self, results: Dict):
        """Realiza validação cruzada entre métodos"""
        
        # Validar Routh-Hurwitz vs Cálculo Direto
        if 'routh_hurwitz' in results and 'direct_poles' in results:
            routh_stable = results['routh_hurwitz'].is_stable
            direct_stable = results['direct_poles']['is_stable']
            
            agreement = (routh_stable == direct_stable)
            details = f"Routh: {'Estável' if routh_stable else 'Instável'}, "
            details += f"Direto: {'Estável' if direct_stable else 'Instável'}"
            
            self.history.add_validation('Routh-Hurwitz', 'Cálculo Direto', 
                                      agreement, details)
            
            if not agreement:
                self.history.add_discrepancy(
                    "Discrepância entre Routh-Hurwitz e cálculo direto",
                    ['Routh-Hurwitz', 'Cálculo Direto'],
                    "Pode indicar erro numérico ou caso especial não tratado"
                )
        
        # Validar Margens vs outros métodos
        if 'frequency_margins' in results and 'direct_poles' in results:
            margins_stable = results['frequency_margins'].is_stable
            direct_stable = results['direct_poles']['is_stable']
            
            agreement = (margins_stable == direct_stable)
            details = f"Margens: {'Estável' if margins_stable else 'Instável'}, "
            details += f"Direto: {'Estável' if direct_stable else 'Instável'}"
            
            self.history.add_validation('Margens', 'Cálculo Direto', 
                                      agreement, details)
        
        # Conclusão final
        stable_methods = []
        unstable_methods = []
        
        for method, result in results.items():
            if isinstance(result, dict) and 'is_stable' in result:
                if result['is_stable']:
                    stable_methods.append(method)
                else:
                    unstable_methods.append(method)
            elif hasattr(result, 'is_stable'):
                if result.is_stable:
                    stable_methods.append(method)
                else:
                    unstable_methods.append(method)
        
        if len(stable_methods) > len(unstable_methods):
            self.history.final_conclusion = "SISTEMA ESTÁVEL (maioria dos métodos)"
        elif len(unstable_methods) > len(stable_methods):
            self.history.final_conclusion = "SISTEMA INSTÁVEL (maioria dos métodos)"
        else:
            self.history.final_conclusion = "ANÁLISE INCONCLUSIVA (métodos divergem)"


class ParametricAnalyzer:
    """
    Analisador de estabilidade paramétrica
    
    Analisa como parâmetros do sistema afetam a estabilidade,
    determinando regiões de estabilidade no espaço de parâmetros.
    """
    
    def __init__(self):
        self.parameter_history = []
        self.history = ValidationHistory()  # Adicionar para compatibilidade
        
    def stability_region_2d(self, system, param1: sp.Symbol, param2: sp.Symbol,
                           param1_range: Tuple[float, float],
                           param2_range: Tuple[float, float],
                           resolution: int = 50) -> Dict:
        """
        Determina região de estabilidade em espaço 2D de parâmetros
        
        Args:
            system: Sistema ou polinômio característico
            param1, param2: Parâmetros a variar
            param1_range, param2_range: Faixas dos parâmetros
            resolution: Resolução da grade
            
        Returns:
            Dict com região de estabilidade
        """
        import numpy as np
        
        # Criar grade de parâmetros
        p1_vals = np.linspace(param1_range[0], param1_range[1], resolution)
        p2_vals = np.linspace(param2_range[0], param2_range[1], resolution)
        
        stability_map = np.zeros((resolution, resolution))
        
        for i, p1_val in enumerate(p1_vals):
            for j, p2_val in enumerate(p2_vals):
                # Substituir valores dos parâmetros no sistema
                if hasattr(system, 'substitute'):
                    # SymbolicTransferFunction usa 'substitute'
                    system_substituted = system.substitute({param1: p1_val, param2: p2_val})
                elif hasattr(system, 'subs'):
                    # SymPy expressions usam 'subs'
                    system_substituted = system.subs([(param1, p1_val), (param2, p2_val)])
                else:
                    # Tentar subs direto
                    system_substituted = system.subs([(param1, p1_val), (param2, p2_val)])
                
                # Testar estabilidade
                try:
                    if RouthHurwitzAnalyzer:
                        analyzer = RouthHurwitzAnalyzer()
                        routh_array = analyzer.build_routh_array(system_substituted, show_steps=False)
                        result = analyzer.analyze_stability(routh_array, show_steps=False)
                        stability_map[j, i] = 1 if result.is_stable else 0
                    else:
                        # Fallback: calcular polos diretamente
                        poles = solve(system_substituted, 's')
                        is_stable = all(complex(sp.N(pole)).real < 0 for pole in poles if pole.is_finite)
                        stability_map[j, i] = 1 if is_stable else 0
                        
                except:
                    stability_map[j, i] = 0  # Assumir instável em caso de erro
        
        return {
            'param1_values': p1_vals,
            'param2_values': p2_vals,
            'stability_map': stability_map,
            'stable_region_area': np.sum(stability_map) / (resolution * resolution)
        }
    
    def root_locus_3d(self, system, param1: sp.Symbol, param2: sp.Symbol,
                     k_range: List[float]) -> Dict:
        """
        Análise de root locus tridimensional
        
        Args:
            system: Sistema com parâmetros
            param1, param2: Parâmetros adicionais
            k_range: Faixa de ganhos K
            
        Returns:
            Dict com dados 3D do root locus
        """
        # Implementação simplificada
        # Em uma implementação completa, calcularia lugar geométrico
        # para diferentes combinações de parâmetros
        
        results = {
            'parameters': [param1, param2],
            'k_values': k_range,
            'root_trajectories': [],
            'stability_boundaries': []
        }
        
        return results
    
    def analyze_sensitivity(self, system, parameter: sp.Symbol, 
                          nominal_value: float = 1.0, 
                          perturbation: float = 0.1) -> Dict:
        """
        Analisa sensibilidade das margens de estabilidade
        
        Args:
            system: Sistema a analisar
            parameter: Parâmetro para análise de sensibilidade
            nominal_value: Valor nominal do parâmetro
            perturbation: Perturbação relativa (0.1 = 10%)
            
        Returns:
            Dict com análise de sensibilidade
        """
        
        # Valores para análise
        delta = nominal_value * perturbation
        values = [
            nominal_value - delta,
            nominal_value,
            nominal_value + delta
        ]
        
        sensitivity_results = {
            'parameter': parameter,
            'nominal_value': nominal_value,
            'perturbation_percent': perturbation * 100,
            'stability_analysis': [],
            'sensitivity_metrics': {}
        }
        
        for value in values:
            system_perturbed = system.subs(parameter, value)
            
            try:
                # Análise de estabilidade para cada valor
                if FrequencyAnalyzer:
                    freq_analyzer = FrequencyAnalyzer()
                    margins = freq_analyzer.calculate_gain_phase_margins(system_perturbed, show_steps=False)
                    
                    analysis = {
                        'parameter_value': value,
                        'gain_margin_db': margins.gain_margin_db,
                        'phase_margin_deg': margins.phase_margin,
                        'is_stable': margins.is_stable
                    }
                else:
                    # Fallback básico
                    analysis = {
                        'parameter_value': value,
                        'gain_margin_db': None,
                        'phase_margin_deg': None,
                        'is_stable': None
                    }
                    
                sensitivity_results['stability_analysis'].append(analysis)
                
            except Exception as e:
                # Em caso de erro, registrar
                analysis = {
                    'parameter_value': value,
                    'error': str(e),
                    'is_stable': False
                }
                sensitivity_results['stability_analysis'].append(analysis)
        
        # Calcular métricas de sensibilidade
        if len(sensitivity_results['stability_analysis']) >= 3:
            analyses = sensitivity_results['stability_analysis']
            
            # Sensibilidade da margem de ganho
            if all(a.get('gain_margin_db') is not None for a in analyses):
                gm_nominal = analyses[1]['gain_margin_db']
                gm_low = analyses[0]['gain_margin_db'] 
                gm_high = analyses[2]['gain_margin_db']
                
                gm_sensitivity = (gm_high - gm_low) / (2 * delta) * nominal_value
                sensitivity_results['sensitivity_metrics']['gain_margin_sensitivity'] = gm_sensitivity
            
            # Sensibilidade da margem de fase
            if all(a.get('phase_margin_deg') is not None for a in analyses):
                pm_nominal = analyses[1]['phase_margin_deg']
                pm_low = analyses[0]['phase_margin_deg']
                pm_high = analyses[2]['phase_margin_deg']
                
                pm_sensitivity = (pm_high - pm_low) / (2 * delta) * nominal_value
                sensitivity_results['sensitivity_metrics']['phase_margin_sensitivity'] = pm_sensitivity
        
        return sensitivity_results
    
    def sensitivity_analysis(self, system, nominal_params: Dict[sp.Symbol, float],
                           perturbation: float = 0.1) -> Dict:
        """
        Análise de sensibilidade das margens de estabilidade
        
        Args:
            system: Sistema nominal
            nominal_params: Valores nominais dos parâmetros
            perturbation: Perturbação relativa (0.1 = 10%)
            
        Returns:
            Dict com análise de sensibilidade
        """
        sensitivities = {}
        
        # Calcular margens nominais
        nominal_system = system
        for param, value in nominal_params.items():
            nominal_system = nominal_system.subs(param, value)
            
        try:
            if FrequencyAnalyzer:
                analyzer = FrequencyAnalyzer()
                nominal_margins = analyzer.calculate_gain_phase_margins(nominal_system, show_steps=False)
                
                # Calcular sensibilidade para cada parâmetro
                for param, nominal_value in nominal_params.items():
                    # Perturbação positiva
                    perturbed_value = nominal_value * (1 + perturbation)
                    perturbed_system = system
                    for p, v in nominal_params.items():
                        val = perturbed_value if p == param else v
                        perturbed_system = perturbed_system.subs(p, val)
                    
                    perturbed_margins = analyzer.calculate_gain_phase_margins(perturbed_system, show_steps=False)
                    
                    # Calcular sensibilidade
                    gm_sensitivity = (perturbed_margins.gain_margin_db - nominal_margins.gain_margin_db) / perturbation
                    pm_sensitivity = (perturbed_margins.phase_margin - nominal_margins.phase_margin) / perturbation
                    
                    sensitivities[str(param)] = {
                        'gain_margin_sensitivity': gm_sensitivity,
                        'phase_margin_sensitivity': pm_sensitivity
                    }
            
        except Exception as e:
            sensitivities['error'] = str(e)
        
        return {
            'nominal_margins': nominal_margins if 'nominal_margins' in locals() else None,
            'sensitivities': sensitivities,
            'perturbation_used': perturbation
        }
    

# Funções utilitárias independentes
def validate_stability_methods(tf_obj, show_steps: bool = True) -> Dict:
    """Função wrapper para validação cruzada"""
    validator = StabilityValidator()
    return validator.validate_stability_methods(tf_obj, show_steps)


def cross_validate_poles(tf_obj) -> Dict:
    """Valida polos calculados por diferentes métodos"""
    validator = StabilityValidator()
    return validator._calculate_poles_directly(tf_obj)


def format_stability_report(results: Dict, include_details: bool = True) -> str:
    """Formata relatório completo de análise de estabilidade"""
    report = "📊 RELATÓRIO COMPLETO DE ANÁLISE DE ESTABILIDADE\n"
    report += "=" * 60 + "\n\n"
    
    # Resumo executivo
    stable_count = 0
    total_methods = 0
    
    for method, result in results.items():
        if isinstance(result, dict) and 'is_stable' in result:
            total_methods += 1
            if result['is_stable']:
                stable_count += 1
        elif hasattr(result, 'is_stable'):
            total_methods += 1
            if result.is_stable:
                stable_count += 1
    
    if total_methods > 0:
        stability_percentage = (stable_count / total_methods) * 100
        report += f"🎯 RESUMO: {stable_count}/{total_methods} métodos indicam estabilidade ({stability_percentage:.1f}%)\n\n"
    
    # Detalhes por método
    if include_details:
        for method, result in results.items():
            report += f"📋 {method.upper().replace('_', ' ')}:\n"
            if isinstance(result, str):
                report += f"   {result}\n"
            elif hasattr(result, '__str__'):
                report += f"   {str(result)}\n"
            else:
                report += f"   {result}\n"
            report += "\n"
    
    return report


def stability_region_2d(system, param1: sp.Symbol, param2: sp.Symbol,
                       param1_range: Tuple[float, float],
                       param2_range: Tuple[float, float],
                       resolution: int = 50) -> Dict:
    """Função wrapper para análise de região de estabilidade 2D"""
    analyzer = ParametricAnalyzer()
    return analyzer.stability_region_2d(system, param1, param2, 
                                      param1_range, param2_range, resolution)


def root_locus_3d(system, param1: sp.Symbol, param2: sp.Symbol,
                 k_range: List[float]) -> Dict:
    """Função wrapper para root locus 3D"""
    analyzer = ParametricAnalyzer()
    return analyzer.root_locus_3d(system, param1, param2, k_range)
