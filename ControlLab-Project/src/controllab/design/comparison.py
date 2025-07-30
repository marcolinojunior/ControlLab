"""
ControlLab - Comparação de Métodos de Projeto
============================================

Este módulo implementa comparação quantitativa entre diferentes métodos
de projeto de controladores, incluindo análise de Pareto e benchmarking.

Características:
- Comparação multi-critério de controladores
- Análise de Pareto para trade-offs
- Benchmarking de performance
- Ranqueamento automático de soluções
"""

import sympy as sp
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass, field
from ..core.symbolic_tf import SymbolicTransferFunction
from ..core.symbolic_ss import SymbolicStateSpace
from .design_utils import ControllerResult, DesignSpecifications
from .specifications import PerformanceSpec, verify_specifications

@dataclass
class ComparisonResult:
    """
    Resultado de comparação entre controladores
    
    Atributos:
        controllers: Lista de controladores comparados
        scores: Pontuações para cada critério
        rankings: Ranqueamento por critério
        pareto_optimal: Soluções Pareto-ótimas
        recommendations: Recomendações baseadas na análise
    """
    controllers: List[SymbolicTransferFunction] = field(default_factory=list)
    scores: Dict[str, List[float]] = field(default_factory=dict)
    rankings: Dict[str, List[int]] = field(default_factory=dict)
    pareto_optimal: List[int] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    overall_best: Optional[int] = None

def compare_controller_designs(plant: SymbolicTransferFunction,
                              controllers: List[SymbolicTransferFunction],
                              specifications: PerformanceSpec,
                              criteria: List[str] = None) -> ComparisonResult:
    """
    Compara diferentes projetos de controladores
    
    Args:
        plant: Planta do sistema
        controllers: Lista de controladores a comparar
        specifications: Especificações de desempenho
        criteria: Critérios de comparação
    
    Returns:
        ComparisonResult: Resultado da comparação
    """
    if criteria is None:
        criteria = ['stability', 'performance', 'robustness', 'complexity', 'implementation']
    
    print("⚖️ COMPARAÇÃO DE CONTROLADORES")
    print("=" * 50)
    print(f"🏭 Planta: G(s) = {plant}")
    print(f"📊 Controladores a comparar: {len(controllers)}")
    print(f"📋 Critérios: {criteria}")
    
    result = ComparisonResult()
    result.controllers = controllers
    
    # Inicializar scores
    for criterion in criteria:
        result.scores[criterion] = []
        result.rankings[criterion] = []
    
    # Analisar cada controlador
    closed_loop_systems = []
    for i, controller in enumerate(controllers):
        print(f"\n🔧 CONTROLADOR {i+1}: C{i+1}(s) = {controller}")
        
        try:
            # Sistema em malha fechada
            compensated = controller * plant
            closed_loop = compensated / (1 + compensated)
            closed_loop = closed_loop.simplify()
            closed_loop_systems.append(closed_loop)
            print(f"   🎛️ Malha fechada: T{i+1}(s) = {closed_loop}")

            # DEBUG: Mostrar polos do sistema em malha fechada
            try:
                if hasattr(closed_loop, 'get_poles'):
                    poles = closed_loop.get_poles()
                elif hasattr(closed_loop, 'poles'):
                    poles = closed_loop.poles()
                else:
                    poles = None
                print(f"   [DEBUG] Polos extraídos: {poles}")
            except Exception as e:
                print(f"   [DEBUG] Erro ao extrair polos: {e}")

            # Avaliar critérios
            scores_i = evaluate_controller_criteria(plant, controller, closed_loop, specifications, criteria)

            # DEBUG: Mostrar dicionário de verificação se performance
            if 'performance' in criteria:
                try:
                    from .specifications import verify_specifications
                    verification = verify_specifications(closed_loop, specifications)
                    print(f"   [DEBUG] Verificação de specs: {verification}")
                except Exception as e:
                    print(f"   [DEBUG] Erro na verificação de specs: {e}")

            for criterion in criteria:
                if criterion in scores_i:
                    result.scores[criterion].append(scores_i[criterion])
                    print(f"   📊 {criterion}: {scores_i[criterion]:.3f}")
                else:
                    result.scores[criterion].append(0.0)
                    print(f"   ⚠️ {criterion}: N/A")
        except Exception as e:
            print(f"   ❌ Erro na análise: {e}")
            for criterion in criteria:
                result.scores[criterion].append(0.0)
            closed_loop_systems.append(None)
    
    # Calcular rankings
    print(f"\n📈 RANKINGS POR CRITÉRIO:")
    for criterion in criteria:
        scores = result.scores[criterion]
        # Ranking: maior score = melhor posição (1º lugar)
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        ranking = [0] * len(scores)
        for rank, idx in enumerate(sorted_indices):
            ranking[idx] = rank + 1
        result.rankings[criterion] = ranking
        
        print(f"   🏆 {criterion}:")
        for i, (score, rank) in enumerate(zip(scores, ranking)):
            print(f"      C{i+1}: {score:.3f} (#{rank})")
    
    # Análise de Pareto
    result.pareto_optimal = find_pareto_optimal(result.scores, criteria)
    print(f"\n🎯 SOLUÇÕES PARETO-ÓTIMAS:")
    for idx in result.pareto_optimal:
        print(f"   🏆 Controlador C{idx+1}")
    
    # Ranqueamento geral
    overall_scores = calculate_overall_scores(result.scores, criteria)
    result.overall_best = overall_scores.index(max(overall_scores))
    
    print(f"\n🥇 CLASSIFICAÇÃO GERAL:")
    sorted_overall = sorted(enumerate(overall_scores), key=lambda x: x[1], reverse=True)
    for rank, (idx, score) in enumerate(sorted_overall):
        medal = "🥇" if rank == 0 else "🥈" if rank == 1 else "🥉" if rank == 2 else "🏅"
        print(f"   {medal} #{rank+1}: C{idx+1} - Score: {score:.3f}")
    
    # Gerar recomendações
    result.recommendations = generate_recommendations(result, controllers, criteria)
    
    print(f"\n💡 RECOMENDAÇÕES:")
    for rec in result.recommendations:
        print(f"   {rec}")
    
    return result

def evaluate_controller_criteria(plant: SymbolicTransferFunction,
                                controller: SymbolicTransferFunction,
                                closed_loop: SymbolicTransferFunction,
                                specifications: PerformanceSpec,
                                criteria: List[str]) -> Dict[str, float]:
    """
    Avalia controlador segundo critérios específicos
    
    Args:
        plant: Planta do sistema
        controller: Controlador
        closed_loop: Sistema em malha fechada
        specifications: Especificações
        criteria: Lista de critérios
    
    Returns:
        Dict[str, float]: Scores por critério (0-1, maior é melhor)
    """
    scores = {}
    
    for criterion in criteria:
        try:
            if criterion == 'stability':
                scores[criterion] = evaluate_stability_score(closed_loop)
            elif criterion == 'performance':
                scores[criterion] = evaluate_performance_score(closed_loop, specifications)
            elif criterion == 'robustness':
                scores[criterion] = evaluate_robustness_score(plant, controller)
            elif criterion == 'complexity':
                scores[criterion] = evaluate_complexity_score(controller)
            elif criterion == 'implementation':
                scores[criterion] = evaluate_implementation_score(controller)
            else:
                scores[criterion] = 0.5  # Score neutro para critérios desconhecidos
        
        except Exception as e:
            scores[criterion] = 0.0  # Score zero em caso de erro
    
    return scores

def evaluate_stability_score(closed_loop: SymbolicTransferFunction) -> float:
    """
    Avalia estabilidade do sistema (0-1, maior é melhor)
    
    Args:
        closed_loop: Sistema em malha fechada
    
    Returns:
        float: Score de estabilidade
    """
    try:
        poles = closed_loop.get_poles()
        
        if not poles:
            return 0.5  # Score neutro se não conseguir obter polos
        
        # Verificar se todos os polos são estáveis
        stable_poles = 0
        total_poles = len(poles)
        min_margin = float('inf')
        
        for pole in poles:
            real_part = float(sp.re(pole))
            if real_part < 0:
                stable_poles += 1
                min_margin = min(min_margin, -real_part)
            else:
                return 0.0  # Sistema instável
        
        if stable_poles == total_poles:
            # Sistema estável - score baseado na margem de estabilidade
            # Margem boa: > 1.0, Margem aceitável: 0.1-1.0, Margem crítica: < 0.1
            if min_margin > 1.0:
                return 1.0
            elif min_margin > 0.1:
                return 0.5 + 0.5 * (min_margin - 0.1) / 0.9
            else:
                return 0.1 + 0.4 * min_margin / 0.1
        else:
            return 0.0  # Sistema instável
    
    except:
        return 0.0

def evaluate_performance_score(closed_loop: SymbolicTransferFunction,
                              specifications: PerformanceSpec) -> float:
    """
    Avalia performance do sistema (0-1, maior é melhor)
    
    Args:
        closed_loop: Sistema em malha fechada
        specifications: Especificações desejadas
    
    Returns:
        float: Score de performance
    """
    try:
        # Verificar especificações
        verification = verify_specifications(closed_loop, specifications)
        
        if 'error' in verification:
            return 0.0
        
        # Calcular score baseado em quantas especificações foram atendidas
        total_specs = len(verification)
        met_specs = sum(1 for met in verification.values() if met)
        
        if total_specs == 0:
            return 0.5  # Score neutro se não há especificações
        
        base_score = met_specs / total_specs
        
        # Bônus por margem de folga nas especificações
        # (implementação simplificada)
        bonus = 0.0
        try:
            poles = closed_loop.get_poles()
            if poles:
                # Se sistema tem polos complexos, calcular amortecimento
                complex_poles = [p for p in poles if not sp.re(p).equals(p)]
                if complex_poles:
                    pole = complex_poles[0]
                    wn = abs(complex(float(sp.re(pole)), float(sp.im(pole))))
                    zeta = -float(sp.re(pole)) / wn
                    
                    # Bônus para amortecimento adequado (0.4 < ζ < 0.8)
                    if 0.4 <= zeta <= 0.8:
                        bonus = 0.1
        except:
            pass
        
        return min(1.0, base_score + bonus)
    
    except:
        return 0.0

def evaluate_robustness_score(plant: SymbolicTransferFunction,
                             controller: SymbolicTransferFunction) -> float:
    """
    Avalia robustez do controlador (0-1, maior é melhor)
    
    Args:
        plant: Planta do sistema
        controller: Controlador
    
    Returns:
        float: Score de robustez
    """
    try:
        # Análise baseada na estrutura do controlador
        score = 0.5  # Score base
        
        # Verificar se é integral (melhora robustez a distúrbios)
        s = sp.Symbol('s')
        if controller.expression.has(1/s):
            score += 0.2  # Bônus por ação integral
        
        # Verificar se é derivativo (pode reduzir robustez a ruído)
        if controller.expression.has(s**2) or controller.expression.has(s):
            score -= 0.1  # Penalidade por ação derivativa
        
        # Verificar ordem do controlador (menor ordem = mais robusto)
        try:
            controller_poles = controller.get_poles()
            controller_zeros = controller.get_zeros()
            
            total_order = len(controller_poles) + len(controller_zeros)
            
            if total_order <= 2:
                score += 0.2  # Bônus por baixa ordem
            elif total_order <= 4:
                score += 0.1
            else:
                score -= 0.1  # Penalidade por alta ordem
        except:
            pass
        
        # Verificar ganhos (ganhos muito altos reduzem robustez)
        try:
            dc_gain = abs(controller.evaluate_at(0))
            if dc_gain > 100:
                score -= 0.2  # Penalidade por ganho muito alto
            elif dc_gain > 10:
                score -= 0.1
        except:
            pass
        
        return max(0.0, min(1.0, score))
    
    except:
        return 0.0

def evaluate_complexity_score(controller: SymbolicTransferFunction) -> float:
    """
    Avalia simplicidade do controlador (0-1, maior é melhor/mais simples)
    
    Args:
        controller: Controlador
    
    Returns:
        float: Score de simplicidade
    """
    try:
        score = 1.0  # Score máximo para controlador simples
        
        # Penalizar por ordem
        try:
            poles = controller.get_poles()
            zeros = controller.get_zeros()
            
            total_order = len(poles) + len(zeros)
            
            # Penalidade crescente com a ordem
            if total_order > 5:
                score -= 0.5
            elif total_order > 3:
                score -= 0.3
            elif total_order > 1:
                score -= 0.1
        except:
            pass
        
        # Penalizar por complexidade da expressão
        expr_str = str(controller.expression)
        
        # Contar operações complexas
        complex_ops = expr_str.count('sqrt') + expr_str.count('exp') + expr_str.count('log')
        score -= 0.1 * complex_ops
        
        # Contar parâmetros
        from sympy import symbols
        free_symbols = len(controller.expression.free_symbols)
        if free_symbols > 5:
            score -= 0.2
        elif free_symbols > 3:
            score -= 0.1
        
        return max(0.0, min(1.0, score))
    
    except:
        return 0.5

def evaluate_implementation_score(controller: SymbolicTransferFunction) -> float:
    """
    Avalia facilidade de implementação (0-1, maior é melhor)
    
    Args:
        controller: Controlador
    
    Returns:
        float: Score de implementabilidade
    """
    try:
        score = 1.0  # Score máximo
        
        s = sp.Symbol('s')
        expr = controller.expression
        
        # Verificar se é próprio (grau denominador >= grau numerador)
        try:
            numer_degree = sp.degree(controller.numerator, s)
            denom_degree = sp.degree(controller.denominator, s)
            
            if numer_degree > denom_degree:
                score -= 0.3  # Penalidade por não ser próprio (difícil implementar)
        except:
            pass
        
        # Verificar se tem apenas polos/zeros reais (mais fácil implementar)
        try:
            poles = controller.get_poles()
            zeros = controller.get_zeros()
            
            complex_poles = sum(1 for p in poles if not sp.re(p).equals(p))
            complex_zeros = sum(1 for z in zeros if not sp.re(z).equals(z))
            
            if complex_poles == 0 and complex_zeros == 0:
                score += 0.1  # Bônus por ter apenas raízes reais
        except:
            pass
        
        # Verificar se tem termos derivativos de alta ordem
        if expr.has(s**3) or expr.has(s**4):
            score -= 0.2  # Penalidade por derivadas de alta ordem
        elif expr.has(s**2):
            score -= 0.1  # Penalidade menor por segunda derivada
        
        # Verificar se é digital-friendly (sem logaritmos, exponenciais, etc.)
        if expr.has(sp.log) or expr.has(sp.exp) or expr.has(sp.sqrt):
            score -= 0.2
        
        return max(0.0, min(1.0, score))
    
    except:
        return 0.5

def find_pareto_optimal(scores: Dict[str, List[float]], criteria: List[str]) -> List[int]:
    """
    Encontra soluções Pareto-ótimas
    
    Args:
        scores: Scores por critério
        criteria: Lista de critérios
    
    Returns:
        List[int]: Índices das soluções Pareto-ótimas
    """
    n_solutions = len(scores[criteria[0]])
    pareto_optimal = []
    
    for i in range(n_solutions):
        is_pareto = True
        
        for j in range(n_solutions):
            if i == j:
                continue
            
            # Verificar se solução j domina solução i
            dominates = True
            better_in_at_least_one = False
            
            for criterion in criteria:
                score_i = scores[criterion][i]
                score_j = scores[criterion][j]
                
                if score_j < score_i:
                    dominates = False
                    break
                elif score_j > score_i:
                    better_in_at_least_one = True
            
            if dominates and better_in_at_least_one:
                is_pareto = False
                break
        
        if is_pareto:
            pareto_optimal.append(i)
    
    return pareto_optimal

def calculate_overall_scores(scores: Dict[str, List[float]], 
                           criteria: List[str],
                           weights: Dict[str, float] = None) -> List[float]:
    """
    Calcula scores gerais ponderados
    
    Args:
        scores: Scores por critério
        criteria: Lista de critérios
        weights: Pesos para cada critério
    
    Returns:
        List[float]: Scores gerais
    """
    if weights is None:
        # Pesos padrão uniformes
        weights = {criterion: 1.0 for criterion in criteria}
    
    n_solutions = len(scores[criteria[0]])
    overall_scores = []
    
    for i in range(n_solutions):
        weighted_sum = 0.0
        total_weight = 0.0
        
        for criterion in criteria:
            weight = weights.get(criterion, 1.0)
            weighted_sum += weight * scores[criterion][i]
            total_weight += weight
        
        overall_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        overall_scores.append(overall_score)
    
    return overall_scores

def generate_recommendations(result: ComparisonResult,
                           controllers: List[SymbolicTransferFunction],
                           criteria: List[str]) -> List[str]:
    """
    Gera recomendações baseadas na análise
    
    Args:
        result: Resultado da comparação
        controllers: Lista de controladores
        criteria: Critérios analisados
    
    Returns:
        List[str]: Lista de recomendações
    """
    recommendations = []
    
    # Recomendação geral
    if result.overall_best is not None:
        recommendations.append(f"🥇 Melhor escolha geral: Controlador C{result.overall_best + 1}")
    
    # Recomendações por critério
    for criterion in criteria:
        if criterion in result.rankings:
            best_idx = result.rankings[criterion].index(1)  # Posição 1 = melhor
            recommendations.append(f"🏆 Melhor em {criterion}: Controlador C{best_idx + 1}")
    
    # Recomendações sobre Pareto
    if len(result.pareto_optimal) > 1:
        pareto_names = [f"C{i+1}" for i in result.pareto_optimal]
        recommendations.append(f"⚖️ Soluções equilibradas (Pareto): {', '.join(pareto_names)}")
    elif len(result.pareto_optimal) == 1:
        idx = result.pareto_optimal[0]
        recommendations.append(f"👑 Solução dominante: Controlador C{idx + 1}")
    
    # Análise de trade-offs
    recommendations.append("📊 Trade-offs identificados:")
    
    # Verificar correlações entre critérios
    if 'complexity' in criteria and 'performance' in criteria:
        complexity_scores = result.scores['complexity']
        performance_scores = result.scores['performance']
        
        # Encontrar soluções com melhor trade-off complexidade vs performance
        trade_off_scores = [(c + p) / 2 for c, p in zip(complexity_scores, performance_scores)]
        best_trade_off = trade_off_scores.index(max(trade_off_scores))
        recommendations.append(f"   🔄 Melhor trade-off simplicidade/performance: C{best_trade_off + 1}")
    
    if 'robustness' in criteria and 'performance' in criteria:
        robustness_scores = result.scores['robustness']
        performance_scores = result.scores['performance']
        
        trade_off_scores = [(r + p) / 2 for r, p in zip(robustness_scores, performance_scores)]
        best_trade_off = trade_off_scores.index(max(trade_off_scores))
        recommendations.append(f"   🛡️ Melhor trade-off robustez/performance: C{best_trade_off + 1}")
    
    return recommendations

def pareto_analysis(designs: List[ControllerResult], 
                   criteria: List[str]) -> Dict[str, Any]:
    """
    Análise de Pareto para múltiplos projetos
    
    Args:
        designs: Lista de resultados de projeto
        criteria: Critérios para análise
    
    Returns:
        Dict: Resultado da análise de Pareto
    """
    print("📊 ANÁLISE DE PARETO")
    print("=" * 30)
    
    # Extrair scores dos designs
    scores = {criterion: [] for criterion in criteria}
    
    for i, design in enumerate(designs):
        print(f"\n🔧 Design {i+1}: {design.controller}")
        
        # Calcular scores (implementação simplificada)
        for criterion in criteria:
            if criterion in design.performance_metrics:
                # Normalizar métricas para score 0-1
                score = normalize_metric_to_score(design.performance_metrics[criterion], criterion)
            else:
                score = 0.5  # Score neutro
            
            scores[criterion].append(score)
            print(f"   📊 {criterion}: {score:.3f}")
    
    # Encontrar soluções Pareto-ótimas
    pareto_optimal = find_pareto_optimal(scores, criteria)
    
    result = {
        'scores': scores,
        'pareto_optimal': pareto_optimal,
        'dominated_solutions': [i for i in range(len(designs)) if i not in pareto_optimal],
        'analysis': []
    }
    
    print(f"\n🎯 FRONTEIRA DE PARETO:")
    for idx in pareto_optimal:
        print(f"   🏆 Design {idx+1} - Pareto-ótimo")
        result['analysis'].append(f"Design {idx+1} é Pareto-ótimo")
    
    print(f"\n📉 SOLUÇÕES DOMINADAS:")
    for idx in result['dominated_solutions']:
        print(f"   📊 Design {idx+1} - Dominado")
        result['analysis'].append(f"Design {idx+1} é dominado por outras soluções")
    
    return result

def normalize_metric_to_score(metric_value: float, metric_type: str) -> float:
    """
    Normaliza métrica para score 0-1
    
    Args:
        metric_value: Valor da métrica
        metric_type: Tipo da métrica
    
    Returns:
        float: Score normalizado (0-1, maior é melhor)
    """
    # Implementação simplificada - na prática, usar limites conhecidos
    if metric_type in ['overshoot', 'settling_time', 'steady_state_error']:
        # Para estes, menor é melhor
        return max(0.0, min(1.0, 1.0 / (1.0 + abs(metric_value))))
    else:
        # Para outros, assumir que maior é melhor
        return max(0.0, min(1.0, metric_value))

def sensitivity_comparison(designs: List[ControllerResult],
                          uncertainties: Dict[str, float]) -> Dict[str, Any]:
    """
    Compara sensibilidade de diferentes projetos a incertezas
    
    Args:
        designs: Lista de projetos
        uncertainties: Incertezas paramétricas (parâmetro: variação_%)
    
    Returns:
        Dict: Análise de sensibilidade comparativa
    """
    print("🔍 ANÁLISE DE SENSIBILIDADE COMPARATIVA")
    print("=" * 50)
    print(f"📊 Designs: {len(designs)}")
    print(f"⚠️ Incertezas: {uncertainties}")
    
    sensitivity_results = {
        'designs': [],
        'sensitivities': {},
        'most_robust': None,
        'most_sensitive': None
    }
    
    for i, design in enumerate(designs):
        print(f"\n🔧 Design {i+1}: {design.controller}")
        
        design_sensitivity = {}
        total_sensitivity = 0.0
        
        for param, variation in uncertainties.items():
            # Calcular sensibilidade (implementação simplificada)
            # Em implementação real, seria uma análise paramétrica completa
            
            # Estimativa baseada na estrutura do controlador
            param_sensitivity = estimate_parameter_sensitivity(design.controller, param, variation)
            design_sensitivity[param] = param_sensitivity
            total_sensitivity += param_sensitivity
            
            print(f"   📊 Sensibilidade a {param} (±{variation}%): {param_sensitivity:.3f}")
        
        avg_sensitivity = total_sensitivity / len(uncertainties)
        design_sensitivity['average'] = avg_sensitivity
        
        print(f"   📈 Sensibilidade média: {avg_sensitivity:.3f}")
        
        sensitivity_results['designs'].append(f"Design {i+1}")
        sensitivity_results['sensitivities'][f"Design_{i+1}"] = design_sensitivity
    
    # Encontrar mais robusto e mais sensível
    avg_sensitivities = [sensitivity_results['sensitivities'][f"Design_{i+1}"]['average'] 
                        for i in range(len(designs))]
    
    most_robust_idx = avg_sensitivities.index(min(avg_sensitivities))
    most_sensitive_idx = avg_sensitivities.index(max(avg_sensitivities))
    
    sensitivity_results['most_robust'] = most_robust_idx
    sensitivity_results['most_sensitive'] = most_sensitive_idx
    
    print(f"\n🛡️ MAIS ROBUSTO: Design {most_robust_idx + 1} (sensibilidade: {avg_sensitivities[most_robust_idx]:.3f})")
    print(f"⚠️ MAIS SENSÍVEL: Design {most_sensitive_idx + 1} (sensibilidade: {avg_sensitivities[most_sensitive_idx]:.3f})")
    
    return sensitivity_results

def estimate_parameter_sensitivity(controller: SymbolicTransferFunction,
                                 parameter: str,
                                 variation_percent: float) -> float:
    """
    Estima sensibilidade do controlador a variação paramétrica
    
    Args:
        controller: Controlador
        parameter: Nome do parâmetro
        variation_percent: Variação percentual
    
    Returns:
        float: Índice de sensibilidade (0-1, menor é mais robusto)
    """
    try:
        # Implementação simplificada baseada na estrutura
        expr = controller.expression
        
        # Verificar se o parâmetro aparece na expressão
        free_symbols = [str(sym) for sym in expr.free_symbols]
        
        if parameter not in free_symbols:
            return 0.0  # Não sensível a parâmetro que não aparece
        
        # Estimar sensibilidade baseada na ordem e estrutura
        param_symbol = sp.Symbol(parameter)
        
        # Calcular a derivada simbólica como indicador de sensibilidade
        try:
            derivative = sp.diff(expr, param_symbol)
            derivative_complexity = len(str(derivative))
            
            # Normalizar baseado na complexidade da derivada
            base_complexity = len(str(expr))
            if base_complexity > 0:
                sensitivity_factor = derivative_complexity / base_complexity
            else:
                sensitivity_factor = 1.0
            
            # Ajustar pela magnitude da variação
            scaled_sensitivity = sensitivity_factor * (variation_percent / 100.0)
            
            return min(1.0, scaled_sensitivity)
        
        except:
            # Se não conseguir calcular derivada, usar heurística
            expr_str = str(expr)
            param_count = expr_str.count(parameter)
            
            # Mais ocorrências = maior sensibilidade
            sensitivity = min(1.0, param_count * 0.1 * (variation_percent / 100.0))
            return sensitivity
    
    except:
        return 0.5  # Sensibilidade média em caso de erro
