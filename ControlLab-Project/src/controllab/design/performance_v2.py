"""
ControlLab - Módulo 6: Análise de Desempenho de Controladores
============================================================

Este módulo implementa análise de desempenho seguindo a filosofia educacional:
- Análise passo-a-passo como o Symbolab
- Outputs estruturados com múltiplas métricas
- Explicação do significado físico de cada métrica
- Derivação de fórmulas analíticas

Baseado na Seção 4 do oQUEfazer.md:
"permitir que os usuários derivem fórmulas gerais"
"""

import sympy as sp
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass
from ..core.symbolic_tf import SymbolicTransferFunction


@dataclass
class PerformanceAnalysis:
    """
    Estrutura para análise completa de desempenho
    Seguindo padrão educacional: múltiplas métricas e explicações
    """
    metrics: Dict[str, Any]
    symbolic_derivations: List[Dict[str, Any]]
    physical_interpretations: Dict[str, str]
    design_recommendations: List[str]
    comparative_analysis: Optional[Dict[str, Any]] = None


class EducationalPerformanceAnalyzer:
    """
    Analisador de desempenho com foco educacional total
    
    Implementa análise como o Symbolab:
    1. Derivação de cada métrica
    2. Explicação física
    3. Outputs estruturados
    4. Comparações
    """
    
    def __init__(self):
        self.s = sp.Symbol('s')
        
    def analyze_transient_response_complete(self,
                                          closed_loop_tf: SymbolicTransferFunction,
                                          show_derivations: bool = True) -> PerformanceAnalysis:
        """
        Análise COMPLETA de resposta transitória com todas as derivações
        
        Args:
            closed_loop_tf: Função de transferência de malha fechada
            show_derivations: Mostra todas as derivações simbólicas
            
        Returns:
            PerformanceAnalysis: Análise estruturada completa
        """
        print("🎯 ANÁLISE DE RESPOSTA TRANSITÓRIA - COMPLETA")
        print("=" * 60)
        
        # PASSO 1: Identificação da estrutura
        print("\n📚 PASSO 1: Análise da Estrutura do Sistema")
        print("-" * 40)
        print(f"🔹 Sistema em malha fechada: T(s) = {closed_loop_tf}")
        
        # Obter polos e zeros
        poles = closed_loop_tf.poles()
        zeros = closed_loop_tf.zeros()
        
        print(f"🔹 Polos: {poles}")
        print(f"🔹 Zeros: {zeros}")
        
        derivations = []
        metrics = {}
        
        # PASSO 2: Classificação do sistema
        system_type = self._classify_system_type(poles)
        print(f"\n📊 PASSO 2: Classificação do Sistema")
        print("-" * 40)
        print(f"🏷️ Tipo de sistema: {system_type['type']}")
        print(f"🔹 Característica: {system_type['description']}")
        
        derivations.append({
            "step": 1,
            "title": "Classificação do Sistema",
            "analysis": system_type,
            "poles": poles,
            "classification_criteria": "Baseado na natureza dos polos dominantes"
        })
        
        # PASSO 3: Derivação das métricas (depende do tipo)
        if system_type['type'] == 'segunda_ordem':
            metrics.update(self._analyze_second_order_complete(closed_loop_tf, poles, show_derivations))
        elif system_type['type'] == 'primeira_ordem':
            metrics.update(self._analyze_first_order_complete(closed_loop_tf, poles, show_derivations))
        else:
            metrics.update(self._analyze_higher_order_complete(closed_loop_tf, poles, show_derivations))
        
        # PASSO 4: Interpretações físicas
        interpretations = self._generate_physical_interpretations(metrics, system_type)
        
        # PASSO 5: Recomendações de design
        recommendations = self._generate_design_recommendations(metrics, system_type)
        
        print(f"\n✅ ANÁLISE COMPLETA:")
        print(f"   📊 Métricas calculadas: {len(metrics)}")
        print(f"   📚 Derivações: {len(derivations)}")
        print(f"   🎯 Recomendações: {len(recommendations)}")
        
        return PerformanceAnalysis(
            metrics=metrics,
            symbolic_derivations=derivations,
            physical_interpretations=interpretations,
            design_recommendations=recommendations
        )
    
    def _classify_system_type(self, poles) -> Dict[str, Any]:
        """Classifica tipo de sistema baseado nos polos"""
        
        if len(poles) == 1:
            return {
                "type": "primeira_ordem",
                "description": "Sistema com um polo real dominante",
                "characteristics": ["Sem overshoot", "Resposta exponencial"]
            }
        elif len(poles) == 2:
            # Verificar se são complexos conjugados
            if any(pole.is_real == False for pole in poles):
                return {
                    "type": "segunda_ordem",
                    "description": "Sistema com polos complexos conjugados",
                    "characteristics": ["Pode ter overshoot", "Resposta oscilatória"]
                }
            else:
                return {
                    "type": "segunda_ordem_real",
                    "description": "Sistema com dois polos reais",
                    "characteristics": ["Sem overshoot", "Dois modos exponenciais"]
                }
        else:
            return {
                "type": "ordem_superior",
                "description": "Sistema de ordem superior",
                "characteristics": ["Comportamento complexo", "Múltiplos modos"]
            }
    
    def _analyze_second_order_complete(self, tf, poles, show_derivations) -> Dict[str, Any]:
        """Análise completa de sistema de segunda ordem"""
        
        print("\n🎯 ANÁLISE DE SEGUNDA ORDEM - DERIVAÇÃO COMPLETA")
        print("-" * 50)
        
        metrics = {}
        
        # Extrair parâmetros da forma padrão
        # T(s) = ωn²/(s² + 2ζωn*s + ωn²)
        
        # Obter denominador
        denom = tf.denominator
        print(f"🔹 Denominador: {denom}")
        
        # Extrair coeficientes
        coeffs = sp.Poly(denom, self.s).all_coeffs()
        
        if len(coeffs) == 3:  # s² + as + b
            a2, a1, a0 = coeffs
            
            # Derivar ωn e ζ
            print("\n📐 DERIVAÇÃO DOS PARÂMETROS:")
            print("🔹 Forma padrão: s² + 2ζωn*s + ωn²")
            print(f"🔹 Seu sistema: {a2}s² + {a1}s + {a0}")
            
            # ωn = √(a0/a2)
            wn = sp.sqrt(a0/a2)
            print(f"🔹 ωn = √({a0}/{a2}) = {wn}")
            
            # ζ = a1/(2*√(a0*a2))
            zeta = a1/(2*sp.sqrt(a0*a2))
            print(f"🔹 ζ = {a1}/(2√({a0}·{a2})) = {zeta}")
            
            metrics['wn'] = wn
            metrics['zeta'] = zeta
            
            # Derivar métricas de desempenho
            print("\n📊 MÉTRICAS DE DESEMPENHO:")
            
            # Tempo de subida (0% a 100%)
            if show_derivations:
                print("🔹 Tempo de subida tr (0% a 100%):")
                print("   Para sistema subamortecido (ζ < 1):")
                print("   tr = (π - β)/ωd, onde β = arccos(ζ)")
                print("   ωd = ωn√(1 - ζ²) (freq. natural amortecida)")
            
            wd = wn * sp.sqrt(1 - zeta**2)
            beta = sp.acos(zeta)
            tr = (sp.pi - beta)/wd
            
            metrics['wd'] = wd
            metrics['tr'] = tr
            
            print(f"   ωd = {wn}√(1 - {zeta}²) = {wd}")
            print(f"   tr = (π - arccos({zeta}))/{wd} = {tr}")
            
            # Tempo de pico
            tp = sp.pi/wd
            metrics['tp'] = tp
            print(f"🔹 Tempo de pico: tp = π/ωd = {tp}")
            
            # Overshoot percentual
            if show_derivations:
                print("🔹 Overshoot percentual:")
                print("   Mp = exp(-ζπ/√(1-ζ²)) × 100%")
            
            Mp = sp.exp(-zeta*sp.pi/sp.sqrt(1-zeta**2)) * 100
            metrics['Mp'] = Mp
            print(f"   Mp = exp(-{zeta}π/√(1-{zeta}²)) × 100% = {Mp}%")
            
            # Tempo de estabelecimento (critério 2%)
            ts_2 = 4/(zeta*wn)  # Critério 2%
            ts_5 = 3/(zeta*wn)  # Critério 5%
            
            metrics['ts_2'] = ts_2
            metrics['ts_5'] = ts_5
            
            if show_derivations:
                print("🔹 Tempo de estabelecimento:")
                print("   Critério 2%: ts = 4/(ζωn)")
                print("   Critério 5%: ts = 3/(ζωn)")
            
            print(f"   ts(2%) = 4/({zeta}·{wn}) = {ts_2}")
            print(f"   ts(5%) = 3/({zeta}·{wn}) = {ts_5}")
        
        return metrics
    
    def _analyze_first_order_complete(self, tf, poles, show_derivations) -> Dict[str, Any]:
        """Análise completa de sistema de primeira ordem"""
        
        print("\n🎯 ANÁLISE DE PRIMEIRA ORDEM")
        print("-" * 30)
        
        pole = poles[0]
        tau = -1/pole  # Constante de tempo
        
        metrics = {
            'pole': pole,
            'time_constant': tau,
            'tr_10_90': 2.2 * tau,  # Tempo de subida 10% a 90%
            'ts_2': 4 * tau,        # Tempo estabelecimento 2%
            'ts_5': 3 * tau         # Tempo estabelecimento 5%
        }
        
        if show_derivations:
            print(f"🔹 Polo: p = {pole}")
            print(f"🔹 Constante de tempo: τ = -1/p = {tau}")
            print("🔹 Resposta ao degrau: y(t) = 1 - e^(-t/τ)")
            print(f"🔹 Tempo de subida (10%-90%): tr = 2.2τ = {metrics['tr_10_90']}")
            print(f"🔹 Tempo de estabelecimento (2%): ts = 4τ = {metrics['ts_2']}")
        
        return metrics
    
    def _analyze_higher_order_complete(self, tf, poles, show_derivations) -> Dict[str, Any]:
        """Análise de sistemas de ordem superior"""
        
        print("\n🎯 ANÁLISE DE ORDEM SUPERIOR")
        print("-" * 30)
        
        # Encontrar polos dominantes
        dominant_poles = self._find_dominant_poles(poles)
        
        metrics = {
            'all_poles': poles,
            'dominant_poles': dominant_poles,
            'order': len(poles)
        }
        
        print(f"🔹 Ordem do sistema: {len(poles)}")
        print(f"🔹 Polos dominantes: {dominant_poles}")
        print("🔹 Análise baseada nos polos dominantes")
        
        return metrics
    
    def _find_dominant_poles(self, poles):
        """Encontra polos dominantes (mais próximos do eixo imaginário)"""
        
        real_parts = []
        for pole in poles:
            if pole.is_real:
                real_parts.append(abs(pole))
            else:
                real_parts.append(abs(sp.re(pole)))
        
        # Polos com menor parte real (em módulo) são dominantes
        min_real = min(real_parts)
        dominant = [pole for pole, real_part in zip(poles, real_parts) 
                   if abs(real_part - min_real) < 1e-10]
        
        return dominant
    
    def _generate_physical_interpretations(self, metrics, system_type) -> Dict[str, str]:
        """Gera interpretações físicas das métricas"""
        
        interpretations = {}
        
        if 'wn' in metrics:
            interpretations['wn'] = "Frequência natural: velocidade de oscilação livre do sistema"
        
        if 'zeta' in metrics:
            interpretations['zeta'] = "Fator de amortecimento: controla overshoot e oscilações"
        
        if 'Mp' in metrics:
            interpretations['Mp'] = "Overshoot: máximo ultrapassagem do valor final"
        
        if 'tr' in metrics:
            interpretations['tr'] = "Tempo de subida: rapidez da resposta inicial"
        
        if 'tp' in metrics:
            interpretations['tp'] = "Tempo de pico: quando ocorre máximo overshoot"
        
        if 'ts_2' in metrics:
            interpretations['ts_2'] = "Tempo de estabelecimento: quando o sistema se estabiliza"
        
        return interpretations
    
    def _generate_design_recommendations(self, metrics, system_type) -> List[str]:
        """Gera recomendações de design baseadas nas métricas"""
        
        recommendations = []
        
        if system_type['type'] == 'segunda_ordem':
            if 'zeta' in metrics:
                recommendations.append("Para reduzir overshoot: aumentar ζ (mais amortecimento)")
                recommendations.append("Para resposta mais rápida: aumentar ωn")
                recommendations.append("ζ = 0.707 fornece resposta otimizada (sem overshoot excessivo)")
        
        elif system_type['type'] == 'primeira_ordem':
            recommendations.append("Para resposta mais rápida: mover polo para esquerda")
            recommendations.append("Sistema de primeira ordem nunca tem overshoot")
        
        recommendations.append("Verificar estabilidade: todos os polos no semiplano esquerdo")
        recommendations.append("Considerar efeito de zeros na resposta")
        
        return recommendations

    def compare_controllers_complete(self,
                                   controllers: Dict[str, SymbolicTransferFunction],
                                   plant: SymbolicTransferFunction) -> Dict[str, PerformanceAnalysis]:
        """
        Comparação completa entre diferentes controladores
        
        Args:
            controllers: Dicionário {nome: controlador}
            plant: Planta do sistema
            
        Returns:
            Dict[str, PerformanceAnalysis]: Análises comparativas
        """
        print("🎯 COMPARAÇÃO DE CONTROLADORES - ANÁLISE COMPLETA")
        print("=" * 60)
        
        analyses = {}
        
        for name, controller in controllers.items():
            print(f"\n🔍 ANALISANDO: {name}")
            print("-" * 30)
            
            # Calcular malha fechada
            # T(s) = C(s)G(s)/(1 + C(s)G(s))
            open_loop = controller * plant
            closed_loop = open_loop / (1 + open_loop)
            
            # Análise individual
            analysis = self.analyze_transient_response_complete(closed_loop, show_derivations=False)
            analyses[name] = analysis
        
        # Comparação
        print("\n📊 COMPARAÇÃO RESUMIDA:")
        print("-" * 30)
        
        for name, analysis in analyses.items():
            print(f"🏷️ {name}:")
            if 'Mp' in analysis.metrics:
                print(f"   Overshoot: {analysis.metrics['Mp']}%")
            if 'tr' in analysis.metrics:
                print(f"   Tempo subida: {analysis.metrics['tr']}")
            if 'ts_2' in analysis.metrics:
                print(f"   Tempo estabelec.: {analysis.metrics['ts_2']}")
        
        return analyses


def demonstrate_performance_analysis():
    """
    Demonstração da análise de desempenho educacional
    """
    print("🎓 DEMONSTRAÇÃO ANÁLISE DE DESEMPENHO")
    print("=" * 50)
    
    analyzer = EducationalPerformanceAnalyzer()
    
    # Sistema de segunda ordem típico
    s = sp.Symbol('s')
    wn, zeta = sp.symbols('omega_n zeta', real=True, positive=True)
    
    # T(s) = ωn²/(s² + 2ζωn*s + ωn²)
    tf_second_order = SymbolicTransferFunction(
        wn**2,
        s**2 + 2*zeta*wn*s + wn**2
    )
    
    print(f"\n📊 Sistema exemplo: T(s) = {tf_second_order}")
    
    # Análise completa
    analysis = analyzer.analyze_transient_response_complete(tf_second_order)
    
    print(f"\n📋 RESULTADO DA ANÁLISE:")
    print(f"   Métricas: {list(analysis.metrics.keys())}")
    print(f"   Interpretações: {len(analysis.physical_interpretations)}")
    print(f"   Recomendações: {len(analysis.design_recommendations)}")
    
    print("\n✅ ANÁLISE DE DESEMPENHO DEMONSTRADA!")

if __name__ == "__main__":
    demonstrate_performance_analysis()
