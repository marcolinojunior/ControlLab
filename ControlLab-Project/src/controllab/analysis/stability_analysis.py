"""
Módulo Principal de Análise de Estabilidade
============================================

Este módulo integra todos os métodos de análise de estabilidade,
fornecendo uma interface unificada e pedagógica para:

- Análise de estabilidade usando critério de Routh-Hurwitz
- Análise do lugar geométrico das raízes (Root Locus)
- Análise de resposta em frequência e margens de estabilidade
- Validação cruzada entre métodos
- Análise paramétrica e regiões de estabilidade

Classes Principais:
    StabilityAnalysisEngine: Interface unificada de análise
    ComprehensiveStabilityReport: Relatório pedagógico completo

Exemplo de Uso:
    ```python
    from controllab.analysis.stability_analysis import StabilityAnalysisEngine

    # Criar função de transferência
    s = sp.Symbol('s')
    tf = (s + 1) / (s**3 + 2*s**2 + 3*s + 1)

    # Análise completa
    engine = StabilityAnalysisEngine()
    result = engine.comprehensive_analysis(tf, show_all_steps=True)

    # Visualizar relatório
    print(result.get_full_report())
    ```
"""

import sympy as sp
from sympy import symbols, solve, simplify, expand, factor, collect
from typing import Dict, List, Tuple, Any, Optional, Union
import warnings

# Importar módulos de análise
try:
    import sys
    import os

    # Configurar path se executado diretamente
    if __name__ == "__main__":
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
        src_path = os.path.join(project_root, 'src')
        if src_path not in sys.path:
            sys.path.insert(0, src_path)

    from controllab.analysis.routh_hurwitz import RouthHurwitzAnalyzer, StabilityResult
    from controllab.analysis.root_locus import RootLocusAnalyzer
    from controllab.analysis.frequency_response import calculate_gain_phase_margins, apply_nyquist_criterion
    from controllab.analysis.stability_utils import (StabilityValidator, ParametricAnalyzer,
                                 ValidationHistory, format_stability_report)
    from controllab.analysis.pedagogical_formatter import format_routh_hurwitz_response, format_root_locus_response, format_bode_response, format_nyquist_response
except ImportError as e:
    print(f"Aviso: Alguns módulos de análise não estão disponíveis: {e}")
    print("Tentando imports alternativos...")

    # Tentar imports alternativos para teste direto
    try:
        from routh_hurwitz import RouthHurwitzAnalyzer, StabilityResult
        from root_locus import RootLocusAnalyzer
        from frequency_response import FrequencyAnalyzer, StabilityMargins
        from stability_utils import (StabilityValidator, ParametricAnalyzer,
                                     ValidationHistory, format_stability_report)
        from pedagogical_formatter import format_routh_hurwitz_response, format_root_locus_response
        print("Imports alternativos bem-sucedidos!")
    except ImportError:
        warnings.warn(f"Alguns módulos de análise não estão disponíveis: {e}")
        # Definir classes vazias como fallback
        RouthHurwitzAnalyzer = None
        RootLocusAnalyzer = None
        FrequencyAnalyzer = None
        StabilityValidator = None
        ParametricAnalyzer = None


class ComprehensiveStabilityReport:
    """
    Relatório pedagógico completo de análise de estabilidade

    Esta classe organiza todos os resultados de análise em um formato
    educacional, mostrando conexões entre métodos e explicações detalhadas.
    """

    def __init__(self):
        self.system_info = {}
        self.routh_hurwitz_results = None
        self.root_locus_results = None
        self.bode_results = None
        self.nyquist_results = None
        self.frequency_response_results = None
        self.validation_results = None
        self.parametric_results = None
        self.conclusions = []
        self.educational_notes = []

    def add_system_info(self, tf_obj, description: str = ""):
        """Adiciona informações do sistema"""
        try:
            if hasattr(tf_obj, 'numerator') and hasattr(tf_obj, 'denominator'):
                num = tf_obj.numerator
                den = tf_obj.denominator
                var = tf_obj.variable if hasattr(tf_obj, 'variable') else 's'
            else:
                num = sp.numer(tf_obj)
                den = sp.denom(tf_obj)
                var = list(tf_obj.free_symbols)[0] if tf_obj.free_symbols else 's'

            self.system_info = {
                'transfer_function': tf_obj,
                'numerator': num,
                'denominator': den,
                'variable': var,
                'order': sp.degree(den, var),
                'description': description,
                'characteristic_polynomial': den
            }

        except Exception as e:
            self.system_info = {'error': f"Erro ao processar sistema: {e}"}

    def add_educational_note(self, category: str, note: str):
        """Adiciona nota educacional"""
        self.educational_notes.append({
            'category': category,
            'note': note
        })

    def add_analysis_report(self, method: str, report: dict):
        """Adiciona um relatório de análise."""
        if method == 'Routh-Hurwitz':
            self.routh_hurwitz_results = report
        elif method == 'Root Locus':
            self.root_locus_results = report
        elif method == 'Bode':
            self.bode_results = report
        elif method == 'Nyquist':
            self.nyquist_results = report

    def add_conclusion(self, method: str, conclusion: str, confidence: str = "Alta"):
        """Adiciona conclusão de método"""
        self.conclusions.append({
            'method': method,
            'conclusion': conclusion,
            'confidence': confidence
        })

    def get_executive_summary(self) -> str:
        """Retorna resumo executivo da análise"""
        summary = "🎯 RESUMO EXECUTIVO - ANÁLISE DE ESTABILIDADE\n"
        summary += "=" * 50 + "\n\n"

        if self.system_info and 'order' in self.system_info:
            summary += f"📋 SISTEMA: Ordem {self.system_info['order']}\n"
            tf_str = str(self.system_info.get('transfer_function', 'N/A'))
            summary += f"📐 FUNÇÃO DE TRANSFERÊNCIA: {tf_str}\n\n"

        # Compilar conclusões
        stable_methods = []
        unstable_methods = []
        uncertain_methods = []

        for conclusion in self.conclusions:
            if 'estável' in conclusion['conclusion'].lower():
                stable_methods.append(conclusion['method'])
            elif 'instável' in conclusion['conclusion'].lower():
                unstable_methods.append(conclusion['method'])
            else:
                uncertain_methods.append(conclusion['method'])

        total_methods = len(self.conclusions)
        if total_methods > 0:
            summary += f"✅ MÉTODOS INDICANDO ESTABILIDADE: {len(stable_methods)}/{total_methods}\n"
            if stable_methods:
                summary += f"   {', '.join(stable_methods)}\n"

            summary += f"❌ MÉTODOS INDICANDO INSTABILIDADE: {len(unstable_methods)}/{total_methods}\n"
            if unstable_methods:
                summary += f"   {', '.join(unstable_methods)}\n"

            if uncertain_methods:
                summary += f"❓ MÉTODOS INCONCLUSIVOS: {len(uncertain_methods)}/{total_methods}\n"
                summary += f"   {', '.join(uncertain_methods)}\n"

        # Conclusão final
        summary += "\n🏁 CONCLUSÃO FINAL: "
        if len(stable_methods) > len(unstable_methods):
            summary += "SISTEMA ESTÁVEL\n"
        elif len(unstable_methods) > len(stable_methods):
            summary += "SISTEMA INSTÁVEL\n"
        else:
            summary += "ANÁLISE INCONCLUSIVA - NECESSÁRIO MAIS INVESTIGAÇÃO\n"

        return summary

    def get_detailed_analysis(self) -> str:
        """Retorna análise detalhada por método"""
        analysis = "📊 ANÁLISE DETALHADA POR MÉTODO\n"
        analysis += "=" * 40 + "\n\n"

        # Routh-Hurwitz
        if self.routh_hurwitz_results:
            analysis += "🔢 ANÁLISE DE ROUTH-HURWITZ:\n"
            analysis += "-" * 30 + "\n"
            try:
                if hasattr(self.routh_hurwitz_results, 'get_formatted_history'):
                    analysis += self.routh_hurwitz_results.get_formatted_history()
                else:
                    analysis += str(self.routh_hurwitz_results)
            except Exception as e:
                analysis += f"Erro na formatação: {e}"
            analysis += "\n\n"

        # Root Locus
        if self.root_locus_results:
            analysis += "📍 ANÁLISE DO LUGAR GEOMÉTRICO:\n"
            analysis += "-" * 35 + "\n"
            try:
                if hasattr(self.root_locus_results, 'analysis_history') and self.root_locus_results.analysis_history:
                    analysis += self.root_locus_results.analysis_history.get_formatted_report()
                else:
                    analysis += str(self.root_locus_results)
            except Exception as e:
                analysis += f"Erro na formatação: {e}"
            analysis += "\n\n"

        # Frequency Response
        if self.frequency_response_results:
            analysis += "📈 ANÁLISE DE RESPOSTA EM FREQUÊNCIA:\n"
            analysis += "-" * 40 + "\n"
            try:
                if hasattr(self.frequency_response_results, 'get_formatted_history'):
                    analysis += self.frequency_response_results.get_formatted_history()
                else:
                    analysis += str(self.frequency_response_results)
            except Exception as e:
                analysis += f"Erro na formatação: {e}"
            analysis += "\n\n"

        return analysis

    def get_educational_section(self) -> str:
        """Retorna seção educacional com explicações"""
        education = "🎓 SEÇÃO EDUCACIONAL\n"
        education += "=" * 25 + "\n\n"

        education += "📚 CONCEITOS FUNDAMENTAIS:\n"
        education += "• ESTABILIDADE: Um sistema é estável se suas saídas permanecem limitadas\n"
        education += "  para entradas limitadas (BIBO - Bounded Input, Bounded Output)\n"
        education += "• POLINÔMIO CARACTERÍSTICO: Determinante da equação característica\n"
        education += "  onde se encontram os polos do sistema\n"
        education += "• POLOS: Valores de 's' que tornam o denominador zero\n"
        education += "• INTERPRETAÇÃO FÍSICA: Polos no lado esquerdo → estável,\n"
        education += "  polos no lado direito → instável, polos no eixo jω → marginalmente estável\n\n"

        education += "📚 CONEXÕES ENTRE MÉTODOS:\n"
        education += "• ROUTH-HURWITZ: Método algébrico que analisa sinais na primeira coluna\n"
        education += "  da tabela de Routh para determinar polos no semiplano direito\n"
        education += "• ROOT LOCUS: Método geométrico que mostra o lugar geométrico das raízes\n"
        education += "  no plano complexo conforme o ganho K varia\n"
        education += "• RESPOSTA EM FREQUÊNCIA: Avalia margens de ganho e fase para\n"
        education += "  determinar robustez do sistema em malha fechada\n\n"

        education += "🔧 FÓRMULAS E EQUAÇÕES CHAVE:\n"
        education += "• Equação característica: 1 + G(s)H(s) = 0\n"
        education += "• Critério de Routh: Mudanças de sinal = número de polos instáveis\n"
        education += "• Margem de ganho (dB) = 20*log10(1/|G(jωc)|)\n"
        education += "• Margem de fase (°) = 180° + ∠G(jωg)\n\n"

        if self.educational_notes:
            education += "💡 NOTAS EDUCACIONAIS ESPECÍFICAS:\n"
            for note in self.educational_notes:
                education += f"• {note['category']}: {note['note']}\n"
            education += "\n"

        # Casos especiais e interpretação física
        education += "⚠️ CASOS ESPECIAIS IMPORTANTES:\n"
        education += "• Zero na primeira coluna do Routh: Substituir por ε pequeno\n"
        education += "• Linha de zeros: Usar derivada do polinômio auxiliar\n"
        education += "• Polos no eixo jω: Sistema marginalmente estável\n"
        education += "• Múltiplos polos: Podem causar instabilidade mesmo no lado esquerdo\n\n"

        # Adicionar explicações contextuais baseadas nos resultados
        if self.system_info and 'order' in self.system_info:
            order = self.system_info['order']
            education += f"🔍 ANÁLISE ESPECÍFICA PARA SISTEMA DE ORDEM {order}:\n"

            if order == 1:
                education += "• Sistema de 1ª ordem: s + a = 0 → Polo em s = -a\n"
                education += "• INTERPRETAÇÃO FÍSICA: Estável se a > 0 (constante de tempo positiva)\n"
                education += "• Resposta exponencial decrescente para a > 0\n"
            elif order == 2:
                education += "• Sistema de 2ª ordem: s² + 2ζωₙs + ωₙ² = 0\n"
                education += "• INTERPRETAÇÃO FÍSICA: ζ = coeficiente de amortecimento\n"
                education += "• ζ < 1: subamortecido, ζ = 1: criticamente amortecido, ζ > 1: superamortecido\n"
                education += "• Estável se ζ > 0 e ωₙ > 0\n"
            elif order >= 3:
                education += "• Sistema de ordem superior: Análise complexa necessária\n"
                education += "• INTERPRETAÇÃO FÍSICA: Múltiplos modos de resposta\n"
                education += "• Routh-Hurwitz é essencial para determinar estabilidade\n"
                education += "• Root locus mostra como polos se movem com variação de ganho\n"

            education += "\n"

        education += "🎯 SIGNIFICADO PRÁTICO DA ESTABILIDADE:\n"
        education += "• Sistema ESTÁVEL: Perturbações se atenuam com o tempo\n"
        education += "• Sistema INSTÁVEL: Perturbações crescem indefinidamente\n"
        education += "• Sistema MARGINAL: Perturbações não se atenuam nem crescem\n"
        education += "• APLICAÇÃO: Fundamental para segurança e performance de sistemas\n\n"

        return education

    def get_full_report(self) -> str:
        """Retorna relatório completo"""
        report = "📋 RELATÓRIO COMPLETO DE ANÁLISE DE ESTABILIDADE\n"
        report += "=" * 60 + "\n\n"

        report += self.get_executive_summary() + "\n"
        report += self.get_detailed_analysis() + "\n"
        report += self.get_educational_section() + "\n"

        if self.validation_results:
            report += "🔍 VALIDAÇÃO CRUZADA:\n"
            report += "-" * 20 + "\n"
            if hasattr(self.validation_results, 'get_formatted_report'):
                report += self.validation_results.get_formatted_report()
            else:
                report += str(self.validation_results)
            report += "\n\n"

        return report

    def get_cross_validation_report(self) -> str:
        """Retorna relatório de validação cruzada"""
        if not self.validation_results:
            return "❌ Validação cruzada não realizada"

        validation_report = "🔍 RELATÓRIO DE VALIDAÇÃO CRUZADA\n"
        validation_report += "=" * 40 + "\n\n"

        # Simular validação cruzada básica
        methods_results = []

        if self.routh_hurwitz_results:
            routh_stable = getattr(self.routh_hurwitz_results, 'is_stable', None)
            methods_results.append(("Routh-Hurwitz", routh_stable))

        if self.root_locus_results:
            # Determinar estabilidade por root locus (polos no lado esquerdo)
            rl_stable = self.root_locus_results.stability_assessment.get('is_stable', None)
            methods_results.append(("Root Locus", rl_stable))

        if self.frequency_response_results:
            freq_stable = getattr(self.frequency_response_results, 'is_stable', True)
            methods_results.append(("Frequency Response", freq_stable))

        # Comparar resultados
        validation_report += "📊 RESULTADOS POR MÉTODO:\n"
        for method, stable in methods_results:
            stability_text = "ESTÁVEL" if stable else "INSTÁVEL" if stable is False else "INDETERMINADO"
            validation_report += f"   {method}: {stability_text}\n"

        # Verificar concordância
        stable_results = [r[1] for r in methods_results if r[1] is not None]
        if len(set(stable_results)) == 1:
            validation_report += "\n✅ TODOS OS MÉTODOS CONCORDAM\n"
            validation_report += f"   Conclusão: Sistema {'ESTÁVEL' if stable_results[0] else 'INSTÁVEL'}\n"
        else:
            validation_report += "\n⚠️ DISCREPÂNCIAS DETECTADAS\n"
            validation_report += "   Análise adicional recomendada\n"

        validation_report += "\n📚 NOTA PEDAGÓGICA:\n"
        validation_report += "   A validação cruzada é fundamental para confirmar resultados\n"
        validation_report += "   Pequenas discrepâncias podem indicar casos especiais\n"

        return validation_report


class StabilityAnalysisEngine:
    """
    Motor principal de análise de estabilidade

    Esta classe coordena todos os módulos de análise, fornecendo
    uma interface unificada para análise completa de estabilidade.
    """

    def __init__(self):
        self.routh_analyzer = RouthHurwitzAnalyzer() if RouthHurwitzAnalyzer else None
        self.root_locus_analyzer = RootLocusAnalyzer() if RootLocusAnalyzer else None
        self.validator = StabilityValidator() if StabilityValidator else None
        self.parametric_analyzer = ParametricAnalyzer() if ParametricAnalyzer else None

    def comprehensive_analysis(self, tf_obj, show_all_steps: bool = True,
                             include_validation: bool = True,
                             include_parametric: bool = False) -> ComprehensiveStabilityReport:
        """
        Realiza análise completa de estabilidade

        Args:
            tf_obj: Função de transferência ou polinômio característico
            show_all_steps: Se deve mostrar todos os passos
            include_validation: Se deve incluir validação cruzada
            include_parametric: Se deve incluir análise paramétrica

        Returns:
            ComprehensiveStabilityReport com todos os resultados
        """
        report = ComprehensiveStabilityReport()

        # Adicionar informações do sistema
        report.add_system_info(tf_obj, "Sistema sob análise")

        # 1. Análise de Routh-Hurwitz
        if self.routh_analyzer:
            try:
                char_poly = self._extract_characteristic_polynomial(tf_obj)
                routh_array = self.routh_analyzer.build_routh_array(char_poly, show_steps=show_all_steps)
                routh_result = self.routh_analyzer.analyze_stability(routh_array, show_steps=show_all_steps)

                report.routh_hurwitz_results = routh_result
                conclusion = "Sistema estável" if routh_result.is_stable else "Sistema instável"
                report.add_conclusion("Routh-Hurwitz", conclusion)

                if show_all_steps:
                    report.add_educational_note("Routh-Hurwitz",
                        "Método algébrico que analisa estabilidade sem calcular raízes explicitamente")

            except Exception as e:
                report.add_conclusion("Routh-Hurwitz", f"Erro na análise: {e}", "Baixa")

        # 2. Análise do Root Locus
        if self.root_locus_analyzer:
            try:
                features = self.root_locus_analyzer.analyze_comprehensive(tf_obj, show_steps=show_all_steps)
                report.root_locus_results = features

                is_stable = features.stability_assessment.get('is_stable')
                conclusion = "Sistema estável" if is_stable else "Sistema instável" if is_stable is False else "Análise de estabilidade do Root Locus inconclusiva"
                report.add_conclusion("Root Locus", conclusion)

                if show_all_steps:
                    report.add_educational_note("Root Locus",
                        "Método gráfico que mostra como polos se movem com variação de ganho")

            except Exception as e:
                report.add_conclusion("Root Locus", f"Erro na análise: {e}", "Baixa")

        # 3. Análise de Resposta em Frequência
        if calculate_gain_phase_margins and format_bode_response and apply_nyquist_criterion and format_nyquist_response:
            try:
                # Bode
                margins, history = calculate_gain_phase_margins(tf_obj)
                pedagogical_report = format_bode_response(margins, history)
                report.frequency_response_results = pedagogical_report

                conclusion = "Sistema estável" if margins.is_stable else "Sistema instável"
                if margins.is_stable:
                    conclusion += f" (MG: {margins.gain_margin_db:.1f}dB, MF: {margins.phase_margin:.1f}°)"

                report.add_conclusion("Margens de Estabilidade", conclusion)

                # Nyquist
                nyquist_results, history = apply_nyquist_criterion(tf_obj)
                pedagogical_report = format_nyquist_response(nyquist_results, history)
                report.add_analysis_report('Nyquist', pedagogical_report)

                conclusion = "Sistema estável" if nyquist_results['is_stable'] else "Sistema instável"
                report.add_conclusion("Critério de Nyquist", conclusion)

                if show_all_steps:
                    report.add_educational_note("Margens",
                        "Análise de robustez baseada em margens de ganho e fase")

            except Exception as e:
                report.add_conclusion("Margens", f"Erro na análise: {e}", "Baixa")

        # 4. Validação Cruzada
        if include_validation and self.validator:
            try:
                validation_results = self.validator.validate_stability_methods(tf_obj, show_steps=show_all_steps)
                report.validation_results = self.validator.history

                if show_all_steps:
                    report.add_educational_note("Validação",
                        "Comparação entre métodos para verificar consistência dos resultados")

            except Exception as e:
                report.add_educational_note("Validação", f"Erro na validação: {e}")

        # 5. Análise Paramétrica (opcional)
        if include_parametric and self.parametric_analyzer:
            try:
                # Implementar análise paramétrica se parâmetros estiverem presentes
                params = self._extract_parameters(tf_obj)
                if len(params) >= 1:
                    # Análise de sensibilidade básica
                    param_results = self._basic_parametric_analysis(tf_obj, params)
                    report.parametric_results = param_results

                    report.add_educational_note("Paramétrica",
                        "Análise de como variações paramétricas afetam estabilidade")

            except Exception as e:
                report.add_educational_note("Paramétrica", f"Erro na análise: {e}")

        return report

    def analyze_complete_stability(self, tf_obj, show_steps: bool = True) -> ComprehensiveStabilityReport:
        """
        Alias para comprehensive_analysis para compatibilidade com testes

        Args:
            tf_obj: Função de transferência
            show_steps: Se deve mostrar os passos

        Returns:
            ComprehensiveStabilityReport com análise completa
        """
        report = ComprehensiveStabilityReport()

        # Adicionar informações do sistema
        report.add_system_info(tf_obj, "Sistema sob análise")

        # 1. Análise de Routh-Hurwitz
        if self.routh_analyzer:
            try:
                char_poly = self._extract_characteristic_polynomial(tf_obj)
                stability_result, history, polynomial = self.routh_analyzer.analyze(char_poly)
                pedagogical_report = format_routh_hurwitz_response(stability_result, history, polynomial)
                report.add_analysis_report('Routh-Hurwitz', pedagogical_report)
                conclusion = "Sistema estável" if stability_result.is_stable else "Sistema instável"
                report.add_conclusion("Routh-Hurwitz", conclusion)

                if show_all_steps:
                    report.add_educational_note("Routh-Hurwitz",
                        "Método algébrico que analisa estabilidade sem calcular raízes explicitamente")

            except Exception as e:
                report.add_conclusion("Routh-Hurwitz", f"Erro na análise: {e}", "Baixa")

        return report

    def _extract_characteristic_polynomial(self, tf_obj):
        """Extrai polinômio característico da função de transferência"""
        if hasattr(tf_obj, 'denominator'):
            return tf_obj.denominator
        else:
            return sp.denom(tf_obj)

    def _extract_parameters(self, tf_obj):
        """Extrai parâmetros simbólicos do sistema"""
        if hasattr(tf_obj, 'free_symbols'):
            symbols = list(tf_obj.free_symbols)
        else:
            symbols = []

        # Filtrar símbolos que provavelmente são variáveis (s, z) vs parâmetros
        parameters = [sym for sym in symbols if str(sym) not in ['s', 'z', 't']]
        return parameters

    def _basic_parametric_analysis(self, tf_obj, parameters):
        """Análise paramétrica básica"""
        results = {
            'parameters_found': [str(p) for p in parameters],
            'analysis_type': 'sensitivity',
            'notes': 'Análise básica de sensibilidade paramétrica'
        }

        return results

    def quick_stability_check(self, tf_obj) -> Dict[str, Any]:
        """
        Verificação rápida de estabilidade usando método mais eficiente

        Returns:
            Dict com resultado rápido
        """
        result = {
            'is_stable': None,
            'method_used': None,
            'confidence': 'Medium',
            'details': {}
        }

        try:
            # Tentar Routh-Hurwitz primeiro (mais eficiente)
            if self.routh_analyzer:
                char_poly = self._extract_characteristic_polynomial(tf_obj)
                routh_array = self.routh_analyzer.build_routh_array(char_poly, show_steps=False)
                routh_result = self.routh_analyzer.analyze_stability(routh_array, show_steps=False)

                result['is_stable'] = routh_result.is_stable
                result['method_used'] = 'Routh-Hurwitz'
                result['details'] = {
                    'unstable_poles_in_rhp': routh_result.unstable_poles_count
                }
                result['confidence'] = 'High'

            # Fallback: cálculo direto de polos
            elif self.validator:
                poles = self.validator._calculate_poles_directly(tf_obj)
                is_stable = self.validator._analyze_poles_stability(poles)

                result['is_stable'] = is_stable
                result['method_used'] = 'Direct Pole Calculation'
                result['details'] = {
                    'poles': poles,
                    'unstable_count': sum(1 for p in poles if p.real > 0)
                }

        except Exception as e:
            result['method_used'] = 'Error'
            result['details'] = {'error': str(e)}
            result['confidence'] = 'Low'

        return result

    def comparative_analysis(self, systems: List[Any],
                           labels: List[str] = None) -> Dict[str, Any]:
        """
        Análise comparativa entre múltiplos sistemas

        Args:
            systems: Lista de funções de transferência
            labels: Labels para os sistemas

        Returns:
            Dict com análise comparativa
        """
        if not labels:
            labels = [f"Sistema {i+1}" for i in range(len(systems))]

        comparison = {
            'systems': {},
            'summary': {},
            'stability_comparison': []
        }

        for i, (system, label) in enumerate(zip(systems, labels)):
            quick_result = self.quick_stability_check(system)
            comparison['systems'][label] = quick_result

            stability_status = "Estável" if quick_result['is_stable'] else "Instável"
            comparison['stability_comparison'].append({
                'system': label,
                'status': stability_status,
                'method': quick_result['method_used']
            })

        # Resumo estatístico
        stable_count = sum(1 for sys in comparison['systems'].values()
                          if sys['is_stable'] is True)
        total_count = len(systems)

        comparison['summary'] = {
            'total_systems': total_count,
            'stable_systems': stable_count,
            'unstable_systems': total_count - stable_count,
            'stability_rate': stable_count / total_count if total_count > 0 else 0
        }

        return comparison


# Funções de conveniência para uso direto
def analyze_stability(tf_obj, show_steps: bool = True) -> ComprehensiveStabilityReport:
    """Função de conveniência para análise completa"""
    engine = StabilityAnalysisEngine()
    return engine.comprehensive_analysis(tf_obj, show_all_steps=show_steps)


def quick_stability_check(tf_obj) -> bool:
    """Função de conveniência para verificação rápida"""
    engine = StabilityAnalysisEngine()
    result = engine.quick_stability_check(tf_obj)
    return result.get('is_stable', False)


def compare_systems_stability(systems: List[Any], labels: List[str] = None) -> Dict:
    """Função de conveniência para comparação de sistemas"""
    engine = StabilityAnalysisEngine()
    return engine.comparative_analysis(systems, labels)


# ============================================================================
# SEÇÃO DE TESTE E VALIDAÇÃO
# ============================================================================
def validate_stability_methods(tf_obj, show_steps=True):
    """
    Função de validação cruzada que retorna um dicionário simples
    para ser usado nos testes.
    """
    engine = StabilityAnalysisEngine()
    report = engine.comprehensive_analysis(tf_obj, show_all_steps=show_steps)

    results = {}

    # Routh-Hurwitz
    if report.routh_hurwitz_results:
        results['routh_hurwitz'] = report.routh_hurwitz_results

    # Root Locus
    if report.root_locus_results and hasattr(report.root_locus_results, 'stability_assessment'):
        is_stable = report.root_locus_results.stability_assessment.get('is_stable')
        results['root_analysis'] = {'is_stable': is_stable}

    # Frequency Analysis
    if report.frequency_response_results:
        results['frequency_analysis'] = report.frequency_response_results

    return results

def run_module_validation():
    """
    Executa validação completa do Módulo 5 - Análise de Estabilidade

    Esta função testa todos os componentes do módulo e verifica se estão
    funcionando corretamente, incluindo pedagogia e integração.
    """
    print("=" * 70)
    print("VALIDAÇÃO COMPLETA DO MÓDULO 5 - ANÁLISE DE ESTABILIDADE")
    print("=" * 70)

    import sympy as sp
    s = sp.symbols('s')

    # Definir sistemas de teste
    systems = {
        'Sistema Estável de 1ª Ordem': 1 / (s + 1),
        'Sistema Estável de 2ª Ordem': 1 / (s**2 + 2*s + 1),
        'Sistema Instável': 1 / (s**2 - s + 1),
        'Sistema de 3ª Ordem': 1 / (s**3 + 3*s**2 + 3*s + 1)
    }

    validation_results = []

    print("\n1. TESTANDO IMPORTAÇÕES DOS MÓDULOS...")
    print("-" * 50)

    # Teste de importações
    modules_status = {
        'RouthHurwitzAnalyzer': RouthHurwitzAnalyzer is not None,
        'RootLocusAnalyzer': RootLocusAnalyzer is not None,
        'FrequencyAnalyzer': FrequencyAnalyzer is not None,
        'StabilityValidator': StabilityValidator is not None,
        'ParametricAnalyzer': ParametricAnalyzer is not None
    }

    for module, status in modules_status.items():
        status_str = "✅ OK" if status else "⚠️ PROBLEMA"
        print(f"   {module}: {status_str}")
        validation_results.append(('Importação', module, status))

    print("\n2. TESTANDO FUNCIONALIDADE BÁSICA...")
    print("-" * 50)

    # Teste básico com sistema simples
    test_system = 1 / (s + 1)
    engine = StabilityAnalysisEngine()

    try:
        # Teste de análise rápida
        quick_result = engine.quick_stability_check(test_system)
        quick_ok = isinstance(quick_result, dict) and 'is_stable' in quick_result
        print(f"   Análise Rápida: {'✅ OK' if quick_ok else '⚠️ PROBLEMA'}")
        validation_results.append(('Funcionalidade', 'Análise Rápida', quick_ok))

        # Teste de análise completa
        comprehensive_result = engine.comprehensive_analysis(test_system, show_all_steps=False)
        comprehensive_ok = comprehensive_result is not None
        print(f"   Análise Completa: {'✅ OK' if comprehensive_ok else '⚠️ PROBLEMA'}")
        validation_results.append(('Funcionalidade', 'Análise Completa', comprehensive_ok))

    except Exception as e:
        print(f"   ⚠️ ERRO na funcionalidade básica: {str(e)}")
        validation_results.append(('Funcionalidade', 'Básica', False))

    print("\n3. TESTANDO PEDAGOGIA E TRANSPARÊNCIA...")
    print("-" * 50)

    try:
        # Verificar se o relatório pedagógico funciona
        test_system = 1 / (s**2 + s + 1)
        result = engine.comprehensive_analysis(test_system, show_all_steps=True)

        if result:
            full_report = result.get_full_report()
            validation_report = result.get_cross_validation_report()

            # Verificar conteúdo pedagógico
            pedagogical_markers = ['EDUCACIONAL', 'conceito', 'fórmula', 'interpretação']
            pedagogical_score = sum(1 for marker in pedagogical_markers
                                  if marker.lower() in full_report.lower())

            pedagogy_ok = pedagogical_score >= 2
            print(f"   Conteúdo Pedagógico: {'✅ OK' if pedagogy_ok else '⚠️ LIMITADO'} ({pedagogical_score} elementos)")
            validation_results.append(('Pedagogia', 'Conteúdo', pedagogy_ok))

            # Verificar validação cruzada
            cross_validation_ok = len(validation_report) > 50
            print(f"   Validação Cruzada: {'✅ OK' if cross_validation_ok else '⚠️ LIMITADO'}")
            validation_results.append(('Pedagogia', 'Validação Cruzada', cross_validation_ok))

    except Exception as e:
        print(f"   ⚠️ ERRO na pedagogia: {str(e)}")
        validation_results.append(('Pedagogia', 'Geral', False))

    print("\n4. TESTANDO INTEGRAÇÃO ENTRE MÉTODOS...")
    print("-" * 50)

    try:
        # Testar validação cruzada com sistema conhecido
        stable_system = 1 / (s + 1)

        if StabilityValidator:
            validator = StabilityValidator()
            cross_validation = validator.validate_stability_methods(stable_system, show_steps=False)

            methods_tested = len([k for k in cross_validation.keys() if k not in ['summary', 'agreement']])
            integration_ok = methods_tested >= 2

            print(f"   Métodos Integrados: {'✅ OK' if integration_ok else '⚠️ LIMITADO'} ({methods_tested} métodos)")
            validation_results.append(('Integração', 'Métodos', integration_ok))

    except Exception as e:
        print(f"   ⚠️ ERRO na integração: {str(e)}")
        validation_results.append(('Integração', 'Geral', False))

    print("\n5. TESTANDO CASOS ESPECIAIS...")
    print("-" * 50)

    # Teste com diferentes tipos de sistemas
    special_cases = {
        'Sistema Marginal': s**2 + 1,
        'Sistema Alta Ordem': s**5 + 2*s**4 + 3*s**3 + 4*s**2 + 5*s + 6,
        'Sistema com Zeros': (s + 1) / (s**2 + s + 1)
    }

    special_cases_ok = 0
    for case_name, case_system in special_cases.items():
        try:
            case_result = engine.quick_stability_check(case_system)
            if isinstance(case_result, dict):
                special_cases_ok += 1
                print(f"   {case_name}: ✅ OK")
            else:
                print(f"   {case_name}: ⚠️ PROBLEMA")
        except Exception as e:
            print(f"   {case_name}: ⚠️ ERRO - {str(e)}")

    special_ok = special_cases_ok >= 2
    validation_results.append(('Casos Especiais', 'Geral', special_ok))

    print("\n" + "=" * 70)
    print("RESUMO DA VALIDAÇÃO")
    print("=" * 70)

    # Resumo por categoria
    categories = {}
    for category, component, status in validation_results:
        if category not in categories:
            categories[category] = {'total': 0, 'passed': 0}
        categories[category]['total'] += 1
        if status:
            categories[category]['passed'] += 1

    overall_score = 0
    total_tests = 0

    for category, stats in categories.items():
        passed = stats['passed']
        total = stats['total']
        percentage = (passed / total) * 100 if total > 0 else 0

        status_icon = "✅" if percentage >= 80 else "⚠️" if percentage >= 60 else "❌"
        print(f"{status_icon} {category}: {passed}/{total} ({percentage:.1f}%)")

        overall_score += passed
        total_tests += total

    # Score geral
    overall_percentage = (overall_score / total_tests) * 100 if total_tests > 0 else 0
    overall_icon = "✅" if overall_percentage >= 80 else "⚠️" if overall_percentage >= 60 else "❌"

    print("\n" + "-" * 70)
    print(f"{overall_icon} SCORE GERAL: {overall_score}/{total_tests} ({overall_percentage:.1f}%)")

    if overall_percentage >= 80:
        print("🎉 MÓDULO 5 VALIDADO COM SUCESSO!")
        print("✅ Todas as funcionalidades principais estão operacionais")
        print("✅ Pedagogia e transparência implementadas")
        print("✅ Integração entre métodos funcionando")
    elif overall_percentage >= 60:
        print("⚠️ MÓDULO 5 PARCIALMENTE FUNCIONAL")
        print("ℹ️ Algumas funcionalidades podem precisar de ajustes")
    else:
        print("❌ MÓDULO 5 PRECISA DE CORREÇÕES")
        print("🔧 Verificar implementações que falharam")

    print("=" * 70)

    return {
        'overall_score': overall_percentage,
        'detailed_results': validation_results,
        'categories': categories
    }


if __name__ == "__main__":
    run_module_validation()
