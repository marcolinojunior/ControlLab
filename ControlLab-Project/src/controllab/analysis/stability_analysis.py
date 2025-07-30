import sympy as sp
from typing import Dict, List, Any
import warnings

# Importações diretas e limpas. Se alguma destas falhar,
# é um sintoma de um problema de ambiente que PRECISAMOS de saber.
from .routh_hurwitz import RouthHurwitzAnalyzer, StabilityResult
from .root_locus import RootLocusAnalyzer
from .frequency_response import calculate_gain_phase_margins, apply_nyquist_criterion
from .stability_utils import StabilityValidator
from .pedagogical_formatter import format_routh_hurwitz_response, format_root_locus_response, format_bode_response, format_nyquist_response


class ComprehensiveStabilityReport:
    """
    Relatório pedagógico completo de análise de estabilidade

    Esta classe organiza todos os resultados de análise em um formato
    educacional, mostrando conexões entre métodos e explicações detalhadas.
    """

    def __init__(self):
        self.system_info = {}
        self.detailed_reports: Dict[str, Dict] = {}
        self.conclusions: List[Dict] = []
        self.educational_notes: List[Dict] = []

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

    def add_analysis_report(self, method: str, report: dict):
        """Adiciona um relatório pedagógico já formatado."""
        self.detailed_reports[method] = report

    def add_conclusion(self, method: str, conclusion: str, confidence: str = "Alta"):
        """Adiciona conclusão de método"""
        self.conclusions.append({
            'method': method,
            'conclusion': conclusion,
            'confidence': confidence
        })

    def get_detailed_analysis(self) -> str:
        analysis_str = ""
        for method, report in self.detailed_reports.items():
            analysis_str += f"--- ANÁLISE DE {method.upper()} ---\n"
            # Formata o dicionário para uma string bonita (pode ser melhorado)
            for step in report.get('steps', []):
                analysis_str += f"  - {step['title']}: {step['explanation']}\n"
            analysis_str += "\n"
        return analysis_str

class StabilityAnalysisEngine:
    """
    Motor principal de análise de estabilidade

    Esta classe coordena todos os módulos de análise, fornecendo
    uma interface unificada para análise completa de estabilidade.
    """

    def __init__(self):
        self.routh_analyzer = RouthHurwitzAnalyzer()
        self.root_locus_analyzer = RootLocusAnalyzer()
        self.validator = StabilityValidator()

    def comprehensive_analysis(self, tf_obj, **kwargs) -> ComprehensiveStabilityReport:
        report = ComprehensiveStabilityReport()
        report.add_system_info(tf_obj)
        char_poly = report.system_info['characteristic_polynomial']

        # --- FLUXO PARA ROUTH-HURWITZ ---
        if self.routh_analyzer:
            try:
                # 1. Chamar o "motor de cálculo" para obter os dados brutos
                stability_result, history, polynomial = self.routh_analyzer.analyze(char_poly)

                # 2. Passar os dados brutos para o "tradutor pedagógico"
                pedagogical_report = format_routh_hurwitz_response(stability_result, history, polynomial)

                # 3. Adicionar o relatório final formatado ao nosso contentor de dados
                report.add_analysis_report('Routh-Hurwitz', pedagogical_report)
                report.add_conclusion("Routh-Hurwitz", pedagogical_report['conclusion'])

            except Exception as e:
                report.add_conclusion("Routh-Hurwitz", f"Erro na análise: {e}", "Baixa")

        # --- FLUXO PARA ROOT LOCUS (APLIQUE O MESMO PADRÃO) ---
        if self.root_locus_analyzer:
            try:
                # 1. Chamar o "motor"
                features, is_stable = self.root_locus_analyzer.get_locus_features(tf_obj)

                # 2. Chamar o "tradutor"
                pedagogical_report = format_root_locus_response(features, is_stable)

                # 3. Adicionar ao relatório
                report.add_analysis_report('Root Locus', pedagogical_report)
                report.add_conclusion("Root Locus", pedagogical_report['conclusion'])

            except Exception as e:
                report.add_conclusion("Root Locus", f"Erro na análise: {e}", "Baixa")

        return report

def validate_stability_methods(tf_obj, show_steps=True):
    engine = StabilityAnalysisEngine()
    report = engine.comprehensive_analysis(tf_obj, show_all_steps=show_steps)

    results = {}

    # Routh-Hurwitz
    if 'Routh-Hurwitz' in report.detailed_reports:
        routh_conclusion = report.detailed_reports['Routh-Hurwitz']['conclusion']
        results['routh_hurwitz'] = {'is_stable': routh_conclusion == 'Sistema estável'}

    # Root Locus
    if 'Root Locus' in report.detailed_reports:
        root_locus_conclusion = report.detailed_reports['Root Locus']['conclusion']
        results['root_analysis'] = {'is_stable': root_locus_conclusion == 'Sistema estável'}

    return results
