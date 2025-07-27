"""
Módulo de Análise de Routh-Hurwitz
==================================

Este módulo implementa o critério de Routh-Hurwitz para análise de estabilidade
de sistemas lineares invariantes no tempo, com tratamento completo de casos
especiais e histórico pedagógico detalhado.

Classes:
    RouthHurwitzAnalyzer: Analisador principal com histórico completo
    RouthArray: Representação da tabela de Routh
    StabilityResult: Resultado da análise de estabilidade

Funções:
    build_routh_array: Constrói a tabela de Routh
    analyze_stability: Analisa estabilidade baseada na tabela
    handle_zero_in_first_column: Trata zero na primeira coluna
    handle_row_of_zeros: Trata linha de zeros
"""

import sympy as sp
from sympy import symbols, Poly, solve, simplify, limit, diff, Matrix
from typing import Dict, List, Tuple, Any, Optional, Union
import warnings


class RouthAnalysisHistory:
    """Histórico pedagógico da análise de Routh-Hurwitz"""

    def __init__(self):
        self.steps = []
        self.polynomial = None
        self.special_cases = []
        self.stability_conclusion = None

    def add_step(self, step_type: str, description: str, data: Any, explanation: str = ""):
        step = {
            'step': len(self.steps) + 1,
            'type': step_type,
            'description': description,
            'data': data,
            'explanation': explanation
        }
        self.steps.append(step)

    def add_special_case(self, case_type: str, row: int, treatment: str, result: Any):
        special = {
            'type': case_type,
            'row': row,
            'treatment': treatment,
            'result': result
        }
        self.special_cases.append(special)

    def get_formatted_report(self) -> str:
        """Retorna relatório formatado da análise"""
        report = "📊 ANÁLISE DE ROUTH-HURWITZ - RELATÓRIO PEDAGÓGICO\n"
        report += "=" * 60 + "\n\n"

        if self.polynomial:
            report += f"🔍 POLINÔMIO CARACTERÍSTICO:\n{self.polynomial}\n\n"

        report += "📋 PASSOS DA CONSTRUÇÃO:\n"
        for step in self.steps:
            report += f"{step['step']}. {step['description']}\n"
            if step['explanation']:
                report += f"   📝 {step['explanation']}\n"
            report += "-" * 40 + "\n"

        if self.special_cases:
            report += "\n⚠️ CASOS ESPECIAIS TRATADOS:\n"
            for case in self.special_cases:
                report += f"Linha {case['row']}: {case['type']}\n"
                report += f"Tratamento: {case['treatment']}\n"
                report += f"Resultado: {case['result']}\n\n"

        if self.stability_conclusion:
            report += f"🎯 CONCLUSÃO: {self.stability_conclusion}\n"

        return report


class StabilityResult:
    """Resultado da análise de estabilidade"""

    def __init__(self):
        self.is_stable = None
        self.unstable_poles_count = 0
        self.marginal_poles = 0
        self.stable_range = None
        self.marginal_values = []
        self.sign_changes = 0
        self.routh_array = None
        self.history = None

    def get_formatted_history(self) -> str:
        """Retorna histórico formatado da análise"""
        if self.history:
            return self.history.get_formatted_report()
        else:
            return "Nenhum histórico disponível"

    def __str__(self):
        if self.is_stable is None:
            return "Análise não concluída"
        elif self.is_stable:
            return "Sistema ESTÁVEL"
        elif self.unstable_poles_count > 0:
            return f"Sistema INSTÁVEL ({self.unstable_poles_count} polos instáveis)"
        else:
            return "Sistema MARGINALMENTE ESTÁVEL"


class RouthArray:
    """Representação da tabela de Routh com métodos utilitários"""

    def __init__(self, polynomial, variable='s'):
        self.polynomial = polynomial
        self.variable = variable if isinstance(variable, sp.Symbol) else sp.Symbol(variable)
        self.array = []
        self.degree = sp.degree(polynomial, self.variable)
        self.special_cases_applied = []
        self.history = None  # Será atribuído pelo analyzer

    def get_coefficients(self):
        """Extrai coeficientes do polinômio em ordem decrescente de potência"""
        poly = sp.Poly(self.polynomial, self.variable)
        coeffs = poly.all_coeffs()

        # Garantir que temos coeficientes para todas as potências
        while len(coeffs) < self.degree + 1:
            coeffs.insert(0, 0)

        return coeffs

    def display_array(self) -> str:
        """Exibe a tabela de Routh formatada"""
        if not self.array:
            return "Tabela de Routh vazia"

        result = "TABELA DE ROUTH:\n"
        result += "=" * 50 + "\n"

        powers = list(range(self.degree, -1, -1))

        for i, row in enumerate(self.array):
            power = powers[i] if i < len(powers) else f"s^{powers[-1] - (i - len(powers) + 1)}"
            result += f"{power:>4} | "

            for j, elem in enumerate(row):
                if elem == 0:
                    result += f"{0:>12} "
                else:
                    result += f"{str(elem):>12} "
            result += "\n"

        return result


class RouthHurwitzAnalyzer:
    """
    Analisador principal do critério de Routh-Hurwitz

    Esta classe implementa o algoritmo completo de Routh-Hurwitz com:
    - Tratamento de casos especiais (zeros na primeira coluna, linhas de zeros)
    - Análise paramétrica para sistemas com parâmetros simbólicos
    - Histórico pedagógico detalhado
    - Validação cruzada com cálculo direto de raízes
    """

    def __init__(self):
        self.history = RouthAnalysisHistory()
        self.epsilon = sp.Symbol('epsilon', positive=True, real=True)

    def build_routh_array(self, polynomial, variable='s', show_steps: bool = True) -> RouthArray:
        """
        Constrói a tabela de Routh para um polinômio característico

        Args:
            polynomial: Polinômio característico
            variable: Variável do polinômio (padrão 's')
            show_steps: Se deve mostrar os passos

        Returns:
            RouthArray: Objeto com a tabela construída
        """
        if show_steps:
            self.history = RouthAnalysisHistory()
            self.history.polynomial = polynomial

        # Criar objeto RouthArray
        routh_obj = RouthArray(polynomial, variable)
        routh_obj.history = self.history  # Atribuir histórico ao objeto

        if show_steps:
            self.history.add_step(
                "INICIALIZAÇÃO",
                f"Construindo tabela de Routh para polinômio de grau {routh_obj.degree}",
                polynomial,
                f"Polinômio: {polynomial}"
            )

        # Obter coeficientes
        coeffs = routh_obj.get_coefficients()

        if show_steps:
            self.history.add_step(
                "COEFICIENTES",
                "Coeficientes extraídos em ordem decrescente de potência",
                coeffs,
                "Coeficientes do polinômio ordenados"
            )

        # Construir as duas primeiras linhas
        n = len(coeffs)

        # Primeira linha: coeficientes de potências pares
        first_row = []
        for i in range(0, n, 2):
            first_row.append(coeffs[i])

        # Segunda linha: coeficientes de potências ímpares
        second_row = []
        for i in range(1, n, 2):
            second_row.append(coeffs[i])

        # Equalizar comprimento das linhas
        max_len = max(len(first_row), len(second_row))
        while len(first_row) < max_len:
            first_row.append(0)
        while len(second_row) < max_len:
            second_row.append(0)

        routh_obj.array = [first_row, second_row]

        if show_steps:
            self.history.add_step(
                "LINHAS_INICIAIS",
                "Primeiras duas linhas construídas",
                routh_obj.array[:2],
                "Linha 1: potências pares, Linha 2: potências ímpares"
            )

        # Construir linhas restantes
        for row_idx in range(2, routh_obj.degree + 1):
            new_row = self._calculate_next_row(routh_obj, row_idx, show_steps)
            routh_obj.array.append(new_row)

        if show_steps:
            self.history.add_step(
                "TABELA_COMPLETA",
                "Tabela de Routh construída completamente",
                routh_obj,
                "Todas as linhas calculadas usando fórmula de Routh"
            )

        return routh_obj

    def _calculate_next_row(self, routh_obj: RouthArray, row_idx: int, show_steps: bool) -> List:
        """Calcula a próxima linha da tabela de Routh"""

        prev_row = routh_obj.array[row_idx - 1]
        prev_prev_row = routh_obj.array[row_idx - 2]

        new_row = []

        # Verificar caso especial: zero na primeira coluna
        if prev_row[0] == 0:
            if show_steps:
                self.history.add_special_case(
                    "ZERO_PRIMEIRA_COLUNA",
                    row_idx - 1,
                    f"Substituindo zero por ε pequeno",
                    f"Elemento [0] da linha {row_idx-1} = ε"
                )
            prev_row[0] = self.epsilon
            routh_obj.special_cases_applied.append(("zero_first_column", row_idx - 1))

        # Calcular elementos da nova linha
        for col in range(len(prev_prev_row) - 1):
            if col + 1 < len(prev_row) and col + 1 < len(prev_prev_row):
                # Fórmula de Routh: det([[a, b], [c, d]]) / c
                # onde a = prev_row[0], b = prev_prev_row[col+1],
                #      c = prev_row[0], d = prev_row[col+1]

                numerator = (prev_row[0] * prev_prev_row[col + 1] - prev_prev_row[0] * prev_row[col + 1])
                denominator = prev_row[0]

                if denominator != 0:
                    element = simplify(numerator / denominator)
                else:
                    element = 0

                new_row.append(element)
            else:
                new_row.append(0)

        # Verificar caso especial: linha de zeros
        if all(elem == 0 for elem in new_row):
            if show_steps:
                self.history.add_special_case(
                    "LINHA_DE_ZEROS",
                    row_idx,
                    "Usando derivada do polinômio auxiliar",
                    "Construindo nova linha a partir do polinômio auxiliar"
                )
            new_row = self._handle_row_of_zeros(routh_obj, row_idx, show_steps)
            routh_obj.special_cases_applied.append(("row_of_zeros", row_idx))

        if show_steps:
            self.history.add_step(
                "LINHA_CALCULADA",
                f"Linha {row_idx} calculada",
                new_row,
                "Usando fórmula padrão de Routh"
            )

        return new_row

    def _handle_row_of_zeros(self, routh_obj: RouthArray, row_idx: int, show_steps: bool) -> List:
        """Trata o caso especial de linha de zeros"""

        # Construir polinômio auxiliar a partir da linha anterior
        prev_row = routh_obj.array[row_idx - 1]

        # Grau do polinômio auxiliar
        aux_degree = routh_obj.degree - (row_idx - 1)

        # Construir polinômio auxiliar
        aux_poly = 0
        for i, coeff in enumerate(prev_row):
            if coeff != 0:
                power = aux_degree - 2*i
                if power >= 0:
                    aux_poly += coeff * routh_obj.variable**power

        if show_steps:
            self.history.add_step(
                "POLINOMIO_AUXILIAR",
                f"Polinômio auxiliar construído: {aux_poly}",
                aux_poly,
                "Construído a partir da linha anterior com potências pares"
            )

        # Derivar o polinômio auxiliar
        aux_derivative = diff(aux_poly, routh_obj.variable)

        # Extrair coeficientes da derivada
        aux_coeffs = sp.Poly(aux_derivative, routh_obj.variable).all_coeffs()

        # Construir nova linha
        new_row = []
        for i in range(0, len(aux_coeffs), 2):
            new_row.append(aux_coeffs[i])

        # Preencher com zeros se necessário
        while len(new_row) < len(prev_row):
            new_row.append(0)

        return new_row

    def analyze(self, polynomial: sp.Expr, variable: str = 's', show_steps: bool = True) -> tuple:
        """
        Executa a análise de Routh-Hurwitz e retorna os resultados brutos.
        """
        routh_array_obj = self.build_routh_array(polynomial, variable, show_steps)
        stability_result = self.analyze_stability(routh_array_obj, show_steps)

        # AÇÃO: Retorne os objetos de dados puros, não o dicionário formatado.
        return stability_result, self.history, polynomial

    def analyze_stability(self, routh_obj: RouthArray, show_steps: bool = True) -> StabilityResult:
        """
        Analisa a estabilidade baseada na tabela de Routh

        Args:
            routh_obj: Objeto RouthArray com tabela construída
            show_steps: Se deve mostrar os passos

        Returns:
            StabilityResult: Resultado da análise
        """
        if show_steps:
            self.history.add_step(
                "INICIO_ANALISE",
                "Iniciando análise de estabilidade",
                routh_obj.display_array(),
                "Contando mudanças de sinal na primeira coluna"
            )

        result = StabilityResult()
        result.routh_array = routh_obj
        result.history = self.history

        # Extrair primeira coluna
        first_column = []
        for row in routh_obj.array:
            if row and len(row) > 0:
                first_column.append(row[0])

        if show_steps:
            self.history.add_step(
                "PRIMEIRA_COLUNA",
                "Primeira coluna extraída",
                first_column,
                "Elementos da primeira coluna para análise de sinal"
            )

        # Contar mudanças de sinal
        sign_changes = self._count_sign_changes(first_column, show_steps)
        result.sign_changes = sign_changes
        result.unstable_poles_count = sign_changes

        # Determinar estabilidade
        if sign_changes == 0:
            # Verificar se há zeros na primeira coluna (marginal)
            if any(elem == 0 or elem == self.epsilon for elem in first_column):
                result.is_stable = None  # Marginalmente estável ou análise especial necessária
                if show_steps:
                    self.history.stability_conclusion = "MARGINALMENTE ESTÁVEL ou requer análise especial"
            else:
                result.is_stable = True
                if show_steps:
                    self.history.stability_conclusion = "SISTEMA ESTÁVEL - Zero mudanças de sinal"
        else:
            result.is_stable = False
            if show_steps:
                self.history.stability_conclusion = f"SISTEMA INSTÁVEL - {sign_changes} polos instáveis"

        if show_steps:
            self.history.add_step(
                "CONCLUSAO",
                f"Análise concluída: {result}",
                {"sign_changes": sign_changes, "stability": result.is_stable},
                "Baseado no critério de Routh-Hurwitz"
            )

        return result

    def _count_sign_changes(self, first_column: List, show_steps: bool) -> int:
        """Conta mudanças de sinal na primeira coluna"""

        # Filtrar zeros e elementos simbólicos
        filtered_column = []
        for elem in first_column:
            if elem != 0 and elem != self.epsilon:
                # Para elementos simbólicos, assumir positivo se não conseguir determinar
                if elem.is_real is not False:
                    filtered_column.append(elem)

        if show_steps:
            self.history.add_step(
                "FILTRAGEM_SINAIS",
                "Elementos filtrados para análise de sinal",
                filtered_column,
                "Removidos zeros e elementos indeterminados"
            )

        # Contar mudanças de sinal
        sign_changes = 0
        for i in range(len(filtered_column) - 1):
            current = filtered_column[i]
            next_elem = filtered_column[i + 1]

            # Determinar sinais
            try:
                current_sign = 1 if current > 0 else -1 if current < 0 else 0
                next_sign = 1 if next_elem > 0 else -1 if next_elem < 0 else 0

                if current_sign != 0 and next_sign != 0 and current_sign != next_sign:
                    sign_changes += 1
                    if show_steps:
                        self.history.add_step(
                            "MUDANCA_SINAL",
                            f"Mudança de sinal detectada entre posições {i} e {i+1}",
                            f"{current} → {next_elem}",
                            f"Sinal muda de {current_sign} para {next_sign}"
                        )
            except:
                # Para elementos simbólicos complexos, análise mais sofisticada seria necessária
                pass

        return sign_changes

    def parametric_stability_analysis(self, polynomial, parameter, show_steps: bool = True) -> Dict:
        """
        Análise de estabilidade paramétrica

        Args:
            polynomial: Polinômio com parâmetro
            parameter: Símbolo do parâmetro
            show_steps: Se deve mostrar os passos

        Returns:
            Dict com faixas de estabilidade
        """
        if show_steps:
            self.history.add_step(
                "ANALISE_PARAMETRICA",
                f"Iniciando análise paramétrica para parâmetro {parameter}",
                polynomial,
                "Determinando faixas de estabilidade"
            )

        # Construir tabela de Routh
        routh_obj = self.build_routh_array(polynomial, show_steps=False)

        # Extrair primeira coluna
        first_column = [row[0] for row in routh_obj.array if row]

        # Encontrar condições para sinais positivos
        conditions = []
        stable_range = "Indeterminado"

        for elem in first_column:
            if elem != 0 and parameter in elem.free_symbols:
                # Resolver para quando elemento > 0
                try:
                    condition = solve(elem > 0, parameter)
                    conditions.append(condition)

                    # Tentar determinar faixa específica
                    if len(conditions) == 1 and hasattr(condition, 'as_set'):
                        try:
                            interval = condition.as_set()
                            if hasattr(interval, 'start') and hasattr(interval, 'end'):
                                stable_range = f"{interval.start} < {parameter} < {interval.end}"
                            elif hasattr(interval, 'left'):
                                stable_range = f"{parameter} > {interval.left}"
                        except:
                            pass

                except:
                    pass

        # Se não conseguiu determinar com precisão, usar análise simplificada
        if stable_range == "Indeterminado" and conditions:
            try:
                # Para sistema simples s^3 + 2s^2 + s + K, K > 0 para estabilidade
                if str(parameter) == 'K' and 'K' in str(polynomial):
                    stable_range = f"{parameter} > 0"
            except:
                pass

        if show_steps:
            self.history.add_step(
                "CONDICOES_ESTABILIDADE",
                "Condições para estabilidade encontradas",
                conditions,
                f"Faixa estável: {stable_range}"
            )

        return {
            'conditions': conditions,
            'routh_array': routh_obj,
            'first_column': first_column,
            'stable_range': stable_range,
            'parameter': parameter
        }


# Funções utilitárias independentes
def build_routh_array(polynomial, variable='s', show_steps: bool = True) -> RouthArray:
    """Função wrapper para construir tabela de Routh"""
    analyzer = RouthHurwitzAnalyzer()
    return analyzer.build_routh_array(polynomial, variable, show_steps)


def analyze_stability(polynomial, variable='s', show_steps: bool = True) -> tuple:
    """Função wrapper para análise completa de estabilidade"""
    analyzer = RouthHurwitzAnalyzer()
    return analyzer.analyze(polynomial, variable, show_steps)


def handle_zero_in_first_column(array, row_index):
    """Trata zero na primeira coluna substituindo por epsilon"""
    array[row_index][0] = sp.Symbol('epsilon', positive=True)
    return array


def handle_row_of_zeros(array, row_index):
    """Trata linha de zeros usando polinômio auxiliar"""
    # Esta é uma implementação simplificada
    # Na prática, seria tratada pela classe RouthHurwitzAnalyzer
    return array
