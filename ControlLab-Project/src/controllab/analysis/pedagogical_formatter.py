import sympy as sp
# Importe as classes de dados que você vai receber
from .routh_hurwitz import StabilityResult, RouthAnalysisHistory

def format_routh_hurwitz_response(
    stability_result: StabilityResult,
    history: RouthAnalysisHistory,
    polynomial: sp.Expr
) -> dict:
    """
    Recebe os resultados brutos da análise de Routh-Hurwitz e os traduz
    em uma resposta pedagógica estruturada em JSON.
    """
    pedagogical_steps = []

    # Passo 1: Polinômio Inicial (usando o dado bruto 'polynomial')
    pedagogical_steps.append({
        "title": "1. Polinômio Característico",
        "explanation": "A estabilidade do sistema é determinada pelas raízes do polinômio característico.",
        "data": {"polynomial_latex": sp.latex(polynomial) + " = 0"}
    })

    # Passo 2: Construção da Tabela (usando o dado bruto 'history')
    table_step = next((s for s in history.steps if s['type'] == "TABELA_COMPLETA"), None)
    if table_step:
        pedagogical_steps.append({
            "title": "3. Construção da Tabela de Routh",
            "explanation": "A tabela de Routh é um arranjo sistemático dos coeficientes.",
            "data": {"routh_table": [[str(item) for item in row] for row in table_step['data'].array]}
        })

    # Estrutura final (usando o dado bruto 'stability_result')
    response = {
        "conclusion": stability_result.is_stable,
        "summary": f"O sistema é {'estável' if stability_result.is_stable else 'instável'} com {stability_result.sign_changes} trocas de sinal.",
        "steps": pedagogical_steps
    }

    return response


def format_nyquist_response(nyquist_results, history) -> dict:
    """
    Recebe os resultados brutos da análise de Nyquist e os traduz
    em uma resposta pedagógica estruturada em JSON.
    """
    pedagogical_steps = []
    for step in history.steps:
        pedagogical_steps.append({
            "title": step['description'],
            "explanation": step.get('explanation', ''),
            "data": {
                "calculation": str(step['calculation']),
                "result": str(step['result'])
            }
        })

    response = {
        "conclusion": f"Sistema {'estável' if nyquist_results['is_stable'] else 'instável'}.",
        "summary": f"Z = N + P = {nyquist_results['encirclements']} + {nyquist_results['poles_rhp']} = {nyquist_results['zeros_rhp']}",
        "steps": pedagogical_steps,
        "nyquist_analysis": {
            "is_stable": nyquist_results['is_stable'],
            "zeros_rhp": nyquist_results['zeros_rhp'],
            "encirclements": nyquist_results['encirclements'],
            "poles_rhp": nyquist_results['poles_rhp']
        }
    }

    return response


def format_bode_response(margins, history) -> dict:
    """
    Recebe os resultados brutos da análise de Bode e os traduz
    em uma resposta pedagógica estruturada em JSON.
    """
    pedagogical_steps = []
    for step in history.steps:
        pedagogical_steps.append({
            "title": step['description'],
            "explanation": step.get('explanation', ''),
            "data": {
                "calculation": str(step['calculation']),
                "result": str(step['result'])
            }
        })

    response = {
        "conclusion": f"Sistema {'estável' if margins.is_stable else 'instável'}.",
        "summary": f"Margem de Ganho: {margins.gain_margin_db:.2f} dB, Margem de Fase: {margins.phase_margin:.2f}°.",
        "steps": pedagogical_steps,
        "margins": {
            "gain_margin_db": margins.gain_margin_db,
            "phase_margin": margins.phase_margin,
            "gain_crossover_freq": margins.gain_crossover_freq,
            "phase_crossover_freq": margins.phase_crossover_freq,
            "is_stable": margins.is_stable
        }
    }

    return response


def format_root_locus_response(features, history) -> dict:
    """
    Recebe os resultados brutos da análise de Root Locus e os traduz
    em uma resposta pedagógica estruturada em JSON.
    """
    pedagogical_steps = []
    for step in history.steps:
        pedagogical_steps.append({
            "title": f"Regra {step['rule']}: {step['rule_name']}",
            "explanation": step.get('explanation', ''),
            "data": {
                "calculation": str(step['calculation']),
                "result": str(step['result'])
            }
        })

    response = {
        "conclusion": "Análise do Lugar das Raízes completa.",
        "summary": f"O sistema tem {features.num_branches} ramos, {len(features.poles)} polos e {len(features.zeros)} zeros.",
        "steps": pedagogical_steps,
        "features": {
            "poles": [str(p) for p in features.poles],
            "zeros": [str(z) for z in features.zeros],
            "num_branches": features.num_branches,
            "asymptotes": {
                "centroid": str(features.asymptotes.get('centroid')),
                "angles": [str(a) for a in features.asymptotes.get('angles', [])]
            },
            "breakaway_points": [str(p) for p in features.breakaway_points],
            "jw_crossings": [str(c) for c in features.jw_crossings]
        }
    }

    return response
