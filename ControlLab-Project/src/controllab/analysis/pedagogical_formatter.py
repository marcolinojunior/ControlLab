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
