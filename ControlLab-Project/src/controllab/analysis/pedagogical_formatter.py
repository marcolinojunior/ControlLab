from controllab.analysis.routh_hurwitz import StabilityResult, RouthAnalysisHistory

def format_routh_hurwitz_response(result: StabilityResult, history: RouthAnalysisHistory) -> dict:
    """
    Formats the Routh-Hurwitz analysis result into a pedagogical JSON response.

    Args:
        result: The stability result from the Routh-Hurwitz analysis.
        history: The history of the Routh-Hurwitz analysis.

    Returns:
        A dictionary containing the pedagogical response.
    """
    response = {
        "title": "Análise de Estabilidade de Routh-Hurwitz",
        "polynomial": str(history.polynomial),
        "steps": [],
        "special_cases": [],
        "conclusion": {}
    }

    for step in history.steps:
        response["steps"].append({
            "step": step['step'],
            "type": step['type'],
            "description": step['description'],
            "data": str(step['data']),
            "explanation": step['explanation']
        })

    for case in history.special_cases:
        response["special_cases"].append({
            "type": case['type'],
            "row": case['row'],
            "treatment": case['treatment'],
            "result": str(case['result'])
        })

    response["conclusion"] = {
        "is_stable": result.is_stable,
        "unstable_poles_count": result.unstable_poles_count,
        "summary": history.stability_conclusion
    }

    return response
