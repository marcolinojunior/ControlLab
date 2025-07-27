import sympy as sp
from controllab.core.history import OperationHistory
from controllab.core.symbolic_tf import SymbolicTransferFunction
from controllab.design.antiwindup import design_antiwindup_compensation, SaturationLimits
from controllab.design.compensators import design_lead_compensator
from controllab.design.design_utils import calculate_performance_metrics


def test_history_context():
    """
    Testa se o contexto da operação é adicionado ao histórico em caso de erro.
    """
    s = sp.Symbol('s')
    plant = SymbolicTransferFunction(1, s**2 + s + 1)
    controller = SymbolicTransferFunction(1, s**2)  # Controlador inválido para decomposição
    saturation_limits = SaturationLimits(u_min=-1, u_max=1)

    try:
        design_antiwindup_compensation(controller, plant, saturation_limits)
    except ValueError as e:
        report = controller.history.get_formatted_report()
        assert "Contexto...: Planta: G(s) = (1) / (s**2 + s + 1)" in report
        assert "Falha na Decomposição" in report


def test_history_warning_second_order():
    """
    Testa se um aviso é adicionado ao histórico quando a aproximação de segunda ordem não é válida.
    """
    s = sp.Symbol('s')
    # Sistema com um terceiro polo próximo aos polos dominantes
    closed_loop = SymbolicTransferFunction(1, (s**2 + 2*s + 2) * (s + 1))
    _, closed_loop = calculate_performance_metrics(closed_loop)
    report = closed_loop.history.get_formatted_report()
    assert "Aproximação de 2ª ordem não confiável" in report


def test_history_warning_lead_compensator():
    """
    Testa se um aviso é adicionado ao histórico para um grande avanço de fase.
    """
    s = sp.Symbol('s')
    plant = SymbolicTransferFunction(1, s * (s + 1))
    result = design_lead_compensator(plant, 70, 5)
    report = result.controller.history.get_formatted_report()
    assert "O avanço de fase de 70° é muito grande" in report
