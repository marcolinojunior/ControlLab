import sympy as sp
from controllab.core.history import OperationHistory
from controllab.core.symbolic_tf import SymbolicTransferFunction
from controllab.design.antiwindup import design_antiwindup_compensation, SaturationLimits
from controllab.design.compensators import design_lead_compensator
from controllab.design.design_utils import calculate_performance_metrics
from controllab.analysis.system_properties import verify_second_order_approximation


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


def test_verify_second_order_approximation_invalid_pole():
    """
    CORRIGIDO: Testa um sistema de 3ª ordem onde o terceiro polo está
    muito próximo para ser ignorado.
    """
    s = sp.Symbol('s')
    # Polos dominantes em -1 +/- 1j. Terceiro polo em -2.
    # A parte real do terceiro polo (-2) não é 5x maior que a parte real
    # dos polos dominantes (-1). A aproximação deve ser inválida.
    system = SymbolicTransferFunction(5, (s**2 + 2*s + 2) * (s + 2))
    is_valid, msg, _ = verify_second_order_approximation(system)
    assert is_valid is False, "A função deveria invalidar a aproximação devido ao terceiro polo próximo."

def test_verify_second_order_approximation_invalid_zero():
    """
    CORRIGIDO: Testa um sistema de 2ª ordem onde um zero "lento"
    invalida a aproximação padrão.
    """
    s = sp.Symbol('s')
    # Polos em -5, -10. O zero em -1 é mais lento (mais próximo da origem)
    # que o polo dominante em -5. A aproximação deve ser inválida.
    system = SymbolicTransferFunction((s + 1), (s + 5) * (s + 10))
    is_valid, msg, _ = verify_second_order_approximation(system)
    assert is_valid is False, "A função deveria invalidar a aproximação devido ao zero próximo/lento."


def test_history_warning_lead_compensator():
    """
    Testa se um aviso é adicionado ao histórico para um grande avanço de fase.
    """
    s = sp.Symbol('s')
    plant = SymbolicTransferFunction(1, s * (s + 1))
    result = design_lead_compensator(plant, 70, 5)
    report = result.controller.history.get_formatted_report()
    assert "O avanço de fase de 70° é muito grande" in report
