"""
ControlLab - Módulo de Projeto de Controladores e Observadores
==============================================================

Este módulo implementa métodos de projeto de controladores clássicos e modernos,
com foco na derivação simbólica de ganhos e estruturas de compensação.

Características:
- Explicações step-by-step similares ao Symbolab
- Projeto de compensadores clássicos (PID, Lead, Lag)
- Alocação de polos e projeto de observadores
- Controle ótimo (LQR/LQG)
- Análise de robustez e desempenho
- Validação automática de especificações

Classes Principais:
- CompensatorDesigner: Projeto sistemático de compensadores
- StateSpaceDesigner: Projeto em espaço de estados
- PerformanceAnalyzer: Análise de desempenho e robustez
"""

# Importações condicionais para evitar erros de importação circular
try:
    from .design_utils import (
        ControllerResult,
        DesignSpecifications,
        validate_closed_loop_stability,
        calculate_performance_metrics
    )
    
    from .compensators import (
        PID, Lead, Lag, LeadLag,
        CompensatorDesigner,
        design_lead_compensator,
        design_lag_compensator,
        design_pid_tuning,
        design_by_root_locus
    )

    from .pole_placement import (
        check_controllability,
        acker,
        place_poles_robust,
        StateSpaceDesigner
    )

    from .observer import (
        check_observability,
        acker_observer,
        design_luenberger_observer,
        ObserverDesigner,
    )
    
    # Importar novos módulos implementados
    from .specifications import (
        PerformanceSpec,
        verify_specifications,
        pole_placement_from_specs
    )
    
    from .visualization import (
        show_compensator_effect,
        plot_design_tradeoffs,
        animate_pole_placement,
        visualize_controller_design_process
    )
    
    from .comparison import (
        ComparisonResult,
        compare_controller_designs,
        pareto_analysis,
        sensitivity_comparison
    )
    
    from .antiwindup import (
        SaturationLimits,
        AntiWindupResult,
        design_antiwindup_compensation,
        auto_tune_antiwindup_parameters
    )

    from .lqr import (
        solve_are_symbolic,
        lqr_design,
        analyze_lqr_sensitivity,
        LQRDesigner
    )

    from .performance import (
        analyze_transient_response,
        calculate_performance_indices,
        sensitivity_analysis,
        robustness_analysis
    )
    
except ImportError as e:
    # Importações básicas para desenvolvimento
    print(f"⚠️ Alguns módulos não estão disponíveis: {e}")
    
    # Definir classes vazias para não quebrar importações
    class ControllerResult:
        def __init__(self, controller):
            self.controller = controller
    
    class DesignSpecifications:
        pass

# Versão do módulo
__version__ = "1.0.0"

# Lista de exports
__all__ = [
    # Compensadores clássicos
    'PID', 'Lead', 'Lag', 'LeadLag',
    'CompensatorDesigner',
    'design_lead_compensator',
    'design_lag_compensator', 
    'design_pid_tuning',
    'design_by_root_locus',
    
    # Alocação de polos
    'check_controllability',
    'acker',
    'place_poles_robust',
    'StateSpaceDesigner',
    
    # Observadores
    'check_observability',
    'acker_observer',
    'design_luenberger_observer',
    'ObserverDesigner',
    
    # Controle ótimo
    'solve_are_symbolic',
    'lqr_design',
    'analyze_lqr_sensitivity',
    'LQRDesigner',
    
    # Utilitários
    'ControllerResult',
    'DesignSpecifications',
    'validate_closed_loop_stability',
    'calculate_performance_metrics',
    
    # Análise de desempenho
    'analyze_transient_response',
    'calculate_performance_indices',
    'sensitivity_analysis',
    'robustness_analysis'
]
