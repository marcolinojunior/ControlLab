"""
Módulo de Modelagem de Sistemas - ControlLab
==========================================

Este módulo implementa ferramentas para modelagem de sistemas dinâmicos,
incluindo transformadas de Laplace, conversões entre representações e
modelagem de sistemas físicos.

Submódulos:
    laplace_transform: Transformadas de Laplace com passos pedagógicos
    partial_fractions: Expansão em frações parciais detalhada
    conversions: Conversões entre TF e SS, conexões de sistemas
    canonical_forms: Formas canônicas (controlável, observável, modal, Jordan)
    physical_systems: Modelagem de sistemas físicos (mecânico, elétrico, térmico)
"""

# Importações principais
try:
    # Transformadas de Laplace
    from .laplace_transform import (
        LaplaceTransformer,
        from_ode,
        apply_laplace_transform,
        inverse_laplace_transform,
        unit_step_laplace,
        unit_impulse_laplace,
        exponential_laplace,
        sinusoidal_laplace,
        cosinusoidal_laplace
    )
    
    # Frações parciais
    from .partial_fractions import (
        PartialFractionExpander,
        explain_partial_fractions,
        find_residues_symbolic,
        handle_repeated_poles,
        handle_complex_poles,
        convert_to_quadratic_form
    )
    
    # Conversões
    from .conversions import (
        tf_to_ss,
        ss_to_tf,
        parallel_to_series,
        series_to_parallel,
        feedback_connection,
        analyze_system_connection,
        validate_conversion
    )
    
    # Formas canônicas
    from .canonical_forms import (
        controllable_canonical,
        observable_canonical,
        modal_canonical,
        jordan_canonical,
        compare_canonical_forms,
        get_canonical_form_documentation
    )
    
    # Sistemas físicos
    from .physical_systems import (
        PhysicalSystemBase,
        MechanicalSystem,
        ElectricalSystem,
        ThermalSystem,
        create_mass_spring_damper,
        create_rlc_circuit,
        create_thermal_system
    )
    
    # Visualização de passos
    from .step_visualization import (
        show_laplace_steps,
        show_partial_fraction_steps,
        export_to_html,
        export_to_pdf,
        create_jupyter_visualization,
        StepVisualizationHistory
    )
    
    # Validações pedagógicas
    from .validation import (
        check_pole_zero_cancellation,
        check_causality,
        check_bibo_stability,
        check_minimum_phase,
        validate_system_properties,
        ValidationResult
    )
    
    # Casos especiais
    from .special_cases import (
        create_time_delay_system,
        analyze_rhp_zeros,
        simplify_pole_zero_cancellation,
        handle_initial_conditions,
        decompose_improper_system,
        SpecialCaseHandler
    )
    
    # Integração com outros módulos
    from .integration import (
        EducationalPipeline,
        create_educational_pipeline,
        integrate_with_core,
        create_educational_workflow
    )
    
    # Módulo 7 - Sistemas Discretos (Transformada Z)
    from .z_transform import (
        ZTransformer,
        ZTransformResult,
        apply_z_transform,
        inverse_z_transform,
        from_difference_equation
    )
    
    # Discretização de sistemas
    from .discretization import (
        DiscretizationMethods,
        DiscretizationResult,
        compare_discretization_methods
    )
    
    # Estabilidade de sistemas discretos
    from .discrete_stability import (
        DiscreteStabilityAnalyzer,
        StabilityResult,
        analyze_discrete_stability,
        compare_stability_methods
    )
    
    # Lugar das raízes discreto
    from .discrete_root_locus import (
        DiscreteRootLocus,
        DiscreteRootLocusResult,
        plot_discrete_root_locus,
        analyze_discrete_performance
    )
    
    # Resposta em frequência discreta
    from .discrete_frequency_response import (
        DiscreteFrequencyAnalyzer,
        DiscreteFrequencyResult,
        analyze_discrete_frequency_response,
        compare_continuous_discrete_frequency
    )
    
except ImportError as e:
    # Fallback para desenvolvimento sem dependências
    import warnings
    warnings.warn(f"Algumas funcionalidades do módulo modeling não estão disponíveis: {e}")
    
    # Definir classes/funções básicas como fallback
    class LaplaceTransformer:
        def __init__(self):
            pass
    
    def from_ode(*args, **kwargs):
        raise NotImplementedError("SymPy necessário para transformadas de Laplace")
    
    def explain_partial_fractions(*args, **kwargs):
        raise NotImplementedError("SymPy necessário para frações parciais")
    
    def tf_to_ss(*args, **kwargs):
        raise NotImplementedError("SymPy necessário para conversões")
    
    def controllable_canonical(*args, **kwargs):
        raise NotImplementedError("SymPy necessário para formas canônicas")
    
    class MechanicalSystem:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("SymPy necessário para sistemas físicos")


# Funcionalidades principais disponíveis
__all__ = [
    # Transformadas de Laplace
    'LaplaceTransformer',
    'from_ode',
    'apply_laplace_transform', 
    'inverse_laplace_transform',
    'unit_step_laplace',
    'unit_impulse_laplace',
    'exponential_laplace',
    'sinusoidal_laplace',
    'cosinusoidal_laplace',
    
    # Frações parciais
    'PartialFractionExpander',
    'explain_partial_fractions',
    'find_residues_symbolic',
    'handle_repeated_poles',
    'handle_complex_poles',
    'convert_to_quadratic_form',
    
    # Conversões
    'tf_to_ss',
    'ss_to_tf',
    'parallel_to_series',
    'series_to_parallel',
    'feedback_connection',
    'analyze_system_connection',
    'validate_conversion',
    
    # Formas canônicas
    'controllable_canonical',
    'observable_canonical',
    'modal_canonical',
    'jordan_canonical',
    'compare_canonical_forms',
    'get_canonical_form_documentation',
    
    # Sistemas físicos
    'PhysicalSystemBase',
    'MechanicalSystem',
    'ElectricalSystem',
    'ThermalSystem',
    'create_mass_spring_damper',
    'create_rlc_circuit',
    'create_thermal_system',
    
    # Visualização de passos
    'show_laplace_steps',
    'show_partial_fraction_steps',
    'export_to_html',
    'export_to_pdf',
    'create_jupyter_visualization',
    'StepVisualizationHistory',
    
    # Validações pedagógicas
    'check_pole_zero_cancellation',
    'check_causality',
    'check_bibo_stability',
    'check_minimum_phase',
    'validate_system_properties',
    'ValidationResult',
    
    # Casos especiais
    'create_time_delay_system',
    'analyze_rhp_zeros',
    'simplify_pole_zero_cancellation',
    'handle_initial_conditions',
    'decompose_improper_system',
    'SpecialCaseHandler',
    
    # Módulo 7 - Sistemas Discretos
    'ZTransformer',
    'ZTransformResult',
    'apply_z_transform',
    'inverse_z_transform',
    'from_difference_equation',
    
    # Discretização
    'DiscretizationMethods',
    'DiscretizationResult',
    'compare_discretization_methods',
    
    # Estabilidade discreta
    'DiscreteStabilityAnalyzer',
    'StabilityResult',
    'analyze_discrete_stability',
    'compare_stability_methods',
    
    # Lugar das raízes discreto
    'DiscreteRootLocus',
    'DiscreteRootLocusResult',
    'plot_discrete_root_locus',
    'analyze_discrete_performance',
    
    # Resposta em frequência discreta
    'DiscreteFrequencyAnalyzer',
    'DiscreteFrequencyResult',
    'analyze_discrete_frequency_response',
    'compare_continuous_discrete_frequency'
]


# Documentação de uso
MODELING_USAGE_EXAMPLES = """
📚 EXEMPLOS DE USO - MÓDULO MODELING
===================================

🔄 1. TRANSFORMADAS DE LAPLACE:
```python
from controllab.modeling import apply_laplace_transform
import sympy as sp

t, s = sp.symbols('t s')
f = sp.exp(-2*t) * sp.sin(3*t)
F = apply_laplace_transform(f, t, s, show_steps=True)
```

📐 2. FRAÇÕES PARCIAIS:
```python
from controllab.modeling import explain_partial_fractions

s = sp.Symbol('s')
tf = 1 / (s*(s+1)*(s+2))
expansion = explain_partial_fractions(tf)
print(expansion.show_explanation())
```

🔧 3. CONVERSÕES TF ↔ SS:
```python
from controllab.modeling import tf_to_ss, ss_to_tf

# TF para SS forma controlável
tf_expr = 1 / (s**2 + 3*s + 2)
ss_system = tf_to_ss(tf_expr, form='controllable')

# SS para TF
tf_recovered = ss_to_tf(ss_system)
```

🏗️ 4. FORMAS CANÔNICAS:
```python
from controllab.modeling import controllable_canonical, modal_canonical

# Forma controlável
controllable = controllable_canonical(tf_expr)

# Forma modal (se possível)
modal = modal_canonical(controllable)
```

⚙️ 5. SISTEMAS FÍSICOS:
```python
from controllab.modeling import MechanicalSystem

# Sistema massa-mola-amortecedor
system = MechanicalSystem(mass=1, damping=0.5, stiffness=2)
system.derive_equations(show_steps=True)
system.apply_laplace_modeling(show_steps=True)

print(system.get_system_summary())
print(system.history.get_formatted_derivation())
```

🔌 6. CIRCUITO ELÉTRICO:
```python
from controllab.modeling import ElectricalSystem

# Circuito RLC série
circuit = ElectricalSystem(resistance=1, inductance=0.1, capacitance=0.01)
circuit.derive_equations(show_steps=True)
circuit.apply_laplace_modeling(show_steps=True)
```

🌡️ 7. SISTEMA TÉRMICO:
```python
from controllab.modeling import ThermalSystem

# Sistema térmico de 1ª ordem
thermal = ThermalSystem(thermal_resistance=10, thermal_capacitance=2)
thermal.derive_equations(show_steps=True)
analysis = thermal.analyze_thermal_response()
```

🔗 8. CONEXÕES DE SISTEMAS:
```python
from controllab.modeling import feedback_connection, parallel_to_series

# Realimentação
G = 1/(s+1)
H = 0.5
T = feedback_connection(G, H, sign=-1)  # Realimentação negativa

# Paralelo
tf_list = [1/(s+1), 2/(s+2)]
parallel_tf = parallel_to_series(tf_list)
```
"""
