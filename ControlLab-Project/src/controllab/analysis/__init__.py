"""
Módulo de Análise de Sistemas de Controle
=========================================

Este módulo fornece ferramentas completas de análise para sistemas de controle,
incluindo análise de resposta temporal, frequencial e estabilidade.

Submódulos:
    temporal: Análise de resposta temporal (Módulo 4) - A ser implementado
    stability_analysis: Interface unificada de análise de estabilidade (Módulo 5)
    routh_hurwitz: Análise de estabilidade usando critério de Routh-Hurwitz
    root_locus: Análise do lugar geométrico das raízes
    frequency_response: Análise de resposta em frequência e margens
    stability_utils: Utilitários e validação cruzada
    
Exemplo de Uso:
    ```python
    # Análise de estabilidade completa
    from controllab.analysis import analyze_stability
    stability_report = analyze_stability(system, show_steps=True)
    print(stability_report.get_full_report())
    
    # Verificação rápida de estabilidade
    from controllab.analysis import quick_stability_check
    is_stable = quick_stability_check(system)
    ```
"""

__version__ = "1.0.0"

# Importações do módulo de estabilidade (Módulo 5)
try:
    from .stability_analysis import (
        # Classes principais
        StabilityAnalysisEngine,
        ComprehensiveStabilityReport,
        
        # Funções de conveniência
        analyze_stability,
        quick_stability_check,
        compare_systems_stability
    )
    
    from .routh_hurwitz import (
        RouthHurwitzAnalyzer,
        StabilityResult,
        RouthAnalysisHistory
    )
    
    from .root_locus import (
        RootLocusAnalyzer,
        LocusFeatures,
        LocusHistory
    )
    
    from .frequency_response import (
        FrequencyAnalyzer,
        StabilityMargins,
        FrequencyAnalysisHistory
    )
    
    from .stability_utils import (
        StabilityValidator,
        ParametricAnalyzer,
        ValidationHistory,
        validate_stability_methods,
        format_stability_report
    )
    
    STABILITY_ANALYSIS_AVAILABLE = True
    
except ImportError as e:
    import warnings
    warnings.warn(f"Módulo de estabilidade não disponível: {e}")
    STABILITY_ANALYSIS_AVAILABLE = False
    
    # Definir placeholders para evitar erros
    StabilityAnalysisEngine = None
    analyze_stability = None
    quick_stability_check = None

# Módulo temporal (Módulo 4) - A ser implementado no futuro
TEMPORAL_ANALYSIS_AVAILABLE = False

# Placeholder para funções que serão implementadas no módulo temporal
def analyze_step_response(*args, **kwargs):
    """Placeholder - será implementado no Módulo 4"""
    raise NotImplementedError("Módulo temporal ainda não implementado")

def analyze_impulse_response(*args, **kwargs):
    """Placeholder - será implementado no Módulo 4"""
    raise NotImplementedError("Módulo temporal ainda não implementado")

def analyze_transient_response(*args, **kwargs):
    """Placeholder - será implementado no Módulo 4"""
    raise NotImplementedError("Módulo temporal ainda não implementado")

def compare_responses(*args, **kwargs):
    """Placeholder - será implementado no Módulo 4"""
    raise NotImplementedError("Módulo temporal ainda não implementado")

# Classes placeholder que serão implementadas no módulo temporal
class ResponseCharacteristics:
    """Placeholder - será implementado no Módulo 4"""
    def __init__(self):
        raise NotImplementedError("Módulo temporal ainda não implementado")

class TransientParameters:
    """Placeholder - será implementado no Módulo 4"""
    def __init__(self):
        raise NotImplementedError("Módulo temporal ainda não implementado")

class ComparisonResult:
    """Placeholder - será implementado no Módulo 4"""
    def __init__(self):
        raise NotImplementedError("Módulo temporal ainda não implementado")

# Lista de análises disponíveis
AVAILABLE_ANALYSES = []
if TEMPORAL_ANALYSIS_AVAILABLE:
    AVAILABLE_ANALYSES.append('temporal')
if STABILITY_ANALYSIS_AVAILABLE:
    AVAILABLE_ANALYSES.append('stability')

# Função de diagnóstico
def check_analysis_capabilities():
    """
    Verifica quais capacidades de análise estão disponíveis
    
    Returns:
        Dict com status de cada módulo
    """
    capabilities = {
        'temporal_analysis': TEMPORAL_ANALYSIS_AVAILABLE,
        'stability_analysis': STABILITY_ANALYSIS_AVAILABLE,
        'available_modules': AVAILABLE_ANALYSES
    }
    
    return capabilities

__all__ = [
    # Classes principais do módulo de estabilidade (disponíveis)
    'StabilityAnalysisEngine',
    'ComprehensiveStabilityReport',
    'RouthHurwitzAnalyzer',
    'RootLocusAnalyzer',
    'FrequencyAnalyzer',
    'StabilityValidator',
    'ParametricAnalyzer',
    
    # Funções de conveniência estabilidade (disponíveis)
    'analyze_stability',
    'quick_stability_check',
    'compare_systems_stability',
    'validate_stability_methods',
    'format_stability_report',
    
    # Classes de dados estabilidade (disponíveis)
    'StabilityResult',
    'LocusFeatures',
    'StabilityMargins',
    'ValidationHistory',
    'RouthAnalysisHistory',
    'LocusHistory',
    'FrequencyAnalysisHistory',
    
    # Placeholders do módulo temporal (a implementar)
    'analyze_step_response',
    'analyze_impulse_response', 
    'analyze_transient_response',
    'compare_responses',
    'ResponseCharacteristics',
    'TransientParameters',
    'ComparisonResult',
    
    # Funções utilitárias
    'check_analysis_capabilities',
    
    # Constantes de disponibilidade
    'TEMPORAL_ANALYSIS_AVAILABLE',
    'STABILITY_ANALYSIS_AVAILABLE',
    'AVAILABLE_ANALYSES',
]