"""
ControlLab Web Module - Aplicação Web Educacional Interativa

Este módulo implementa uma aplicação web educacional do tipo "Symbolab para Engenharia de Controle"
com foco em descarregamento cognitivo para usuários com TDAH.

Arquitetura Fundamental: Dualidade Simbólico-Numérica
- Todas as operações começam simbolicamente (backend ControlLab)
- Explicações step-by-step com fundamentação teórica
- Conversão para domínio numérico apenas para visualização
- Transparência total do processo matemático

Filosofia "Anti-Caixa-Preta"
"""

__version__ = "1.0.0"
__author__ = "ControlLab Team"

# Exportações principais (import com try/except para evitar erros de dependências)
try:
    from .analysis_maestro import AnalysisMaestro
    MAESTRO_AVAILABLE = True
except ImportError:
    MAESTRO_AVAILABLE = False

try:
    from .ai_tutor import SocraticTutor
    TUTOR_AVAILABLE = True
except ImportError:
    TUTOR_AVAILABLE = False

try:
    from .websocket_server import ControlLabWebSocketServer
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False

try:
    from .smart_plots import SmartPlotter, VisualizationManager
    PLOTS_AVAILABLE = True
except ImportError:
    PLOTS_AVAILABLE = False

__all__ = [
    'AnalysisMaestro' if MAESTRO_AVAILABLE else None,
    'SocraticTutor' if TUTOR_AVAILABLE else None,
    'ControlLabWebSocketServer' if WEBSOCKET_AVAILABLE else None,
    'SmartPlotter' if PLOTS_AVAILABLE else None,
    'VisualizationManager' if PLOTS_AVAILABLE else None
]

# Remover None values
__all__ = [item for item in __all__ if item is not None]

# Status de disponibilidade
WEB_MODULE_STATUS = {
    "maestro_available": MAESTRO_AVAILABLE,
    "tutor_available": TUTOR_AVAILABLE,
    "websocket_available": WEBSOCKET_AVAILABLE,
    "plots_available": PLOTS_AVAILABLE
}
