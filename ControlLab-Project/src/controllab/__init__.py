"""
ControlLab - Laboratório de Controle
===================================

Sistema completo de análise e projeto de sistemas de controle.
"""

__version__ = "1.0.0"
__author__ = "ControlLab Development Team"

# Importações principais do pacote ControlLab
try:
    from .analysis import *
    from .core import *
    from .design import *
    from .modeling import *
    from .numerical import *
    from .web import *
except ImportError:
    pass  # Módulos podem não estar disponíveis durante instalação
