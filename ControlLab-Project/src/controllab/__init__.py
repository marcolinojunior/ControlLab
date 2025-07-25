"""
ControlLab - Laboratório de Controle
===================================

Sistema completo de análise e projeto de sistemas de controle.
"""

__version__ = "1.0.0"
__author__ = "ControlLab Development Team"

# Importações principais do módulo
try:
    from controllab.analysis import *
except ImportError:
    pass  # Módulos podem não estar disponíveis durante instalação
