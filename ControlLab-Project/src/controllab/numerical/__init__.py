#!/usr/bin/env python3
"""
Interface Numérica - ControlLab
Módulo de integração entre representações simbólicas e computações numéricas
"""

# Importações condicionais para evitar dependências obrigatórias
try:
    import control
    CONTROL_AVAILABLE = True
except ImportError:
    CONTROL_AVAILABLE = False
    control = None

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import scipy
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    scipy = None

from .interface import NumericalInterface
from .compiler import ExpressionCompiler, CompiledFunction
from .validation import NumericalValidator
from .factory import NumericalSystemFactory

# Verificar dependências disponíveis
def get_available_backends():
    """Retorna backends numéricos disponíveis"""
    backends = {}
    if NUMPY_AVAILABLE:
        backends['numpy'] = np
    if CONTROL_AVAILABLE:
        backends['control'] = control
    if SCIPY_AVAILABLE:
        backends['scipy'] = scipy
    return backends

def check_numerical_dependencies():
    """Verifica se dependências numéricas estão disponíveis"""
    status = {
        'numpy': NUMPY_AVAILABLE,
        'control': CONTROL_AVAILABLE,
        'scipy': SCIPY_AVAILABLE,
        'all_available': NUMPY_AVAILABLE and CONTROL_AVAILABLE
    }
    return status

# Classes principais disponíveis
__all__ = [
    'NumericalInterface',
    'ExpressionCompiler', 
    'CompiledFunction',
    'NumericalValidator',
    'NumericalSystemFactory',
    'get_available_backends',
    'check_numerical_dependencies',
    'NUMPY_AVAILABLE',
    'CONTROL_AVAILABLE',
    'SCIPY_AVAILABLE'
]
