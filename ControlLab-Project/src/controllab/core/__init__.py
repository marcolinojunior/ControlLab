#!/usr/bin/env python3
"""
Núcleo Simbólico - ControlLab
Módulo principal para computação simbólica em sistemas de controle
"""

from .symbolic_tf import SymbolicTransferFunction
from .symbolic_ss import SymbolicStateSpace
from .history import OperationHistory, OperationStep
from .symbolic_utils import (
    create_laplace_variable,
    create_z_variable,
    poly_from_roots,
    validate_proper_tf,
    cancel_common_factors,
    extract_poles_zeros,
    create_proper_tf,
    expand_partial_fractions,
    symbolic_stability_analysis,
    convert_to_latex_formatted
)

# Importações condicionais para módulos avançados (requerem dependências externas)
def get_advanced_modules():
    """
    Importa módulos avançados somente quando necessário
    Evita erros de importação quando dependências não estão instaladas
    """
    advanced = {}
    
    try:
        from .stability_analysis import (
            RouthHurwitzAnalyzer,
            NyquistAnalyzer,
            BodeAnalyzer,
            RootLocusAnalyzer
        )
        advanced['stability'] = {
            'RouthHurwitzAnalyzer': RouthHurwitzAnalyzer,
            'NyquistAnalyzer': NyquistAnalyzer,
            'BodeAnalyzer': BodeAnalyzer,
            'RootLocusAnalyzer': RootLocusAnalyzer
        }
    except ImportError:
        advanced['stability'] = None
    
    try:
        from .controller_design import (
            PIDController,
            LeadLagCompensator,
            StateSpaceController,
            ObserverDesign
        )
        advanced['controllers'] = {
            'PIDController': PIDController,
            'LeadLagCompensator': LeadLagCompensator,
            'StateSpaceController': StateSpaceController,
            'ObserverDesign': ObserverDesign
        }
    except ImportError:
        advanced['controllers'] = None
    
    try:
        from .transforms import (
            LaplaceTransform,
            ZTransform,
            FourierTransform
        )
        advanced['transforms'] = {
            'LaplaceTransform': LaplaceTransform,
            'ZTransform': ZTransform,
            'FourierTransform': FourierTransform
        }
    except ImportError:
        advanced['transforms'] = None
    
    try:
        from .visualization import (
            SymbolicPlotter,
            LaTeXGenerator,
            BlockDiagramGenerator
        )
        advanced['visualization'] = {
            'SymbolicPlotter': SymbolicPlotter,
            'LaTeXGenerator': LaTeXGenerator,
            'BlockDiagramGenerator': BlockDiagramGenerator
        }
    except ImportError:
        advanced['visualization'] = None
    
    return advanced

__all__ = [
    # Classes principais (sempre disponíveis)
    'SymbolicTransferFunction',
    'SymbolicStateSpace',
    
    # Sistema de histórico
    'OperationHistory',
    'OperationStep',
    
    # Utilitários simbólicos
    'create_laplace_variable',
    'create_z_variable',
    'poly_from_roots',
    'validate_proper_tf',
    'cancel_common_factors',
    'extract_poles_zeros',
    'create_proper_tf',
    'expand_partial_fractions',
    'symbolic_stability_analysis',
    'convert_to_latex_formatted',
    
    # Função para carregar módulos avançados
    'get_advanced_modules'
]