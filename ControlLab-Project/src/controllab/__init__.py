"""
ControlLab: Uma biblioteca de engenharia de controlo simbólica e pedagógica.
Este ficheiro __init__.py torna o 'controllab' um pacote importável.
"""
from .core.symbolic_tf import SymbolicTransferFunction
from .core.symbolic_ss import SymbolicStateSpace
from .analysis.stability_analysis import analyze_stability
from .design.compensators import PID, Lead, Lag
