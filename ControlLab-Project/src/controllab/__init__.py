"""
ControlLab - Laboratório de Controle
===================================

Sistema completo de análise e projeto de sistemas de controle.
"""

__version__ = "1.0.0"
__author__ = "ControlLab Development Team"

# Importações principais do módulo
try:
    # Em src/controllab/__init__.py
# Seja explícito sobre o que quer exportar do seu pacote.
# Se a intenção é que os utilizadores possam fazer `import controllab` e aceder
# às classes principais, importe-as aqui explicitamente.

    from .core.symbolic_tf import SymbolicTransferFunction
    from .core.symbolic_ss import SymbolicStateSpace
    from .analysis.stability_analysis import analyze_stability

# Adicione outras importações de alto nível que queira expor aqui.
# Não use "import *".
except ImportError:
    pass  # Módulos podem não estar disponíveis durante instalação
