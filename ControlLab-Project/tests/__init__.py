"""
Arquivo de Inicialização dos Testes
====================================

Configuração inicial para os testes do módulo de análise de estabilidade.
"""

import sys
import os

# Configurar path para importação dos módulos
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)
