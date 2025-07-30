#!/usr/bin/env python3
"""
🧮 Enhanced LaTeX Response Enhancer - ControlLab
Versão aprimorada que usa ponte simbólica para serialização
"""

import sys
import os

# Adicionar path para websocket_symbolic_bridge
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from websocket_symbolic_bridge import WebSocketSymbolicBridge

try:
    import sympy as sp
    from ..core.visualization import LaTeXGenerator
    LATEX_AVAILABLE = True
except ImportError:
    LATEX_AVAILABLE = False

class EnhancedLaTeXResponseEnhancer:
    """
    Enhanced LaTeX response enhancer usando ponte simbólica
    """
    
    def __init__(self):
        if LATEX_AVAILABLE:
            self.latex_generator = LaTeXGenerator()
        self.symbolic_bridge = WebSocketSymbolicBridge()
    
    def enhance_response_with_latex(self, response):
        """
        Aplica enhancement LaTeX usando ponte simbólica para serialização segura
        """
        if not LATEX_AVAILABLE:
            return response
        
        enhanced_response = response.copy()
        
        # Processar campos que podem conter SymPy
        latex_fields = {}
        
        for key, value in response.items():
            if self._contains_sympy_expression(value):
                # Gerar LaTeX
                latex_repr = self._generate_latex_safely(value)
                if latex_repr:
                    latex_fields[f"{key}_latex"] = latex_repr
                
                # Usar ponte simbólica para serialização
                enhanced_response[key] = self.symbolic_bridge.prepare_for_websocket({key: value})[key]
        
        # Adicionar campos LaTeX
        enhanced_response.update(latex_fields)
        enhanced_response['has_latex'] = len(latex_fields) > 0
        enhanced_response['show_latex'] = True
        
        return enhanced_response
    
    def _contains_sympy_expression(self, obj):
        """Verifica se contém expressões SymPy"""
        if isinstance(obj, sp.Basic):
            return True
        elif isinstance(obj, (list, tuple)):
            return any(self._contains_sympy_expression(item) for item in obj)
        elif isinstance(obj, dict):
            return any(self._contains_sympy_expression(v) for v in obj.values())
        return False
    
    def _generate_latex_safely(self, obj):
        """Gera LaTeX de forma segura"""
        try:
            if isinstance(obj, sp.Basic):
                return self.latex_generator.transfer_function_to_latex_from_expr(obj)
            else:
                return str(obj)
        except Exception:
            return str(obj)
