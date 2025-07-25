#!/usr/bin/env python3
"""
Teste específico para debug do erro paramétrico
"""

import sys
import traceback
from pathlib import Path
import sympy as sp

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from controllab.analysis.stability_utils import (
    ParametricAnalyzer,
    stability_region_2d,
    root_locus_3d
)
from controllab.core import SymbolicTransferFunction

def debug_parametric_error():
    """Debug específico do erro paramétrico"""
    print("🔧 DEBUGANDO ERRO PARAMÉTRICO...")
    
    # Sistema paramétrico: G(s) = K1/(s^2 + K2*s + 1)
    s, K1, K2 = sp.symbols('s K1 K2')
    num = K1
    den = s**2 + K2*s + 1
    G = SymbolicTransferFunction(num, den)
    
    print(f"Sistema: {G}")
    print(f"Parâmetros: K1={K1}, K2={K2}")
    
    try:
        print("\n1. Testando stability_region_2d standalone...")
        result1 = stability_region_2d(
            G, K1, K2, 
            param1_range=(0.1, 10), 
            param2_range=(0.1, 5)
        )
        print(f"✅ stability_region_2d resultado: {type(result1)}")
        print(f"   Chaves: {list(result1.keys()) if isinstance(result1, dict) else 'não é dict'}")
        
    except Exception as e:
        print(f"💥 ERRO em stability_region_2d: {e}")
        print(traceback.format_exc())
    
    try:
        print("\n2. Testando root_locus_3d standalone...")
        result2 = root_locus_3d(G, K1, K2, k_range=[0.1, 1.0, 5.0])
        print(f"✅ root_locus_3d resultado: {type(result2)}")
        print(f"   Chaves: {list(result2.keys()) if isinstance(result2, dict) else 'não é dict'}")
        
    except Exception as e:
        print(f"💥 ERRO em root_locus_3d: {e}")
        print(traceback.format_exc())

if __name__ == "__main__":
    debug_parametric_error()
