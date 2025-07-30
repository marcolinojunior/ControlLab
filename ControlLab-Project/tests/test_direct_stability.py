import unittest
import sympy as sp
import sys
from pathlib import Path

# Adicionar src ao path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from controllab.analysis.root_locus import RootLocusAnalyzer
from controllab.core.symbolic_tf import SymbolicTransferFunction

class TestDirectStability(unittest.TestCase):
    def setUp(self):
        self.analyzer = RootLocusAnalyzer()
        self.s = sp.Symbol('s')

    def test_unstable_system(self):
        G = SymbolicTransferFunction(1, self.s - 1)
        is_stable = self.analyzer._is_stable_from_root_locus(G)
        self.assertFalse(is_stable)

    def test_stable_system(self):
        G = SymbolicTransferFunction(1, self.s + 1)
        is_stable = self.analyzer._is_stable_from_root_locus(G)
        self.assertTrue(is_stable)

if __name__ == '__main__':
    unittest.main()
