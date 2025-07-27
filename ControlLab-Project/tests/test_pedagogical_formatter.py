import unittest
import sympy as sp
from controllab.analysis.routh_hurwitz import RouthHurwitzAnalyzer, StabilityResult, RouthAnalysisHistory
from controllab.analysis.pedagogical_formatter import format_routh_hurwitz_response

class TestPedagogicalFormatter(unittest.TestCase):

    def test_format_routh_hurwitz_response(self):
        s = sp.Symbol('s')
        polynomial = s**3 + 2*s**2 + 3*s + 4
        analyzer = RouthHurwitzAnalyzer()
        routh_obj = analyzer.build_routh_array(polynomial)
        result = analyzer.analyze_stability(routh_obj)

        response = format_routh_hurwitz_response(result, analyzer.history, polynomial)

        self.assertIsInstance(response['steps'], list)
        self.assertIsInstance(response['conclusion'], bool)

if __name__ == '__main__':
    unittest.main()
