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

        response = format_routh_hurwitz_response(result, analyzer.history)

        self.assertEqual(response['title'], "Análise de Estabilidade de Routh-Hurwitz")
        self.assertEqual(response['polynomial'], "s**3 + 2*s**2 + 3*s + 4")
        self.assertIsInstance(response['steps'], list)
        self.assertIsInstance(response['special_cases'], list)
        self.assertIsInstance(response['conclusion'], dict)
        self.assertIn('is_stable', response['conclusion'])
        self.assertIn('unstable_poles_count', response['conclusion'])
        self.assertIn('summary', response['conclusion'])

if __name__ == '__main__':
    unittest.main()
