import unittest
import sympy as sp
from controllab.analysis.routh_hurwitz import RouthHurwitzAnalyzer, StabilityResult, RouthAnalysisHistory
from controllab.analysis.pedagogical_formatter import format_routh_hurwitz_response

class TestPedagogicalFormatter(unittest.TestCase):

    def test_format_routh_hurwitz_response(self):
        s = sp.Symbol('s')
        polynomial = s**3 + 2*s**2 + 3*s + 4
        analyzer = RouthHurwitzAnalyzer()
        stability_result, history, polynomial = analyzer.analyze(polynomial)

        response = format_routh_hurwitz_response(stability_result, history, polynomial)

        self.assertIsInstance(response, dict)
        self.assertIn('conclusion', response)
        self.assertIn('summary', response)
        self.assertIn('steps', response)

        self.assertIsInstance(response['steps'], list)
        self.assertGreater(len(response['steps']), 0)

        # Check the first step
        first_step = response['steps'][0]
        self.assertEqual(first_step['title'], "1. Polinômio Característico")
        self.assertIn('polynomial_latex', first_step['data'])

        # Check the second step
        second_step = response['steps'][1]
        self.assertEqual(second_step['title'], "3. Construção da Tabela de Routh")
        self.assertIn('routh_table', second_step['data'])

        # Check the conclusion
        self.assertEqual(response['conclusion'], stability_result.is_stable)
        self.assertEqual(response['summary'], f"O sistema é {'estável' if stability_result.is_stable else 'instável'} com {stability_result.sign_changes} trocas de sinal.")

if __name__ == '__main__':
    unittest.main()
