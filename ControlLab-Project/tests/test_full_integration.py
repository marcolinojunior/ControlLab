# DENTRO DE: tests/test_full_integration.py
import unittest
print("\n--- A iniciar o Teste de Fumo de Integração Completa do ControlLab ---")

class TestFullModuleIntegration(unittest.TestCase):

    def test_imports_sem_ciclos(self):
        """
        Este teste verifica se todos os principais módulos e classes do ControlLab
        podem ser importados sem causar um ciclo de importação ou bloqueio.
        Não testa a lógica, apenas a capacidade de carregamento do ecossistema.
        """
        print("\n[FASE 1] A verificar a importação dos módulos de baixo nível...")

        try:
            # 1. O Alicerce: Classes de Dados e Utilitários
            from controllab.utils import StabilityResult, ControllerResult, DesignSpecifications
            from controllab.core.history import OperationHistory
            print("  ✅ SUCESSO: Utilitários e Histórico importados.")

            # 2. O Núcleo: As nossas estruturas de dados principais
            from controllab.core.symbolic_tf import SymbolicTransferFunction
            from controllab.core.symbolic_ss import SymbolicStateSpace
            print("  ✅ SUCESSO: Núcleo Simbólico (core) importado.")

        except ImportError as e:
            self.fail(f"FALHA CRÍTICA: Não foi possível importar os módulos de base. Erro: {e}")

        print("\n[FASE 2] A verificar a importação dos 'Motores de Cálculo'...")

        try:
            # 3. Motores de Análise (devem ser independentes)
            from controllab.analysis.routh_hurwitz import RouthHurwitzAnalyzer
            from controllab.analysis.root_locus import RootLocusAnalyzer
            from controllab.analysis.frequency_response import FrequencyAnalyzer
            print("  ✅ SUCESSO: Motores de Análise individuais importados.")

            # 4. Motores de Design (devem ser independentes)
            from controllab.design.compensators import PID, Lead, Lag
            from controllab.design.pole_placement import StateSpaceController
            print("  ✅ SUCESSO: Motores de Design individuais importados.")

        except ImportError as e:
            self.fail(f"FALHA CRÍTICA: Não foi possível importar os motores de cálculo. Erro: {e}")

        print("\n[FASE 3] A verificar a importação dos 'Tradutores' e 'Orquestradores' (onde os ciclos acontecem)...")

        try:
            # 5. A Camada de Formatação (deve ser independente)
            from controllab.analysis.pedagogical_formatter import format_routh_hurwitz_response
            print("  ✅ SUCESSO: Camada de Formatação importada.")

            # 6. O Orquestrador Principal (o teste final)
            # Este é o ficheiro que importa de múltiplos locais e é o mais
            # suscetível a despoletar um ciclo.
            from controllab.analysis.stability_analysis import StabilityAnalysisEngine
            print("  ✅ SUCESSO: O Orquestrador 'StabilityAnalysisEngine' foi importado.")

        except ImportError as e:
            self.fail(f"FALHA CRÍTICA: Não foi possível importar as camadas de alto nível. Erro: {e}")

        print("\n--- Conclusão do Teste de Fumo ---")
        print("🎉 SUCESSO GERAL! Todos os módulos principais foram importados sem bloqueios ou ciclos de importação.")
        self.assertTrue(True, "A arquitetura de importações está saudável.")

if __name__ == '__main__':
    unittest.main()
