# debug_integration.py
print("--- INÍCIO DO TESTE DE INTEGRAÇÃO MANUAL ---")
print("Vamos importar os módulos um por um para encontrar o ciclo.")

try:
    print("\n[PASSO 1] A testar o núcleo - `core`...")
    print("Importando symbolic_tf...")
    from controllab.core import symbolic_tf
    print("Importando symbolic_ss...")
    from controllab.core import symbolic_ss
    print("✅ SUCESSO: O módulo `core` foi importado sem problemas.")

    print("\n[PASSO 2] A testar os utilitários de análise - `analysis.utils`...")
    print("Importando stability_utils...")
    from controllab.analysis import stability_utils
    print("✅ SUCESSO: O módulo `stability_utils` foi importado sem problemas.")

    print("\n[PASSO 3] A testar o motor de Routh-Hurwitz...")
    print("Importando routh_hurwitz...")
    from controllab.analysis import routh_hurwitz
    print("✅ SUCESSO: O módulo `routh_hurwitz` foi importado sem problemas.")

    print("\n[PASSO 4] A testar o motor do Lugar das Raízes...")
    print("Importando root_locus...")
    from controllab.analysis import root_locus
    print("✅ SUCESSO: O módulo `root_locus` foi importado sem problemas.")

    print("\n[PASSO 5] A testar o motor de Resposta em Frequência...")
    print("Importando frequency_response...")
    from controllab.analysis import frequency_response
    print("✅ SUCESSO: O módulo `frequency_response` foi importado sem problemas.")

    print("\n[PASSO 6] A testar o orquestrador - `stability_analysis` (SUSPEITO PRINCIPAL)...")
    print("Importando stability_analysis...")
    from controllab.analysis import stability_analysis
    print("✅ SUCESSO: O orquestrador `stability_analysis` foi importado sem problemas.")

    print("\n🎉 DIAGNÓSTICO: Todas as importações foram concluídas. O ciclo não está no carregamento dos módulos.")

except ImportError as e:
    print(f"\n❌ FALHA: Ocorreu um erro de importação inesperado: {e}")

except Exception as e:
    print(f"\n❌ FALHA: Ocorreu um erro inesperado: {e}")

print("\n--- FIM DO TESTE DE INTEGRAÇÃO MANUAL ---")
