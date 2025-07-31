# -*- coding: utf-8 -*-

"""
================================================================================
ControlLab - Solução Completa e Pedagógica do Forno Industrial
================================================================================
Este script implementa a solução completa para o problema de controle de um
forno industrial, utilizando a filosofia de dualidade simbólico-numérica
e as capacidades de relatório da biblioteca ControlLab.

O código executa cada passo da análise e do projeto, desde a modelagem
simbólica até a simulação numérica e validação final, gerando um
relatório detalhado ao final do processo.
"""

# --- Importações da Biblioteca ControlLab e Ferramentas Padrão ---
try:
    from controllab.core.symbolic_tf import SymbolicTransferFunction
    from controllab.core.symbolic_utils import create_laplace_variable
    from controllab.core.history import OperationHistory
    from controllab.modeling.conversions import feedback_connection
    from controllab.analysis.stability_analysis import StabilityAnalysisEngine
    from controllab.design.compensators import PID
    from controllab.design.specifications import verify_specifications
    from controllab.design.design_utils import DesignSpecifications
    from hotfix_module import PatchedNumericTransferFunction, patched_simulate_system_response
except ImportError as e:
    print(f"❌ ERRO CRÍTICO DE IMPORTAÇÃO: {e}")
    print("   Certifique-se de que a biblioteca ControlLab está instalada e acessível.")
    exit()

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

class ReportGenerator:
    """Coleta informações e gera um relatório Markdown completo no final."""
    def __init__(self, title):
        self.title = title
        self.sections = []

    def add_section(self, title, content):
        self.sections.append({"title": title, "content": content})

    def write_to_file(self, filename="relatorio_final_completo.md"):
        full_content = f"# ✅ {self.title}\n"
        full_content += "*Relatório gerado utilizando a filosofia de dualidade simbólico-numérica da biblioteca ControlLab*\n\n---\n\n"
        for section in self.sections:
            full_content += f"## 🔹 {section['title']}\n"
            full_content += f"{section['content']}\n\n---\n\n"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(full_content)
        print(f"\n✅ Relatório final '{filename}' gerado com sucesso!")

def main():
    """Executa o fluxo completo de solução do problema do forno."""
    report = ReportGenerator("Solução Completa do Problema do Forno Industrial")
    history = OperationHistory()

    # ==========================================================================
    # 1. & 2. Modelagem Simbólica
    # ==========================================================================
    print("\n--- ⚙️  ETAPA 1 & 2: Modelagem Simbólica do Sistema ---")
    s = create_laplace_variable('s')
    planta_tf = SymbolicTransferFunction(5.0, 100.0 * s + 1, s)
    history.add_step("Modelagem Planta", "Criação da FT da planta", None, str(planta_tf))
    print(f"✅ Planta: {planta_tf}")

    report.add_section(
        "Modelagem da Planta",
        f"**Justificativa:** Criar o modelo matemático da planta.\n\n"
        f"**Função de Transferência G(s):**\n```latex\n{sp.latex(planta_tf.numerator / planta_tf.denominator)}\n```"
    )

    # ==========================================================================
    # 5. Projeto do Controlador PI
    # ==========================================================================
    print("\n--- 🛠️  ETAPA 5: Projeto do Controlador PI ---")
    Kp, Ti = 10.0, 20.0
    controlador_pi = PID(Kp, Ki=Kp/Ti, Kd=0.0, variable='s').simplify()
    history.add_step("Projeto Controlador", "Criação do controlador PI", None, str(controlador_pi))
    print(f"✅ Controlador PI: {controlador_pi}")
    report.add_section(
        "Projeto do Controlador PI",
        f"**Justificativa:** Um controlador PI é projetado para zerar o erro em regime permanente e melhorar o tempo de resposta.\n\n"
        f"**Função de Transferência C(s) para Kp={Kp}, Ti={Ti}:**\n```latex\n{sp.latex(controlador_pi.numerator / controlador_pi.denominator)}\n```"
    )

    # ==========================================================================
    # 6. Análise Simbólica da Malha Fechada
    # ==========================================================================
    print("\n--- 🔄 ETAPA 6: Análise da Malha Fechada ---")
    malha_aberta = (planta_tf * controlador_pi).simplify()
    sistema_final_tf = (malha_aberta / (1 + malha_aberta)).simplify()
    history.add_step("Malha Fechada", "Cálculo da FT de malha fechada", str(malha_aberta), str(sistema_final_tf))
    print(f"✅ Sistema em Malha Fechada: {sistema_final_tf}")
    report.add_section(
        "Análise da Malha Fechada",
        f"**Justificativa:** Obter a função de transferência do sistema completo com o controlador para análise.\n\n"
        f"**Função de Transferência H(s):**\n```latex\n{sp.latex(sistema_final_tf.numerator / sistema_final_tf.denominator)}\n```"
    )

    # ==========================================================================
    # 7. Análise de Estabilidade
    # ==========================================================================
    print("\n--- 🛡️  ETAPA 7: Análise de Estabilidade ---")
    stability_engine = StabilityAnalysisEngine()
    stability_report_obj = stability_engine.comprehensive_analysis(sistema_final_tf)
    print("✅ Relatório de estabilidade gerado.")
    report.add_section(
        "Análise de Estabilidade (Routh-Hurwitz)",
        f"**Justificativa:** Garantir matematicamente que o sistema em malha fechada é estável.\n\n"
        f"**Relatório Detalhado Gerado pela Biblioteca:**\n```text\n{stability_report_obj.get_detailed_analysis()}\n```"
    )

    # ==========================================================================
    # 9. Simulação Numérica e Validação
    # ==========================================================================
    print("\n--- 🖥️  ETAPA 9: Simulação Numérica e Validação ---")
    print("Iniciando conversão S->N com o módulo de correção...")
    sistema_numerico = PatchedNumericTransferFunction.from_symbolic(sistema_final_tf)
    history.add_step("Conversão S->N (Corrigida)", "Conversão da FT final para formato numérico via hotfix", str(sistema_final_tf), str(sistema_numerico))
    print(f"✅ Conversão para numérico bem-sucedida: {sistema_numerico}")

    t_sim = np.linspace(0, 400, 2000)
    time, output = patched_simulate_system_response(sistema_numerico, 'step', t_sim)
    print("✅ Simulação da resposta ao degrau concluída.")

    specs = DesignSpecifications(overshoot_percent=5.0, settling_time_seconds=300.0)
    verification_result = verify_specifications(
        closed_loop_response=(time, output),
        specifications=specs
    )

    print("Resultados da verificação de desempenho:")
    print(f"  - Sobressinal: {verification_result['metrics']['overshoot']:.2f}% (Requisito: < {specs.overshoot_percent}%)")
    print(f"  - Tempo de Acomodação: {verification_result['metrics']['settling_time_seconds']:.2f}s (Requisito: < {specs.settling_time_seconds}s)")

    # Gerar o gráfico
    plt.figure(figsize=(14, 8))
    plt.plot(time, output, label='Resposta do Forno com Controlador PI', color='deepskyblue', linewidth=2.5)
    plt.axhline(y=1.0, color='black', linestyle='--', label='Referência (Setpoint)')
    plt.axhline(y=1.0 + specs.overshoot_percent/100, color='red', linestyle=':', label=f'Limite de Sobressinal ({specs.overshoot_percent}%)')
    plt.title('Simulação Final do Sistema Controlado', fontsize=16, fontweight='bold')
    plt.xlabel('Tempo (segundos)', fontsize=12)
    plt.ylabel('Temperatura Normalizada', fontsize=12)
    plt.grid(True)
    plt.legend()
    plt.savefig("simulacao_final_forno.png")
    print("✅ Gráfico da simulação salvo como 'simulacao_final_forno.png'")

    report.add_section(
        "Simulação Numérica e Validação Final",
        f"**Justificativa:** Validar o projeto através da simulação e verificar se as métricas de desempenho atendem aos requisitos.\n\n"
        f"**Resultados da Simulação:**\n"
        f"- Sobressinal Obtido: **{verification_result['metrics']['overshoot']:.2f}%**\n"
        f"- Tempo de Acomodação Obtido: **{verification_result['metrics']['settling_time_seconds']:.2f}s**\n\n"
        f"**Conclusão Final:** {'O controlador projetado **atende** a todas as especificações.' if verification_result['all_specs_met'] else 'O controlador projetado **não atende** a todas as especificações.'}\n\n"
        f"![Gráfico da Simulação](simulacao_final_forno.png)"
    )

    # ==========================================================================
    # Relatório de Histórico
    # ==========================================================================
    print("\n--- 📖 ETAPA FINAL: Histórico de Operações da Biblioteca ---")
    hist_text = history.get_formatted_steps(format_type='text')
    print(hist_text)
    report.add_section(
        "Histórico de Operações (Gerado pela Biblioteca)",
        f"**Justificativa:** Demonstrar a capacidade de rastreamento pedagógico do `ControlLab`.\n\n"
        f"**Histórico Completo:**\n```text\n{hist_text}\n```\n\n"
    )

    # --- Geração do Arquivo de Relatório Final ---
    report.write_to_file()

if __name__ == '__main__':
    main()
