# -*- coding: utf-8 -*-

"""
================================================================================
ControlLab - Depuração, Simulação e Relatório do Forno Industrial
================================================================================
Este script executa uma depuração passo a passo do problema do forno,
imprimindo os resultados de cada função da biblioteca ControlLab no console
em tempo real. Ao final, gera um relatório completo em Markdown.

O objetivo é testar a funcionalidade e a saída de cada componente da
biblioteca, conforme a documentação, para garantir que tudo funcione como
esperado.
"""

# --- Importações da Biblioteca ControlLab ---
# Importações baseadas estritamente na DOCUMENTACAO_FUNCOES_COMPLETA.md
from controllab.core.symbolic_tf import SymbolicTransferFunction
from controllab.core.symbolic_utils import create_laplace_variable, convert_to_latex_formatted
from controllab.core.history import OperationHistory

from controllab.modeling.conversions import feedback_connection
from controllab.modeling.laplace_transform import LaplaceTransformer

from controllab.analysis.stability_analysis import StabilityAnalysisEngine
from controllab.analysis.routh_hurwitz import RouthHurwitzAnalyzer
from controllab.analysis.root_locus import RootLocusAnalyzer

from controllab.design.compensators import PID
from controllab.design.design_utils import DesignSpecifications
from controllab.design.specifications import verify_specifications

from controllab.numerical.interface import NumericalInterface

# Bibliotecas padrão
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

# --- Classe para Geração do Relatório Final ---
class ReportGenerator:
    def __init__(self, title):
        self.title = title
        self.sections = []

    def add_section(self, title, content):
        self.sections.append({"title": title, "content": content})

    def write_to_file(self, filename="relatorio_final.md"):
        full_content = f"# ✅ {self.title}\n"
        full_content += "*Gerado utilizando as funcionalidades de relatório da biblioteca ControlLab*\n\n---\n\n"

        for section in self.sections:
            full_content += f"## 🔹 {section['title']}\n"
            full_content += f"{section['content']}\n\n---\n\n"

        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(full_content)
            print(f"\n✅ Relatório '{filename}' gerado com sucesso!")
        except Exception as e:
            print(f"\n❌ Erro ao salvar o relatório: {e}")

# --- Função Principal de Execução ---
def main():
    report = ReportGenerator("Relatório Pedagógico do Problema do Forno Industrial")

    # ==========================================================================
    # 1. & 2. Modelagem do Sistema e Transformada de Laplace
    # ==========================================================================
    print("\n--- ⚙️  ETAPA 1 & 2: Modelagem e Transformada de Laplace ---")
    s = create_laplace_variable('s')
    K, tau = 5.0, 100.0
    planta_tf = SymbolicTransferFunction(K, tau * s + 1, s)
    print(f"✅ Função de Transferência da Planta G(s): {planta_tf}")

    report.add_section(
        "Modelagem do Sistema e Função de Transferência",
        f"**Justificativa:** Representar o sistema físico por um modelo matemático para análise.\n\n"
        f"**Resultado (G(s)):**\n```latex\n{sp.latex(planta_tf.numerator / planta_tf.denominator)}\n```"
    )

    # ==========================================================================
    # 3. & 4. Análise da Resposta ao Degrau da Planta e Justificativa
    # ==========================================================================
    print("\n--- 📉 ETAPA 3 & 4: Análise da Planta e Justificativa do Controle ---")
    ts_planta = 4 * tau
    print(f"Análise da planta em malha aberta:")
    print(f"  - Tempo de acomodação (ts) ≈ 4 * τ = {ts_planta:.1f} segundos.")
    print(f"  - Conclusão: Sistema muito lento (ts > 300s). É necessário um controlador.")

    report.add_section(
        "Análise da Planta e Justificativa do Controle",
        f"**Justificativa:** Analisar a resposta do sistema sem controle para verificar a necessidade de compensação.\n\n"
        f"**Resultado:**\n- Tempo de Acomodação (ts): **{ts_planta:.1f} segundos**\n"
        f"- Conclusão: O sistema não atende à especificação de tempo de acomodação. **Um controlador é essencial.**"
    )

    # ==========================================================================
    # 5. Projeto do Controlador PI
    # ==========================================================================
    print("\n--- 🛠️  ETAPA 5: Projeto do Controlador PI ---")
    Kp, Ti = 10.0, 20.0
    controlador_pi = PID(Kp, Ki=Kp/Ti, Kd=0.0, variable='s').simplify()
    print(f"✅ Controlador PI Projetado C(s): {controlador_pi}")

    report.add_section(
        "Projeto do Controlador PI",
        f"**Justificativa:** Um controlador PI é usado para zerar o erro em regime permanente e melhorar a velocidade da resposta.\n\n"
        f"**Resultado (C(s)):**\n```latex\n{sp.latex(controlador_pi.numerator / controlador_pi.denominator)}\n```"
    )

    # ==========================================================================
    # 6. LGR (Lugar das Raízes)
    # ==========================================================================
    print("\n--- 📈 ETAPA 6: Análise via Lugar Geométrico das Raízes (LGR) ---")
    malha_aberta_tf = (controlador_pi * planta_tf).simplify()
    print(f"Função de Malha Aberta para LGR, L(s) = C(s)G(s): {malha_aberta_tf}")

    lgr_analyzer = RootLocusAnalyzer()
    lgr_features = lgr_analyzer.get_locus_features(malha_aberta_tf, show_steps=False)
    print("Características do LGR extraídas com sucesso:")
    print(f"  - Polos de Malha Aberta: {lgr_features.poles}")
    print(f"  - Zeros de Malha Aberta: {lgr_features.zeros}")
    print(f"  - Ângulos das Assíntotas: {lgr_features.asymptotes['angles']}")
    print(f"  - Centroide das Assíntotas: {lgr_features.asymptotes['centroid']:.3f}")

    report.add_section(
        "Análise via Lugar Geométrico das Raízes (LGR)",
        f"**Justificativa:** Analisar como os polos da malha fechada se movem com a variação do ganho para avaliar a estabilidade relativa e o desempenho transitório.\n\n"
        f"**Resultado da Análise do LGR (para L(s)):**\n"
        f"- Polos de Malha Aberta: `{lgr_features.poles}`\n"
        f"- Zeros de Malha Aberta: `{lgr_features.zeros}`\n"
        f"- Assíntotas: Ângulos `{lgr_features.asymptotes['angles']}` rad, Centroide em `{lgr_features.asymptotes['centroid']:.3f}`"
    )

    # ==========================================================================
    # 7. Análise de Estabilidade (Critério de Routh-Hurwitz)
    # ==========================================================================
    print("\n--- 🛡️  ETAPA 7: Análise de Estabilidade (Routh-Hurwitz) ---")
    malha_fechada_tf = (malha_aberta_tf / (1 + malha_aberta_tf)).simplify()
    print(f"Função de Malha Fechada H(s): {malha_fechada_tf}")

    stability_engine = StabilityAnalysisEngine()
    stability_report_obj = stability_engine.comprehensive_analysis(malha_fechada_tf, show_all_steps=True)

    print("Resultado da Análise de Estabilidade (extraído do objeto de relatório):")
    # Imprimindo o relatório detalhado diretamente no console
    print(stability_report_obj.get_detailed_analysis())

    report.add_section(
        "Análise de Estabilidade (Routh-Hurwitz)",
        f"**Justificativa:** Garantir matematicamente que o sistema em malha fechada é estável para os ganhos escolhidos.\n\n"
        f"**Relatório Detalhado Gerado pela Biblioteca:**\n```text\n{stability_report_obj.get_detailed_analysis()}\n```"
    )

    # ==========================================================================
    # 8. Erro em Regime Permanente
    # ==========================================================================
    print("\n--- 🎯 ETAPA 8: Análise do Erro em Regime Permanente ---")
    print("Análise teórica:")
    print("  - O controlador PI introduz um polo na origem (integrador), aumentando o tipo do sistema para 1.")
    print("  - Para um sistema do Tipo 1, o erro em regime permanente para uma entrada degrau é teoricamente NULO.")

    report.add_section(
        "Análise do Erro em Regime Permanente",
        f"**Justificativa:** Verificar se o requisito de erro nulo é atendido.\n\n"
        f"**Análise Teórica:**\n- O controlador PI aumenta o tipo do sistema para **Tipo 1**.\n"
        f"- Para uma entrada degrau, um sistema do Tipo 1 tem um erro em regime permanente **nulo** ($e_{{ss}} = 0$)."
    )

    # ==========================================================================
    # 9. Simulação Numérica
    # ==========================================================================
    print("\n--- 🖥️  ETAPA 9: Simulação Numérica e Validação Final ---")
    num_interface = NumericalInterface()
    s_var = malha_fechada_tf.variable
    num_eval = malha_fechada_tf.numerator.evalf()
    den_eval = malha_fechada_tf.denominator.evalf()
    tf_for_conversion = SymbolicTransferFunction(num_eval, den_eval, s_var)
    numeric_sys = num_interface.symbolic_to_control_tf(tf_for_conversion)
    t_sim = np.linspace(0, 400, 2000)

    print("Executando a simulação numérica da resposta ao degrau...")
    time, output = num_interface.compute_step_response(numeric_sys, time_vector=t_sim)
    print("✅ Simulação concluída.")

    specs = DesignSpecifications(overshoot=5.0, settling_time=300.0)
    verification_result = verify_specifications(
        closed_loop_response=(time, output),
        specifications=specs,
    )

    print("Resultados da verificação de desempenho:")
    print(f"  - Sobressinal: {verification_result['metrics']['overshoot']:.2f}% (Requisito: < {specs.overshoot}%)")
    print(f"  - Tempo de Acomodação: {verification_result['metrics']['settling_time']:.2f}s (Requisito: < {specs.settling_time}s)")

    if verification_result['all_specs_met']:
        print("✅ SUCESSO! O sistema controlado atende a todas as especificações de desempenho.")
    else:
        print("❌ ATENÇÃO: O sistema controlado NÃO atende a todas as especificações.")

    # Gerar o gráfico
    plt.figure(figsize=(14, 8))
    plt.plot(time, output, label='Resposta do Forno com Controlador PI', color='deepskyblue', linewidth=2.5)
    plt.axhline(y=1.0, color='black', linestyle='--', label='Referência (Setpoint)')
    plt.axhline(y=1.0 + specs.overshoot/100, color='red', linestyle=':', label=f'Limite de Sobressinal ({specs.overshoot}%)')
    plt.axvline(x=specs.settling_time, color='darkorange', linestyle='--', label=f'Limite de Tempo de Acomodação ({specs.settling_time}s)')
    plt.title('Simulação da Resposta ao Degrau do Sistema Controlado', fontsize=16, fontweight='bold')
    plt.xlabel('Tempo (segundos)', fontsize=12)
    plt.ylabel('Temperatura Normalizada', fontsize=12)
    plt.grid(True, which='both', linestyle='--')
    plt.legend()
    plt.savefig("simulacao_forno.png")
    print("\nGráfico da simulação salvo como 'simulacao_forno.png'")

    report.add_section(
        "Simulação Numérica e Validação Final",
        f"**Justificativa:** Validar o projeto do controlador através da simulação da resposta temporal e verificar se as métricas de desempenho atendem aos requisitos.\n\n"
        f"**Resultados da Simulação:**\n"
        f"- Sobressinal Obtido: **{verification_result['metrics']['overshoot']:.2f}%**\n"
        f"- Tempo de Acomodação Obtido: **{verification_result['metrics']['settling_time']:.2f}s**\n\n"
        f"**Conclusão Final:** {'O controlador projetado **atende** a todas as especificações.' if verification_result['all_specs_met'] else 'O controlador projetado **não atende** a todas as especificações.'}\n\n"
        f"![Gráfico da Simulação](simulacao_forno.png)"
    )

    # --- Geração do Relatório Final ---
    report.write_to_file()

if __name__ == '__main__':
    main()
