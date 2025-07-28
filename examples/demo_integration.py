"""
Demonstração de Workflow Completo
==================================

Este script demonstra um fluxo de trabalho completo com o ControlLab,
desde a definição do sistema até a análise de estabilidade e projeto.
"""

import sympy as sp
from controllab.core.symbolic_tf import SymbolicTransferFunction
from controllab.analysis.stability_analysis import analyze_stability
from controllab.design.compensators import design_pid_tuning

def main():
    print("🎯 DEMONSTRAÇÃO DE WORKFLOW COMPLETO")
    print("=" * 50)

    s = sp.Symbol('s')

    # 1. Definição do Sistema (Planta)
    print("\n1️⃣ DEFINIÇÃO DO SISTEMA")
    plant = SymbolicTransferFunction(1, (s**2 + s + 1)*(s + 2), s)
    print(f"   Planta do sistema: {plant}")

    # 2. Análise de Estabilidade da Planta
    print("\n2️⃣ ANÁLISE DE ESTABILIDADE (PLANTA EM MALHA ABERTA)")

    # Usando a nova API
    report = analyze_stability(plant)

    print(f"   {report.get_executive_summary()}")

    # 3. Projeto de um Controlador PID
    print("\n3️⃣ PROJETO DE CONTROLADOR PID")

    # Exemplo de projeto de um controlador
    # Nota: os requisitos de projeto são apenas para demonstração
    controller_result = design_pid_tuning(plant, 'ziegler-nichols')
    controller = controller_result.controller
    print(f"   Controlador projetado: {controller}")

    # 4. Sistema em Malha Fechada
    print("\n4️⃣ SISTEMA EM MALHA FECHADA")
    closed_loop_system = plant.feedback(controller)
    print(f"   Função de transferência em malha fechada: {closed_loop_system}")

    # 5. Análise de Estabilidade do Sistema em Malha Fechada
    print("\n5️⃣ ANÁLISE DE ESTABILIDADE (MALHA FECHADA)")

    closed_loop_report = analyze_stability(closed_loop_system)

    print(f"   {closed_loop_report.get_executive_summary()}")

    # 6. Histórico de Operações
    print("\n6️⃣ HISTÓRICO DE OPERAÇÕES")

    # Usando a nova API de histórico
    history_report = closed_loop_system.history.get_formatted_report()
    print(history_report)

    print("\n" + "=" * 50)
    print("🎉 WORKFLOW COMPLETO DEMONSTRADO!")

if __name__ == "__main__":
    main()
