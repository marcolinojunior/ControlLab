"""
Demonstração da Filosofia de "Diagnóstico Inteligente"
======================================================

Este script demonstra a filosofia de "Diagnóstico Inteligente" do ControlLab,
mostrando como o sistema captura erros e fornece feedback útil.
"""

# --- INÍCIO DO CÓDIGO DE CORREÇÃO ---
import sys
import os

# Adiciona o diretório raiz do projeto (um nível acima de 'examples') ao path do Python
# para que ele possa encontrar a pasta 'src'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ControlLab-Project'))
sys.path.insert(0, project_root)
# --- FIM DO CÓDIGO DE CORREÇÃO ---

import sympy as sp
from src.controllab.core.symbolic_tf import SymbolicTransferFunction
from controllab.design.antiwindup import design_antiwindup_compensation, SaturationLimits

def main():
    print("🎯 DEMONSTRAÇÃO DO DIAGNÓSTICO INTELIGENTE")
    print("=" * 50)

    print("\n--- Demonstração do Diagnóstico Inteligente ---")

    try:
        # Criar um controlador PI incorreto, como no nosso teste
        s = sp.Symbol('s')

        # Erro: Controlador não é estritamente próprio, o que é um requisito
        # para o projeto de anti-windup.
        flawed_pi = SymbolicTransferFunction(5 + 2/s, s)
        plant = SymbolicTransferFunction(1, s + 1, s)

        print("\n🤔 TENTATIVA DE AÇÃO INCORRETA:")
        print(f"   Controlador: {flawed_pi}")
        print(f"   Planta: {plant}")
        print("   Ação: Projetar compensação anti-windup")

        # Tentar usar uma função que falhará devido ao controlador incorreto
        design_antiwindup_compensation(flawed_pi, plant, SaturationLimits(u_min=-1, u_max=1), 'back_calculation')

    except ValueError as e:
        print("\n✅ Ocorreu um erro, como esperado!")
        print("   O ControlLab forneceu o seguinte relatório de diagnóstico:")
        print("="*50)
        print(e) # Imprime a mensagem de erro rica em contexto
        print("="*50)

        print("\n🔍 ANÁLISE DO ERRO:")
        print("   - A mensagem de erro é clara e informativa.")
        print("   - Ela explica por que a operação falhou (controlador não é estritamente próprio).")
        print("   - Fornece contexto sobre a verificação que falhou.")
        print("   - Sugere possíveis soluções ou próximos passos.")

    print("\n" + "=" * 50)
    print("🎉 DIAGNÓSTICO INTELIGENTE DEMONSTRADO!")

if __name__ == "__main__":
    main()
