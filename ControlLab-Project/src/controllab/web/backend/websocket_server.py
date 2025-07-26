import asyncio
import websockets
import json
import sys
import os
import sympy as sp

# Adiciona o diretório src ao path para que possamos importar os módulos do controllab
# Ajuste o caminho para refletir a estrutura do seu projeto
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))

from src.controllab.core.transforms import LaplaceTransform
from src.controllab.core.symbolic_tf import SymbolicTransferFunction

def to_latex(expr):
    return f"$${sp.latex(expr)}$$"

def format_history(history):
    formatted_steps = []
    for op in history:
        step_str = f"**{op['type']}**: {op['description']}<br>"
        step_str += f"Antes: `{op['before']}`<br>"
        step_str += f"Depois: `{op['after']}`"
        formatted_steps.append(step_str)
    return formatted_steps

async def handler(websocket, path):
    print(f"Cliente conectado: {websocket.remote_address}")

    try:
        async for message in websocket:
            data = json.loads(message)

            if data['type'] == 'analyze_laplace':
                diff_eq_str = data['payload']['diff_eq']
                print(f"Recebido para análise de Laplace: {diff_eq_str}")

                try:
                    # Instancia o analisador de Laplace
                    laplace_analyzer = LaplaceTransform()

                    # Define os símbolos
                    t, s = sp.symbols('t s')
                    y = sp.Function('y')(t)
                    x = sp.Function('x')(t)

                    # Converte a string da EDO para uma expressão sympy
                    lhs, rhs = map(sp.sympify, diff_eq_str.split('='))
                    diff_eq = sp.Eq(lhs, rhs)

                    # Aplica a transformada de Laplace
                    laplace_lhs = laplace_analyzer.transform(diff_eq.lhs, t, s)
                    laplace_rhs = laplace_analyzer.transform(diff_eq.rhs, t, s)

                    # Monta a equação no domínio de Laplace
                    laplace_eq = sp.Eq(laplace_lhs, laplace_rhs)

                    # Formata o histórico
                    steps = format_history(laplace_analyzer.history.get_history())

                    # Adiciona a equação final à lista de passos
                    steps.append(f"**EQUAÇÃO FINAL**: A equação no domínio de Laplace é: {to_latex(laplace_eq)}")

                    response = {
                        'type': 'laplace_result',
                        'steps': steps
                    }
                except Exception as e:
                    response = {
                        'type': 'error',
                        'message': f"Erro ao analisar a equação diferencial: {e}"
                    }

                await websocket.send(json.dumps(response))

    except websockets.exceptions.ConnectionClosed:
        print(f"Cliente desconectado: {websocket.remote_address}")

async def main():
    async with websockets.serve(handler, "localhost", 8765):
        print("Servidor WebSocket (integrado com o núcleo do ControlLab) iniciado em ws://localhost:8765")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
