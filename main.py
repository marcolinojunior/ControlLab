import asyncio
import json
import websockets
import control as ct
import numpy as np
import sympy as sp

# Import the response formatter we will create
from type_handler import format_response

# All functions from the control library that are safe to expose
SAFE_CONTROL_FUNCTIONS = {
    'tf': ct.tf,
    'ss': ct.ss,
    'step_response': ct.step_info,
    'bode_plot': lambda sys, **kwargs: ct.bode_plot(sys, plot=False, **kwargs),
    'root_locus': lambda sys, **kwargs: ct.root_locus(sys, plot=False, **kwargs),
    'nyquist_plot': lambda sys, **kwargs: ct.nyquist_plot(sys, plot=False, **kwargs),
    'margin': ct.margin,
    'feedback': ct.feedback,
    'series': ct.series,
    'parallel': ct.parallel,
}

async def handler(websocket, path):
    """
    Handles a WebSocket connection. Each connection has its own session context.
    """
    print("Cliente conectado.")
    # Cada conexão tem seu próprio contexto para armazenar variáveis (G, H, etc.)
    session_context = {
        'ct': ct,
        'np': np,
        'sp': sp,
        **SAFE_CONTROL_FUNCTIONS
    }

    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                commands = data.get('commands', [])

                if not isinstance(commands, list):
                    raise ValueError("A entrada 'commands' deve ser uma lista.")

                final_result = None
                last_var_name = None

                for cmd in commands:
                    cmd = cmd.strip()
                    if not cmd:
                        continue

                    # Check for assignment to store the variable name
                    if '=' in cmd:
                        var_name = cmd.split('=')[0].strip()
                        if not var_name.isidentifier():
                            raise NameError(f"'{var_name}' não é um nome de variável válido.")
                        exec(cmd, session_context)
                        last_var_name = var_name
                        final_result = f"Variável '{var_name}' definida."
                    else:
                        # This is an analysis command.
                        # We need to handle cases like `bode` which might not have a system passed in.
                        # A simple approach: if a function is called without args, apply it to the last defined system.
                        if '(' not in cmd and ')' not in cmd and last_var_name:
                            eval_cmd = f"{cmd}({last_var_name})"
                        else:
                            eval_cmd = cmd

                        final_result = eval(eval_cmd, session_context)

                # Format the final result and send it back
                response = format_response(final_result)
                await websocket.send(json.dumps(response))

            except Exception as e:
                # Send error message back to the client
                error_response = format_response(e, command=cmd if 'cmd' in locals() else 'N/A')
                await websocket.send(json.dumps(error_response))

    except websockets.exceptions.ConnectionClosed:
        print("Cliente desconectado.")
    except Exception as e:
        print(f"Erro inesperado na conexão: {e}")

async def main():
    """
    Starts the WebSocket server.
    """
    host = "localhost"
    port = 8765
    print(f"Servidor WebSocket iniciado em ws://{host}:{port}")
    async with websockets.serve(handler, host, port):
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())
