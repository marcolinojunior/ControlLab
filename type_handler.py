import json
import numpy as np
import control as ct
import sympy as sp
from control.xferfcn import TransferFunction
from control.statesp import StateSpace
from control.frdata import FRD

def format_response(result, command=""):
    """
    Serializes a result object into a structured JSON format for the frontend.
    """
    try:
        # Handle plot data from control.step_info
        # step_info returns a dictionary with keys 'T', 'y'
        if isinstance(result, dict) and all(k in result for k in ['T', 'y']):
            return {
                "chartData": {
                    "type": "step_response",
                    "x": result['T'].tolist(),
                    "y": result['y'].tolist(),
                    "title": "Resposta ao Degrau",
                    "xlabel": "Tempo (s)",
                    "ylabel": "Amplitude"
                }
            }

        # Handle plot data from control.bode_plot
        # bode_plot returns (mag, phase, omega) when plot=False
        if isinstance(result, tuple) and len(result) == 3 and isinstance(result[0], np.ndarray):
            mag, phase, omega = result
            return {
                "chartData": {
                    "type": "bode",
                    "x": omega.tolist(),
                    "y1": (20 * np.log10(mag)).tolist(), # Magnitude in dB
                    "y2": np.rad2deg(phase).tolist(), # Phase in degrees
                    "title": "Diagrama de Bode",
                    "xlabel": "Frequência (rad/s)",
                    "ylabel1": "Magnitude (dB)",
                    "ylabel2": "Fase (graus)"
                }
            }

        # Handle plot data from control.root_locus
        # root_locus returns (rlist, klist) when plot=False
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], np.ndarray):
            rlist, klist = result
            # Separate real and imaginary parts for plotting
            real_parts = [r.real for r in rlist]
            imag_parts = [r.imag for r in rlist]
            return {
                "chartData": {
                    "type": "root_locus",
                    "real": real_parts,
                    "imag": imag_parts,
                    "title": "Lugar das Raízes",
                    "xlabel": "Eixo Real",
                    "ylabel": "Eixo Imaginário"
                }
            }

        # Handle TransferFunction and StateSpace objects
        if isinstance(result, (TransferFunction, StateSpace)):
            # Use sympy for beautiful LaTeX printing
            try:
                if isinstance(result, TransferFunction):
                    num, den = result.num[0][0], result.den[0][0]
                    s = sp.Symbol('s')
                    num_expr = sum(coef * s**i for i, coef in enumerate(reversed(num)))
                    den_expr = sum(coef * s**i for i, coef in enumerate(reversed(den)))
                    latex_str = sp.latex(num_expr / den_expr)
                else: # StateSpace
                    latex_str = sp.latex(result.A) # Just show matrix A for now as an example

                return {
                    "solution": {
                        "text": str(result),
                        "latex": latex_str
                    }
                }
            except Exception:
                return {"solution": {"text": str(result), "latex": ""}}


        # Handle simple string messages (e.g., variable defined)
        if isinstance(result, str):
            return {"solution": {"text": result}}

        # Handle exceptions
        if isinstance(result, Exception):
            return {
                "error": {
                    "message": f"Erro ao executar o comando: {str(result)}",
                    "command": command
                }
            }

        # Default fallback
        return {"solution": {"text": str(result)}}

    except Exception as e:
        return {
            "error": {
                "message": f"Erro inesperado durante a formatação da resposta: {str(e)}",
                "command": command
            }
        }
