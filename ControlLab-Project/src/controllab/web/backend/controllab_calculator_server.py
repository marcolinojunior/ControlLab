"""
ControlLab Calculator Server
Sistema avançado de análise de controle com interface SymboLab-like

Este servidor integra todas as funcionalidades dos módulos 1-7 do ControlLab
para fornecer análises pedagógicas detalhadas de sistemas de controle.
"""

import asyncio
import websockets
import json

import sys
import traceback
import sympy as sp
import numpy as np
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Importações do ControlLab
from controllab.core import SymbolicTransferFunction
from controllab.analysis import analyze_stability, analyze_step_response, analyze_impulse_response, compare_responses
from controllab.analysis import quick_stability_check
from controllab.design.compensators import design_pid_tuning

# Encoder JSON seguro para objetos SymPy
class ControlLabJSONEncoder(json.JSONEncoder):
    """Encoder JSON customizado para objetos SymPy e ControlLab"""
    
    def default(self, obj):
        """Converte objetos não serializáveis em formato JSON"""
        
        # Objetos SymPy
        if isinstance(obj, sp.Basic):
            return self.serialize_sympy_object(obj)
        
        # Objetos NumPy
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.complex128):
            return {"real": float(obj.real), "imag": float(obj.imag)}
        
        # Números complexos Python
        elif isinstance(obj, complex):
            return {"real": obj.real, "imag": obj.imag}
        
        # Sets
        elif isinstance(obj, set):
            return list(obj)
        
        # Objetos com __dict__
        elif hasattr(obj, '__dict__'):
            return {
                'object_type': type(obj).__name__,
                'attributes': {k: v for k, v in obj.__dict__.items() if not k.startswith('_')}
            }
        
        # Iteráveis (exceto strings)
        elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes)):
            try:
                return list(obj)
            except:
                return str(obj)
        
        # Fallback para string
        return str(obj)
    
    def serialize_sympy_object(self, obj):
        """Serializa objetos SymPy específicos"""
        
        # Casos específicos para diferentes tipos SymPy
        if isinstance(obj, sp.Pow):
            return {
                'type': 'power',
                'base': str(obj.base),
                'exponent': str(obj.exp),
                'string': str(obj),
                'latex': self.safe_latex(obj)
            }
        
        elif isinstance(obj, sp.Symbol):
            return {
                'type': 'symbol',
                'name': str(obj),
                'string': str(obj)
            }
        
        elif isinstance(obj, sp.Add):
            return {
                'type': 'addition',
                'string': str(obj),
                'latex': self.safe_latex(obj)
            }
        
        elif isinstance(obj, sp.Mul):
            return {
                'type': 'multiplication', 
                'string': str(obj),
                'latex': self.safe_latex(obj)
            }
        
        elif isinstance(obj, (sp.Integer, sp.Rational, sp.Float)):
            return {
                'type': 'number',
                'value': float(obj) if obj.is_finite else str(obj),
                'string': str(obj)
            }
        
        else:
            # Objeto SymPy genérico
            return {
                'type': 'sympy_' + type(obj).__name__.lower(),
                'string': str(obj),
                'latex': self.safe_latex(obj)
            }
    
    def safe_latex(self, obj):
        """Conversão segura para LaTeX"""
        try:
            return sp.latex(obj)
        except:
            return str(obj)

def safe_json_dumps(obj, **kwargs):
    """Função wrapper para serialização segura"""
    try:
        return json.dumps(obj, cls=ControlLabJSONEncoder, ensure_ascii=False, **kwargs)
    except Exception as e:
        # Fallback: converter tudo para string
        fallback_obj = recursive_str_convert(obj)
        return json.dumps(fallback_obj, ensure_ascii=False, **kwargs)

def recursive_str_convert(obj):
    """Converte recursivamente objetos para string como último recurso"""
    if isinstance(obj, dict):
        return {str(k): recursive_str_convert(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [recursive_str_convert(item) for item in obj]
    elif isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    else:
        return str(obj)

# Adicionar o diretório do projeto ao path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    # Importar todos os módulos do ControlLab
    from modulo1 import core_symbolic, educational_guidance
    from modulo2 import numerical_engine, interface_adapter
    from modulo3 import laplace_modeling, transform_engine
    from modulo4 import stability_analyzer, performance_evaluator
    from modulo5 import controller_designer, optimization_tools
    from modulo6 import discrete_systems, conversion_tools
    from modulo7 import visualization_engine, advanced_plots
    
    print("✓ Todos os módulos ControlLab importados com sucesso")
except ImportError as e:
    print(f"⚠ Erro na importação dos módulos: {e}")
    # Continuar com funcionalidade limitada


# Nova classe: ControlLabCalculator delegando para ControlLab
class ControlLabCalculator:
    """
    Calculadora baseada no pacote ControlLab
    """
    def __init__(self):
        pass

    def parse_input(self, user_input: str) -> Dict[str, Any]:
        user_input = user_input.strip()
        # Detectar se é uma função de transferência
        if 'G(s)' in user_input or 'H(s)' in user_input or 'C(s)' in user_input:
            return self.parse_transfer_function(user_input)
        # Detectar comandos de análise
        analysis_commands = {
            'estabilidade': 'stability_analysis',
            'stability': 'stability_analysis',
            'bode': 'bode_analysis',
            'nyquist': 'nyquist_analysis',
            'lugar': 'root_locus_analysis',
            'root locus': 'root_locus_analysis',
            'degrau': 'step_response_analysis',
            'step': 'step_response_analysis',
            'impulso': 'impulse_response_analysis',
            'impulse': 'impulse_response_analysis',
            'pid': 'pid_design_analysis',
            'polos': 'pole_zero_analysis',
            'zeros': 'pole_zero_analysis',
            'poles': 'pole_zero_analysis',
            'discretiz': 'discretization_analysis'
        }
        for keyword, analysis_type in analysis_commands.items():
            if keyword in user_input.lower():
                return {
                    'type': 'analysis_command',
                    'analysis_type': analysis_type,
                    'original_input': user_input,
                    'system': self.extract_system_from_command(user_input)
                }
        # Se não detectou padrão específico, tentar interpretar como sistema
        return self.parse_general_system(user_input)

    def parse_transfer_function(self, input_str: str) -> Dict[str, Any]:
        try:
            if '=' in input_str:
                equation_part = input_str.split('=', 1)[1].strip()
            else:
                equation_part = input_str
            equation_part = equation_part.replace('^', '**')
            expr = sp.sympify(equation_part)
            tf = SymbolicTransferFunction(sp.numer(expr), sp.denom(expr))
            return {
                'type': 'transfer_function',
                'tf_obj': tf,
                'original_input': input_str,
                'variables': list(expr.free_symbols)
            }
        except Exception as e:
            return {
                'type': 'error',
                'message': f"Erro ao interpretar função de transferência: {str(e)}",
                'original_input': input_str
            }

    def extract_system_from_command(self, command: str) -> Any:
        import re
        pattern = r'[GHC]\(s\)\s*=\s*([^,\n]+)'
        match = re.search(pattern, command)
        if match:
            tf_str = match.group(1).strip()
            try:
                tf_str = tf_str.replace('^', '**')
                expr = sp.sympify(tf_str)
                tf = SymbolicTransferFunction(sp.numer(expr), sp.denom(expr))
                return {'tf_obj': tf, 'type': 'transfer_function'}
            except:
                pass
        return None

    def parse_general_system(self, input_str: str) -> Dict[str, Any]:
        try:
            expr_str = input_str.replace('^', '**')
            expr = sp.sympify(expr_str)
            tf = SymbolicTransferFunction(sp.numer(expr), sp.denom(expr))
            return {
                'type': 'mathematical_expression',
                'tf_obj': tf,
                'original_input': input_str
            }
        except:
            return {
                'type': 'natural_language',
                'text': input_str,
                'original_input': input_str
            }

    async def analyze_system(self, parsed_input: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if parsed_input['type'] == 'error':
                return parsed_input
            result = {'type': 'analysis_complete', 'input_analysis': parsed_input}
            if parsed_input['type'] in ['transfer_function', 'mathematical_expression']:
                tf_obj = parsed_input['tf_obj']
                result['stability_analysis'] = analyze_stability(tf_obj)
                result['step_response'] = analyze_step_response(tf_obj)
                result['impulse_response'] = analyze_impulse_response(tf_obj)
                result['quick_stability'] = quick_stability_check(tf_obj)
            elif parsed_input['type'] == 'analysis_command':
                if parsed_input.get('system'):
                    tf_obj = parsed_input['system']['tf_obj']
                    if parsed_input['analysis_type'] == 'stability_analysis':
                        result['stability_analysis'] = analyze_stability(tf_obj)
                    elif parsed_input['analysis_type'] == 'step_response_analysis':
                        result['step_response'] = analyze_step_response(tf_obj)
                    elif parsed_input['analysis_type'] == 'impulse_response_analysis':
                        result['impulse_response'] = analyze_impulse_response(tf_obj)
                    elif parsed_input['analysis_type'] == 'pid_design_analysis':
                        result['pid_design'] = design_pid_tuning(tf_obj)
                    # Adicione outros tipos conforme necessário
                else:
                    result['message'] = 'Nenhum sistema detectado para análise.'
            elif parsed_input['type'] == 'natural_language':
                result['message'] = 'Entrada não reconhecida como sistema ou comando.'
            return result
        except Exception as e:
            return {'type': 'error', 'message': f'Erro na análise: {str(e)}', 'traceback': traceback.format_exc()}

class WebSocketServer:
    """Servidor WebSocket para interface ControlLab"""
    
    def __init__(self):
        self.calculator = ControlLabCalculator()
        self.clients = set()
    
    async def register_client(self, websocket):
        """Registra novo cliente"""
        self.clients.add(websocket)
        print(f"Cliente conectado. Total: {len(self.clients)}")
    
    async def unregister_client(self, websocket):
        """Remove cliente"""
        self.clients.discard(websocket)
        print(f"Cliente desconectado. Total: {len(self.clients)}")
    
    async def handle_client(self, websocket):
        """Gerencia conexão com cliente"""
        await self.register_client(websocket)
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    response = await self.process_request(data)
                    await websocket.send(safe_json_dumps(response))
                except json.JSONDecodeError:
                    error_response = {
                        'type': 'error',
                        'message': 'Formato JSON inválido'
                    }
                    await websocket.send(safe_json_dumps(error_response))
                except Exception as e:
                    error_response = {
                        'type': 'error', 
                        'message': f'Erro interno: {str(e)}'
                    }
                    await websocket.send(safe_json_dumps(error_response))
        finally:
            await self.unregister_client(websocket)
    
    async def process_request(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Processa requisição do cliente"""
        
        if data.get('type') == 'analyze':
            user_input = data.get('input', '')
            
            # Parse da entrada
            parsed_input = self.calculator.parse_input(user_input)
            
            # Análise do sistema
            analysis_result = await self.calculator.analyze_system(parsed_input)
            
            # Formato de resposta para interface
            return {
                'type': 'analysis_result',
                'timestamp': data.get('timestamp'),
                'original_input': user_input,
                'parsed_input': parsed_input,
                'analysis': analysis_result,
                'success': analysis_result.get('type') != 'error'
            }
        
        elif data.get('type') == 'ping':
            return {
                'type': 'pong',
                'timestamp': data.get('timestamp'),
                'server_status': 'online'
            }
        
        else:
            return {
                'type': 'error',
                'message': f'Tipo de requisição não reconhecido: {data.get("type")}'
            }

async def main():
    """Função principal do servidor"""
    print("🚀 Iniciando ControlLab Calculator Server...")
    print("📊 Interface SymboLab-like para Sistemas de Controle")
    print("🔗 Integração completa com módulos 1-7 ControlLab")
    print("-" * 60)
    
    server = WebSocketServer()
    
    # Iniciar servidor WebSocket
    start_server = websockets.serve(
        server.handle_client, 
        "localhost", 
        8765,
        ping_interval=30,
        ping_timeout=10
    )
    
    print("✓ Servidor WebSocket iniciado em ws://localhost:8765")
    print("✓ Pronto para receber conexões...")
    print("📱 Abra controllab_interface.html em seu navegador")
    print("\n💡 Exemplos de entrada:")
    print("   • G(s) = 1/(s^2 + 2*s + 1)")
    print("   • Analise a estabilidade de G(s) = K/(s*(s+1))")
    print("   • Projete um controlador PID")
    print("-" * 60)
    
    await start_server
    await asyncio.Future()  # Executar indefinidamente

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Servidor interrompido pelo usuário")
    except Exception as e:
        print(f"\n❌ Erro fatal: {e}")
        traceback.print_exc()
