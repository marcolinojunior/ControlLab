"""
ControlLab Agent Integration - Módulo 8 Completo
Sistema de integração de agentes AI com execução Python e ControlLab

Permite que modelos AI treinados executem código Python e utilizem
o ControlLab através de agentes especializados.
"""

import json
import subprocess
import sys
import ast
import traceback
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import asyncio
import websockets
from pathlib import Path
import importlib.util
import tempfile
import os

# Importações do ControlLab
try:
    from ..analysis.stability_analysis import StabilityAnalysisEngine, analyze_stability
    from ..core.symbolic_tf import SymbolicTransferFunction
    from ..core.symbolic_ss import SymbolicStateSpace
    from ..design.pid_tuning import PIDTuner
    from ..modeling.laplace_transform import LaplaceTransformer
    from ..numerical.interface import NumericalInterface
    from ..visualization.interactive_plots import InteractivePlotGenerator
    from .ai_tutor import SocraticTutor
    from .analysis_maestro import AnalysisMaestro
    
    CONTROLLAB_AVAILABLE = True
except ImportError as e:
    print(f"ControlLab backend não disponível: {e}")
    CONTROLLAB_AVAILABLE = False


class PythonExecutionAgent:
    """
    Agente para execução segura de código Python com acesso ao ControlLab.
    
    Permite que modelos AI executem código Python de forma controlada,
    com sandbox e acesso às funcionalidades do ControlLab.
    """
    
    def __init__(self, max_execution_time: int = 30):
        self.max_execution_time = max_execution_time
        self.execution_history = []
        self.available_modules = self._get_available_modules()
        self.sandbox_globals = self._setup_sandbox()
        
    def _get_available_modules(self) -> Dict[str, Any]:
        """Retorna módulos disponíveis para o agente."""
        modules = {
            'numpy': 'np',
            'matplotlib.pyplot': 'plt',
            'sympy': 'sp',
            'scipy': 'scipy',
        }
        
        if CONTROLLAB_AVAILABLE:
            modules.update({
                'controllab.core': 'cl_core',
                'controllab.analysis': 'cl_analysis',
                'controllab.design': 'cl_design',
                'controllab.modeling': 'cl_modeling',
                'controllab.numerical': 'cl_numerical',
                'controllab.visualization': 'cl_viz'
            })
            
        return modules
    
    def _setup_sandbox(self) -> Dict[str, Any]:
        """Configura sandbox seguro para execução."""
        sandbox = {
            '__builtins__': {
                'print': print,
                'len': len,
                'range': range,
                'enumerate': enumerate,
                'zip': zip,
                'map': map,
                'filter': filter,
                'str': str,
                'int': int,
                'float': float,
                'list': list,
                'dict': dict,
                'tuple': tuple,
                'set': set,
                'abs': abs,
                'min': min,
                'max': max,
                'sum': sum,
                'round': round,
                'type': type,
                'isinstance': isinstance,
                'hasattr': hasattr,
                'getattr': getattr,
            }
        }
        
        # Importa módulos seguros
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            import sympy as sp
            
            sandbox.update({
                'np': np,
                'plt': plt,
                'sp': sp,
            })
            
            if CONTROLLAB_AVAILABLE:
                sandbox.update({
                    'SymbolicTransferFunction': SymbolicTransferFunction,
                    'SymbolicStateSpace': SymbolicStateSpace,
                    'analyze_stability': analyze_stability,
                    'PIDTuner': PIDTuner,
                    'LaplaceTransformer': LaplaceTransformer,
                    'NumericalInterface': NumericalInterface,
                })
                
        except ImportError as e:
            print(f"Módulo não disponível no sandbox: {e}")
            
        return sandbox
    
    def execute_code(self, code: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Executa código Python com segurança.
        
        Args:
            code: Código Python para executar
            context: Contexto adicional para execução
            
        Returns:
            Resultado da execução com output, erros e variáveis
        """
        execution_id = len(self.execution_history) + 1
        start_time = datetime.now()
        
        result = {
            'execution_id': execution_id,
            'timestamp': start_time.isoformat(),
            'code': code,
            'success': False,
            'output': '',
            'error': None,
            'variables': {},
            'plots': [],
            'controllab_results': {}
        }
        
        try:
            # Prepara ambiente de execução
            execution_globals = self.sandbox_globals.copy()
            if context:
                execution_globals.update(context)
                
            # Captura output
            from io import StringIO
            old_stdout = sys.stdout
            sys.stdout = captured_output = StringIO()
            
            # Executa código
            exec(code, execution_globals)
            
            # Restaura stdout
            sys.stdout = old_stdout
            
            # Coleta resultados
            result['success'] = True
            result['output'] = captured_output.getvalue()
            
            # Extrai variáveis criadas
            for key, value in execution_globals.items():
                if not key.startswith('_') and key not in self.sandbox_globals:
                    try:
                        # Serializa apenas tipos básicos
                        if isinstance(value, (int, float, str, list, dict, tuple)):
                            result['variables'][key] = value
                        elif hasattr(value, '__str__'):
                            result['variables'][key] = str(value)
                    except:
                        result['variables'][key] = f"<objeto {type(value).__name__}>"
            
            # Verifica se há resultados específicos do ControlLab
            if CONTROLLAB_AVAILABLE:
                result['controllab_results'] = self._extract_controllab_results(execution_globals)
                
        except Exception as e:
            sys.stdout = old_stdout
            result['error'] = {
                'type': type(e).__name__,
                'message': str(e),
                'traceback': traceback.format_exc()
            }
            
        # Registra execução
        self.execution_history.append(result)
        
        return result
    
    def _extract_controllab_results(self, globals_dict: Dict) -> Dict[str, Any]:
        """Extrai resultados específicos do ControlLab."""
        results = {}
        
        for key, value in globals_dict.items():
            if isinstance(value, SymbolicTransferFunction):
                results[key] = {
                    'type': 'SymbolicTransferFunction',
                    'expression': str(value.expression),
                    'poles': [str(p) for p in value.poles()],
                    'zeros': [str(z) for z in value.zeros()],
                }
            elif isinstance(value, SymbolicStateSpace):
                results[key] = {
                    'type': 'SymbolicStateSpace',
                    'A_matrix': str(value.A),
                    'B_matrix': str(value.B),
                    'C_matrix': str(value.C),
                    'D_matrix': str(value.D),
                }
                
        return results


class ControlLabAgentInterface:
    """
    Interface principal para agentes AI interagirem com o ControlLab.
    
    Fornece métodos de alto nível para análise de sistemas de controle,
    permitindo que agentes AI realizem análises complexas de forma intuitiva.
    """
    
    def __init__(self):
        self.python_agent = PythonExecutionAgent()
        self.session_history = []
        
        if CONTROLLAB_AVAILABLE:
            self.stability_engine = StabilityAnalysisEngine()
            self.maestro = AnalysisMaestro()
            self.tutor = SocraticTutor()
        
    def analyze_transfer_function(self, tf_expression: str, analyses: List[str] = None) -> Dict[str, Any]:
        """
        Analisa uma função de transferência usando o ControlLab.
        
        Args:
            tf_expression: Expressão da função de transferência (ex: "1/(s^2 + 2*s + 1)")
            analyses: Lista de análises a realizar ['stability', 'step', 'bode', 'root_locus']
            
        Returns:
            Resultados das análises
        """
        if not CONTROLLAB_AVAILABLE:
            return {'error': 'ControlLab backend não disponível'}
            
        if analyses is None:
            analyses = ['stability', 'step', 'bode']
            
        code = f"""
# Criação da função de transferência
import sympy as sp
s = sp.Symbol('s')
tf_expr = {tf_expression}
tf = SymbolicTransferFunction(tf_expr)

results = {{'transfer_function': str(tf.expression)}}

# Análises solicitadas
"""
        
        if 'stability' in analyses:
            code += """
# Análise de estabilidade
stability_result = analyze_stability(tf, show_steps=True)
results['stability'] = {
    'stable': stability_result.is_stable,
    'method': stability_result.method_used,
    'details': stability_result.analysis_steps
}
"""
        
        if 'step' in analyses:
            code += """
# Resposta ao degrau
numerical_interface = NumericalInterface()
step_data = numerical_interface.compute_step_response(tf)
results['step_response'] = {
    'time': step_data['time'].tolist(),
    'amplitude': step_data['amplitude'].tolist(),
    'info': step_data['info']
}
"""
        
        if 'bode' in analyses:
            code += """
# Diagrama de Bode
bode_data = numerical_interface.compute_bode_response(tf)
results['bode'] = {
    'frequency': bode_data['frequency'].tolist(),
    'magnitude': bode_data['magnitude'].tolist(),
    'phase': bode_data['phase'].tolist()
}
"""
        
        if 'root_locus' in analyses:
            code += """
# Lugar geométrico das raízes
locus_data = numerical_interface.compute_root_locus(tf)
results['root_locus'] = {
    'real': locus_data['real'].tolist(),
    'imag': locus_data['imag'].tolist(),
    'gains': locus_data['gains'].tolist()
}
"""
        
        code += "\nprint(f'Análise completa de: {tf.expression}')"
        
        execution_result = self.python_agent.execute_code(code)
        
        if execution_result['success']:
            # Extrai resultados da variável 'results'
            if 'results' in execution_result['variables']:
                return execution_result['variables']['results']
        
        return execution_result
    
    def design_pid_controller(self, plant_expression: str, method: str = 'ziegler_nichols') -> Dict[str, Any]:
        """
        Projeta um controlador PID para uma planta.
        
        Args:
            plant_expression: Expressão da planta
            method: Método de sintonia ('ziegler_nichols', 'cohen_coon', 'manual')
            
        Returns:
            Parâmetros do PID e análise de performance
        """
        if not CONTROLLAB_AVAILABLE:
            return {'error': 'ControlLab backend não disponível'}
            
        code = f"""
# Criação da planta
import sympy as sp
s = sp.Symbol('s')
plant_expr = {plant_expression}
plant = SymbolicTransferFunction(plant_expr)

# Projeto do PID
pid_tuner = PIDTuner()
pid_result = pid_tuner.tune_pid(plant, method='{method}')

results = {{
    'plant': str(plant.expression),
    'method': '{method}',
    'pid_parameters': {{
        'Kp': float(pid_result.Kp),
        'Ki': float(pid_result.Ki),
        'Kd': float(pid_result.Kd)
    }},
    'controller_tf': str(pid_result.controller_tf.expression),
    'closed_loop_tf': str(pid_result.closed_loop_tf.expression)
}}

# Análise de performance do sistema em malha fechada
cl_stability = analyze_stability(pid_result.closed_loop_tf)
results['performance'] = {{
    'stable': cl_stability.is_stable,
    'overshoot': pid_result.performance_metrics.get('overshoot', 'N/A'),
    'settling_time': pid_result.performance_metrics.get('settling_time', 'N/A'),
    'rise_time': pid_result.performance_metrics.get('rise_time', 'N/A')
}}

print(f'PID projetado: Kp={pid_result.Kp}, Ki={pid_result.Ki}, Kd={pid_result.Kd}')
"""
        
        execution_result = self.python_agent.execute_code(code)
        
        if execution_result['success'] and 'results' in execution_result['variables']:
            return execution_result['variables']['results']
        
        return execution_result
    
    def compare_systems(self, systems: List[str], analyses: List[str] = None) -> Dict[str, Any]:
        """
        Compara múltiplos sistemas de controle.
        
        Args:
            systems: Lista de expressões de sistemas
            analyses: Análises a realizar para comparação
            
        Returns:
            Comparação dos sistemas
        """
        if not CONTROLLAB_AVAILABLE:
            return {'error': 'ControlLab backend não disponível'}
            
        if analyses is None:
            analyses = ['stability', 'step']
            
        code = f"""
import sympy as sp
s = sp.Symbol('s')

systems_data = []
system_expressions = {systems}

for i, expr in enumerate(system_expressions):
    tf = SymbolicTransferFunction(expr)
    
    system_info = {{
        'index': i,
        'expression': str(tf.expression),
        'poles': [str(p) for p in tf.poles()],
        'zeros': [str(z) for z in tf.zeros()]
    }}
    
    # Análises para cada sistema
    if 'stability' in {analyses}:
        stability = analyze_stability(tf)
        system_info['stability'] = {{
            'stable': stability.is_stable,
            'method': stability.method_used
        }}
    
    if 'step' in {analyses}:
        numerical_interface = NumericalInterface()
        step_data = numerical_interface.compute_step_response(tf)
        system_info['step_info'] = step_data['info']
    
    systems_data.append(system_info)

results = {{
    'comparison': systems_data,
    'summary': f'Comparação de {{len(systems_data)}} sistemas'
}}

print(f'Comparação completa de {{len(systems_data)}} sistemas')
"""
        
        execution_result = self.python_agent.execute_code(code)
        
        if execution_result['success'] and 'results' in execution_result['variables']:
            return execution_result['variables']['results']
        
        return execution_result
    
    def interactive_analysis(self, query: str) -> Dict[str, Any]:
        """
        Análise interativa usando linguagem natural.
        
        Args:
            query: Pergunta ou comando em linguagem natural
            
        Returns:
            Resposta da análise
        """
        if not CONTROLLAB_AVAILABLE:
            return {'error': 'ControlLab backend não disponível'}
            
        # Usa o tutor socrático para interpretar a query
        response = self.tutor.process_query(query)
        
        # Se o tutor sugerir código, executa
        if 'suggested_code' in response:
            code_result = self.python_agent.execute_code(response['suggested_code'])
            response['execution_result'] = code_result
            
        return response


class WebSocketAgentServer:
    """
    Servidor WebSocket para comunicação entre frontend e agentes AI.
    
    Permite que o frontend web se comunique com agentes AI que podem
    executar código Python e utilizar o ControlLab.
    """
    
    def __init__(self, host: str = 'localhost', port: int = 8765):
        self.host = host
        self.port = port
        self.agent_interface = ControlLabAgentInterface()
        self.connected_clients = set()
        
    async def handle_client(self, websocket, path):
        """Manipula conexões de clientes WebSocket."""
        self.connected_clients.add(websocket)
        print(f"Cliente conectado: {websocket.remote_address}")
        
        try:
            async for message in websocket:
                await self.process_message(websocket, message)
        except websockets.exceptions.ConnectionClosed:
            print(f"Cliente desconectado: {websocket.remote_address}")
        finally:
            self.connected_clients.discard(websocket)
    
    async def process_message(self, websocket, message: str):
        """Processa mensagens recebidas dos clientes."""
        try:
            data = json.loads(message)
            message_type = data.get('type')
            payload = data.get('payload', {})
            
            response = {'type': 'response', 'success': False}
            
            if message_type == 'analyze_tf':
                result = self.agent_interface.analyze_transfer_function(
                    payload['expression'], 
                    payload.get('analyses', ['stability'])
                )
                response.update({'success': True, 'data': result})
                
            elif message_type == 'design_pid':
                result = self.agent_interface.design_pid_controller(
                    payload['plant'], 
                    payload.get('method', 'ziegler_nichols')
                )
                response.update({'success': True, 'data': result})
                
            elif message_type == 'compare_systems':
                result = self.agent_interface.compare_systems(
                    payload['systems'], 
                    payload.get('analyses', ['stability'])
                )
                response.update({'success': True, 'data': result})
                
            elif message_type == 'interactive_query':
                result = self.agent_interface.interactive_analysis(payload['query'])
                response.update({'success': True, 'data': result})
                
            elif message_type == 'execute_code':
                result = self.agent_interface.python_agent.execute_code(payload['code'])
                response.update({'success': True, 'data': result})
                
            else:
                response['error'] = f'Tipo de mensagem desconhecido: {message_type}'
            
            await websocket.send(json.dumps(response))
            
        except Exception as e:
            error_response = {
                'type': 'error',
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
            await websocket.send(json.dumps(error_response))
    
    async def broadcast(self, message: Dict[str, Any]):
        """Envia mensagem para todos os clientes conectados."""
        if self.connected_clients:
            message_str = json.dumps(message)
            await asyncio.gather(
                *[client.send(message_str) for client in self.connected_clients],
                return_exceptions=True
            )
    
    def start_server(self):
        """Inicia o servidor WebSocket."""
        print(f"Iniciando servidor WebSocket em {self.host}:{self.port}")
        
        start_server = websockets.serve(
            self.handle_client, 
            self.host, 
            self.port
        )
        
        asyncio.get_event_loop().run_until_complete(start_server)
        asyncio.get_event_loop().run_forever()


# Funções de conveniência para integração rápida
def create_agent_interface() -> ControlLabAgentInterface:
    """Cria interface de agente pronta para uso."""
    return ControlLabAgentInterface()

def analyze_system_with_agent(expression: str, analyses: List[str] = None) -> Dict[str, Any]:
    """Função de conveniência para análise rápida de sistemas."""
    interface = create_agent_interface()
    return interface.analyze_transfer_function(expression, analyses)

def design_controller_with_agent(plant: str, method: str = 'ziegler_nichols') -> Dict[str, Any]:
    """Função de conveniência para projeto de controladores."""
    interface = create_agent_interface()
    return interface.design_pid_controller(plant, method)


if __name__ == "__main__":
    # Exemplo de uso direto
    print("🤖 Iniciando ControlLab Agent Integration")
    
    # Testa interface básica
    agent = create_agent_interface()
    
    # Exemplo de análise
    result = agent.analyze_transfer_function("1/(s**2 + 2*s + 1)", ['stability', 'step'])
    print(f"Resultado da análise: {result}")
    
    # Inicia servidor WebSocket se executado diretamente
    if len(sys.argv) > 1 and sys.argv[1] == '--server':
        server = WebSocketAgentServer()
        server.start_server()
