"""
ControlLab Advanced Agent - Abstração SymboLab-like
Sistema inteligente de tutoria com reconhecimento automático de expressões
e análise interativa similar ao SymboLab.
"""

import json
import re
import sympy as sp
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import asyncio
import websockets

# Importações do ControlLab
try:
    from ..analysis.stability_analysis import analyze_stability, quick_stability_check
    from ..core.symbolic_tf import SymbolicTransferFunction
    from ..design.pid_tuning import PIDTuner
    from ..modeling.laplace_transform import LaplaceTransformer
    from ..numerical.interface import NumericalInterface
    from .function_registry import FUNCTION_REGISTRY, find_functions_for_task, generate_code_for_task
    CONTROLLAB_AVAILABLE = True
except ImportError:
    CONTROLLAB_AVAILABLE = False


class ExpressionType(Enum):
    """Tipos de expressões reconhecidos automaticamente."""
    TRANSFER_FUNCTION = "transfer_function"
    POLYNOMIAL = "polynomial"
    DIFFERENTIAL_EQUATION = "differential_equation"
    STEP_RESPONSE = "step_response"
    BODE_PLOT = "bode_plot"
    ROOT_LOCUS = "root_locus"
    PID_DESIGN = "pid_design"
    STABILITY_ANALYSIS = "stability_analysis"
    UNKNOWN = "unknown"


@dataclass
class MathExpression:
    """Representação de uma expressão matemática reconhecida."""
    raw_input: str
    expression_type: ExpressionType
    parsed_expression: Any
    variables: List[str]
    parameters: Dict[str, Any]
    confidence: float


class IntelligentExpressionParser:
    """
    Parser inteligente de expressões matemáticas para sistemas de controle.
    Similar ao reconhecimento automático do SymboLab.
    """
    
    def __init__(self):
        self.patterns = self._build_patterns()
        self.sympy_parser = sp.sympify
        
    def _build_patterns(self) -> Dict[ExpressionType, List[str]]:
        """Constrói padrões de reconhecimento para diferentes tipos de expressão."""
        return {
            ExpressionType.TRANSFER_FUNCTION: [
                r'G\(s\)\s*=\s*(.+)',
                r'H\(s\)\s*=\s*(.+)',
                r'([0-9\s\+\-\*\/\(\)s\^]+)\s*\/\s*([0-9\s\+\-\*\/\(\)s\^]+)',
                r'(\d+(?:\.\d+)?)\s*\/\s*\(([^)]+)\)',
                r'transfer[_\s]?function',
            ],
            ExpressionType.POLYNOMIAL: [
                r's\^?\d+',
                r'[+-]?\d*s[\^0-9]*',
                r'polynomial',
                r'characteristic',
            ],
            ExpressionType.STABILITY_ANALYSIS: [
                r'stable|stability|estabil',
                r'routh|hurwitz',
                r'pole|zero|root',
                r'margin|margem',
            ],
            ExpressionType.PID_DESIGN: [
                r'pid|PID',
                r'controller|controlador',
                r'tune|tuning|sintonia',
                r'kp|ki|kd',
            ],
            ExpressionType.BODE_PLOT: [
                r'bode|Bode',
                r'frequency|frequencia',
                r'magnitude|phase|fase',
            ],
            ExpressionType.ROOT_LOCUS: [
                r'root\s+locus|lugar\s+das\s+raizes',
                r'evans',
                r'gain|ganho',
            ]
        }
    
    def parse_expression(self, user_input: str) -> MathExpression:
        """
        Analisa entrada do usuário e identifica tipo de expressão.
        
        Args:
            user_input: Entrada do usuário (texto, expressão matemática, comando)
            
        Returns:
            MathExpression com análise completa da entrada
        """
        # Normaliza entrada
        normalized_input = self._normalize_input(user_input)
        
        # Detecta tipo de expressão
        expression_type = self._detect_expression_type(normalized_input)
        
        # Parse específico baseado no tipo
        parsed_expr, variables, parameters, confidence = self._parse_by_type(
            normalized_input, expression_type
        )
        
        return MathExpression(
            raw_input=user_input,
            expression_type=expression_type,
            parsed_expression=parsed_expr,
            variables=variables,
            parameters=parameters,
            confidence=confidence
        )
    
    def _normalize_input(self, text: str) -> str:
        """Normaliza entrada para facilitar parsing."""
        # Remove espaços extras
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Normaliza notação matemática
        text = text.replace('^', '**')  # Potência
        text = text.replace('G(s)', 'G_s')  # Função transferência
        text = text.replace('H(s)', 'H_s')
        
        return text
    
    def _detect_expression_type(self, text: str) -> ExpressionType:
        """Detecta o tipo de expressão baseado em padrões."""
        text_lower = text.lower()
        
        # Verifica cada tipo de expressão
        for expr_type, patterns in self.patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return expr_type
        
        # Se contém s, provavelmente é função transferência
        if 's' in text_lower and any(op in text for op in ['/', '+', '-', '*']):
            return ExpressionType.TRANSFER_FUNCTION
            
        return ExpressionType.UNKNOWN
    
    def _parse_by_type(self, text: str, expr_type: ExpressionType) -> Tuple[Any, List[str], Dict[str, Any], float]:
        """Parse específico baseado no tipo de expressão."""
        
        if expr_type == ExpressionType.TRANSFER_FUNCTION:
            return self._parse_transfer_function(text)
        elif expr_type == ExpressionType.STABILITY_ANALYSIS:
            return self._parse_stability_request(text)
        elif expr_type == ExpressionType.PID_DESIGN:
            return self._parse_pid_request(text)
        else:
            return self._parse_generic(text)
    
    def _parse_transfer_function(self, text: str) -> Tuple[Any, List[str], Dict[str, Any], float]:
        """Parse específico para funções de transferência."""
        try:
            # Remove prefixos como G(s) =
            text = re.sub(r'[GH]\(s\)\s*=\s*', '', text)
            
            # Procura padrão numerador/denominador
            if '/' in text:
                parts = text.split('/')
                if len(parts) == 2:
                    num_str = parts[0].strip()
                    den_str = parts[1].strip()
                    
                    # Remove parênteses externos se presentes
                    den_str = den_str.strip('()')
                    
                    # Converte para SymPy
                    s = sp.Symbol('s')
                    numerator = sp.sympify(num_str, locals={'s': s})
                    denominator = sp.sympify(den_str, locals={'s': s})
                    
                    # Extrai coeficientes
                    num_coeffs = sp.Poly(numerator, s).all_coeffs() if numerator != 0 else [0]
                    den_coeffs = sp.Poly(denominator, s).all_coeffs()
                    
                    tf_data = {
                        'numerator': [float(c) for c in num_coeffs],
                        'denominator': [float(c) for c in den_coeffs],
                        'symbolic_num': numerator,
                        'symbolic_den': denominator
                    }
                    
                    return tf_data, ['s'], {'type': 'transfer_function'}, 0.9
            
            # Fallback para expressão simples
            s = sp.Symbol('s')
            expr = sp.sympify(text, locals={'s': s})
            return {'expression': expr}, ['s'], {'type': 'expression'}, 0.7
            
        except Exception as e:
            return {'error': str(e), 'raw': text}, [], {}, 0.1
    
    def _parse_stability_request(self, text: str) -> Tuple[Any, List[str], Dict[str, Any], float]:
        """Parse para solicitações de análise de estabilidade."""
        return {
            'analysis_type': 'stability',
            'request': text,
            'methods': ['routh_hurwitz', 'pole_analysis', 'frequency_analysis']
        }, [], {'type': 'analysis_request'}, 0.8
    
    def _parse_pid_request(self, text: str) -> Tuple[Any, List[str], Dict[str, Any], float]:
        """Parse para solicitações de projeto PID."""
        return {
            'analysis_type': 'pid_design',
            'request': text,
            'methods': ['ziegler_nichols', 'pole_placement', 'optimization']
        }, [], {'type': 'design_request'}, 0.8
    
    def _parse_generic(self, text: str) -> Tuple[Any, List[str], Dict[str, Any], float]:
        """Parse genérico para outros tipos."""
        return {'text': text, 'type': 'generic'}, [], {}, 0.5


class ControlLabTutor:
    """
    Tutor inteligente similar ao SymboLab para sistemas de controle.
    Fornece explicações passo a passo e análise automática.
    """
    
    def __init__(self):
        self.parser = IntelligentExpressionParser()
        self.conversation_history = []
        self.current_system = None
        self.function_registry = FUNCTION_REGISTRY if CONTROLLAB_AVAILABLE else None
        
    def process_input(self, user_input: str) -> Dict[str, Any]:
        """
        Processa entrada do usuário e retorna resposta estruturada.
        
        Args:
            user_input: Entrada do usuário
            
        Returns:
            Resposta estruturada com análise, explicações e visualizações
        """
        # Parse da entrada
        math_expr = self.parser.parse_expression(user_input)
        
        # Gera resposta baseada no tipo
        response = self._generate_response(math_expr)
        
        # Adiciona ao histórico
        self.conversation_history.append({
            'input': user_input,
            'parsed': math_expr,
            'response': response,
            'timestamp': sp.sympify('now()')  # Placeholder
        })
        
        return response
    
    def _generate_response(self, math_expr: MathExpression) -> Dict[str, Any]:
        """Gera resposta estruturada baseada na expressão analisada."""
        
        if math_expr.expression_type == ExpressionType.TRANSFER_FUNCTION:
            return self._handle_transfer_function(math_expr)
        elif math_expr.expression_type == ExpressionType.STABILITY_ANALYSIS:
            return self._handle_stability_analysis(math_expr)
        elif math_expr.expression_type == ExpressionType.PID_DESIGN:
            return self._handle_pid_design(math_expr)
        else:
            return self._handle_general_question(math_expr)
    
    def _handle_transfer_function(self, math_expr: MathExpression) -> Dict[str, Any]:
        """Manipula entrada de função de transferência."""
        try:
            tf_data = math_expr.parsed_expression
            
            if 'numerator' in tf_data and 'denominator' in tf_data:
                # Cria função de transferência
                if CONTROLLAB_AVAILABLE:
                    tf = SymbolicTransferFunction(
                        tf_data['numerator'], 
                        tf_data['denominator']
                    )
                    self.current_system = tf
                    
                    # Análise automática básica
                    poles = tf.poles()
                    zeros = tf.zeros()
                    is_stable = tf.is_stable()
                    
                    response = {
                        'type': 'transfer_function_analysis',
                        'system': {
                            'numerator': tf_data['numerator'],
                            'denominator': tf_data['denominator'],
                            'poles': [complex(p) for p in poles],
                            'zeros': [complex(z) for z in zeros],
                            'stable': is_stable
                        },
                        'message': self._generate_tf_explanation(tf, poles, zeros, is_stable),
                        'suggested_actions': [
                            {'label': 'Analisar Estabilidade', 'action': 'stability_analysis'},
                            {'label': 'Resposta ao Degrau', 'action': 'step_response'},
                            {'label': 'Diagrama de Bode', 'action': 'bode_plot'},
                            {'label': 'Projetar PID', 'action': 'pid_design'}
                        ],
                        'visualizations': [
                            {'type': 'pole_zero_map', 'data': {'poles': poles, 'zeros': zeros}},
                            {'type': 'step_response', 'data': 'auto_generate'}
                        ]
                    }
                else:
                    response = {
                        'type': 'transfer_function_recognized',
                        'message': f"Reconheci a função de transferência: {tf_data['symbolic_num']}/{tf_data['symbolic_den']}",
                        'error': 'ControlLab backend não disponível'
                    }
                    
                return response
                
        except Exception as e:
            return {
                'type': 'error',
                'message': f"Erro ao analisar função de transferência: {str(e)}",
                'suggestion': 'Verifique a sintaxe. Exemplo: 1/(s^2+2*s+1)'
            }
    
    def _generate_tf_explanation(self, tf, poles, zeros, is_stable) -> str:
        """Gera explicação pedagógica da função de transferência."""
        explanation = f"""
📊 **Análise da Função de Transferência**

🔍 **Sistema Identificado:**
- Ordem: {tf.order()}
- Polos: {len(poles)} | Zeros: {len(zeros)}
- Estabilidade: {'ESTAVEL' if is_stable else 'INSTAVEL'}

INTERPRETACAO FISICA:
- Os polos determinam a resposta natural do sistema
- {"Todos os polos estão no semiplano esquerdo" if is_stable else "Há polos no semiplano direito"}
- {"Sistema convergirá para valor finito" if is_stable else "Sistema pode divergir"}

PROXIMOS PASSOS SUGERIDOS:
1. Analisar resposta temporal (degrau, impulso)
2. Verificar margens de estabilidade (Bode)
3. Projetar controlador se necessário
        """
        return explanation.strip()
    
    def _handle_stability_analysis(self, math_expr: MathExpression) -> Dict[str, Any]:
        """Manipula solicitações de análise de estabilidade."""
        if self.current_system is None:
            return {
                'type': 'request_system',
                'message': 'Para analisar estabilidade, primeiro preciso de uma função de transferência. Por favor, forneça G(s) = ...',
                'example': 'Exemplo: G(s) = 1/(s^2+2*s+1)'
            }
        
        if CONTROLLAB_AVAILABLE:
            try:
                # Realiza análise completa
                stability_report = analyze_stability(self.current_system, show_steps=True)
                
                return {
                    'type': 'stability_analysis_complete',
                    'message': 'Análise de estabilidade concluída com múltiplos métodos:',
                    'analysis': {
                        'routh_hurwitz': 'Critério algébrico aplicado',
                        'pole_analysis': 'Localização de polos verificada',
                        'frequency_analysis': 'Margens calculadas'
                    },
                    'conclusion': stability_report.get_executive_summary(),
                    'detailed_steps': stability_report.get_detailed_analysis(),
                    'visualizations': [
                        {'type': 'routh_table', 'data': 'auto_generate'},
                        {'type': 'pole_zero_map', 'data': 'auto_generate'},
                        {'type': 'nyquist_plot', 'data': 'auto_generate'}
                    ]
                }
            except Exception as e:
                return {
                    'type': 'error',
                    'message': f"Erro na análise: {str(e)}"
                }
        else:
            return {
                'type': 'backend_unavailable',
                'message': 'Backend ControlLab não disponível para análise detalhada'
            }
    
    def _handle_pid_design(self, math_expr: MathExpression) -> Dict[str, Any]:
        """Manipula solicitações de projeto PID."""
        if self.current_system is None:
            return {
                'type': 'request_system',
                'message': 'Para projetar controlador PID, preciso da planta G(s). Forneça a função de transferência.',
                'example': 'Exemplo: G(s) = 1/(s*(s+1))'
            }
        
        return {
            'type': 'pid_design_options',
            'message': 'Métodos de sintonia PID disponíveis:',
            'methods': [
                {'name': 'Ziegler-Nichols', 'description': 'Método clássico baseado em resposta ao degrau'},
                {'name': 'Posicionamento de Polos', 'description': 'Especificação direta de polos desejados'},
                {'name': 'Otimização', 'description': 'Minimização de critério de performance'}
            ],
            'suggested_actions': [
                {'label': 'Ziegler-Nichols', 'action': 'zn_tuning'},
                {'label': 'Especificar Polos', 'action': 'pole_placement'},
                {'label': 'Otimização', 'action': 'optimization_tuning'}
            ]
        }
    
    def _handle_general_question(self, math_expr: MathExpression) -> Dict[str, Any]:
        """Manipula perguntas gerais."""
        
        # Se temos o registry, tenta encontrar funções relevantes
        if self.function_registry and math_expr.raw_input:
            relevant_functions = find_functions_for_task(math_expr.raw_input)
            
            if relevant_functions:
                # Encontrou funções relevantes
                func_suggestions = []
                for func in relevant_functions[:3]:  # Top 3
                    func_suggestions.append({
                        'name': func.name,
                        'purpose': func.purpose,
                        'module': func.module
                    })
                
                return {
                    'type': 'function_suggestions',
                    'message': f"Encontrei {len(relevant_functions)} função(ões) que pode(m) ajudar:",
                    'functions': func_suggestions,
                    'suggested_actions': [f.name for f in relevant_functions[:3]],
                    'educational_context': self.function_registry.get_educational_guidance(relevant_functions[0].name) if relevant_functions else None
                }
        
        return {
            'type': 'general_response',
            'message': f"""
Olá! Sou seu tutor de Engenharia de Controle. Posso ajudar você com:

🔍 **Análise de Sistemas:**
- Funções de transferência: G(s) = num/den
- Estabilidade: Routh-Hurwitz, polos, margens
- Resposta temporal: degrau, impulso, rampa

🎛️ **Projeto de Controladores:**
- PID: Ziegler-Nichols, otimização
- Compensadores: Lead, Lag, Lead-Lag
- Posicionamento de polos

📊 **Visualizações:**
- Diagramas de Bode
- Lugar das raízes
- Resposta temporal

🎓 **Funções Disponíveis:**
{self._get_available_functions_summary() if self.function_registry else 'Registry não disponível'}

💬 **Como posso ajudar?**
Digite uma função de transferência ou faça uma pergunta sobre controle!

Exemplos:
- "G(s) = 1/(s^2+2*s+1)"
- "Analise a estabilidade"
- "Projete um PID"
            """,
            'suggested_inputs': [
                'G(s) = 1/(s+1)',
                'Analise estabilidade',
                'Projete controlador PID'
            ]
        }
    
    def _get_available_functions_summary(self) -> str:
        """Retorna resumo das funções disponíveis."""
        if not self.function_registry:
            return "Registry não disponível"
        
        categories = self.function_registry.categories
        summary = ""
        for category, functions in categories.items():
            summary += f"• {category}: {len(functions)} funções\n"
        
        return summary.strip()


class AdvancedWebSocketServer:
    """
    Servidor WebSocket avançado para comunicação com frontend SymboLab-like.
    """
    
    def __init__(self, host='localhost', port=8765):
        self.host = host
        self.port = port
        self.tutor = ControlLabTutor()
        self.connected_clients = set()
        
    async def handle_client(self, websocket, path):
        """Manipula conexão de cliente WebSocket."""
        self.connected_clients.add(websocket)
        print(f"Cliente conectado: {websocket.remote_address}")
        
        try:
            # Envia mensagem de boas-vindas
            welcome_message = {
                'type': 'welcome',
                'message': 'Conectado ao ControlLab AI Tutor',
                'capabilities': [
                    'Análise de funções de transferência',
                    'Análise de estabilidade',
                    'Projeto de controladores',
                    'Visualizações interativas'
                ]
            }
            await websocket.send(json.dumps(welcome_message))
            
            # Loop principal de comunicação
            async for message in websocket:
                try:
                    data = json.loads(message)
                    response = await self.process_message(data)
                    await websocket.send(json.dumps(response))
                except json.JSONDecodeError:
                    error_response = {
                        'type': 'error',
                        'message': 'Formato de mensagem inválido'
                    }
                    await websocket.send(json.dumps(error_response))
                except Exception as e:
                    error_response = {
                        'type': 'error',
                        'message': f'Erro no processamento: {str(e)}'
                    }
                    await websocket.send(json.dumps(error_response))
                    
        except websockets.exceptions.ConnectionClosed:
            print(f"Cliente desconectado: {websocket.remote_address}")
        finally:
            self.connected_clients.remove(websocket)
    
    async def process_message(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Processa mensagem do cliente."""
        message_type = data.get('type', 'unknown')
        
        if message_type == 'user_input':
            user_input = data.get('content', '')
            response = self.tutor.process_input(user_input)
            return response
        elif message_type == 'action':
            action = data.get('action', '')
            return await self.handle_action(action, data)
        else:
            return {
                'type': 'error',
                'message': f'Tipo de mensagem desconhecido: {message_type}'
            }
    
    async def handle_action(self, action: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Manipula ações específicas do usuário."""
        if action == 'stability_analysis':
            return self.tutor._handle_stability_analysis(None)
        elif action == 'step_response':
            return {'type': 'visualization', 'plot_type': 'step_response'}
        elif action == 'bode_plot':
            return {'type': 'visualization', 'plot_type': 'bode'}
        elif action == 'pid_design':
            return self.tutor._handle_pid_design(None)
        else:
            return {'type': 'error', 'message': f'Ação desconhecida: {action}'}
    
    def start_server(self):
        """Inicia servidor WebSocket."""
        print(f"INICIANDO ControlLab Advanced Agent Server")
        print(f"WebSocket: ws://{self.host}:{self.port}")
        
        async def run_server():
            async with websockets.serve(self.handle_client, self.host, self.port):
                print(f"Servidor WebSocket ativo em ws://{self.host}:{self.port}")
                await asyncio.Future()  # Mantém rodando para sempre
        
        asyncio.run(run_server())


if __name__ == "__main__":
    # Primeiro ativa ambiente virtual
    import sys
    from pathlib import Path
    
    project_root = Path(__file__).parent.parent.parent.parent
    venv_activate = project_root / "venv" / "Scripts" / "Activate.ps1"
    
    if not CONTROLLAB_AVAILABLE:
        print("WARNING: ControlLab backend nao disponivel")
        print(f"DICA: Ative o ambiente virtual: {venv_activate}")
        print("DICA: Execute: .\\venv\\Scripts\\Activate.ps1")
    
    # Inicia servidor
    server = AdvancedWebSocketServer()
    server.start_server()
