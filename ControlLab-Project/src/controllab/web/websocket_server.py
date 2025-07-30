"""
ControlLab Web - WebSocket Server

Servidor WebSocket para comunicação em tempo real entre frontend e backend,
implementando protocolo de mensagens para descarregamento cognitivo.

Classes implementadas:
- ControlLabWebSocketServer: Servidor principal WebSocket
- MessageProtocol: Protocolo de mensagens estruturado
- ConnectionManager: Gerenciamento de conexões ativas
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
import uuid

# Importar solução de serialização simbólica
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from websocket_symbolic_bridge import WebSocketSymbolicBridge, safe_json_serialize

# WebSocket e servidor assíncrono
try:
    import websockets
    from websockets.server import WebSocketServerProtocol
    import aiohttp
    from aiohttp import web, WSMsgType
    WEBSOCKET_AVAILABLE = True
except ImportError:
    print("⚠️ WebSocket dependencies não disponíveis. Install: pip install websockets aiohttp")
    WEBSOCKET_AVAILABLE = False
    # Mock classes para evitar erros de importação
    class WebSocketServerProtocol:
        pass
    class web:
        pass
    class WSMsgType:
        pass

# Integração com backend ControlLab
try:
    from .analysis_maestro import AnalysisMaestro, AnalysisSession
    from .ai_tutor import SocraticTutor
    BACKEND_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Backend ControlLab não disponível para WebSocket: {e}")
    BACKEND_AVAILABLE = False

# LaTeX Integration
try:
    from ..core.visualization import LaTeXGenerator
    from .latex_websocket_patch import LaTeXResponseEnhancer
    LATEX_INTEGRATION = True
except ImportError:
    print("⚠️ LaTeX integration não disponível")
    LATEX_INTEGRATION = False
    
    # Mock classes
    class LaTeXResponseEnhancer:
        def enhance_response_with_latex(self, response):
            return response

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MessageProtocol:
    """
    Protocolo de mensagens estruturado para comunicação Frontend ↔ Backend.
    
    Implementa padrão de mensagens otimizado para descarregamento cognitivo,
    com tipos específicos para cada etapa do processo de análise.
    """
    
    # Tipos de mensagens Frontend → Backend
    FRONTEND_MESSAGES = {
        "ANALYSIS_REQUEST": "Solicitação de análise técnica",
        "SOCRATIC_QUESTION": "Pergunta para tutor socrático", 
        "STUDENT_ANSWER": "Resposta do estudante",
        "SYSTEM_INPUT": "Entrada de sistema (função transferência)",
        "SESSION_CREATE": "Criar nova sessão",
        "SESSION_STATUS": "Status da sessão atual",
        "NATURAL_LANGUAGE": "Comando em linguagem natural"
    }
    
    # Tipos de mensagens Backend → Frontend
    BACKEND_MESSAGES = {
        "AI_PLAN_UPDATE": "Atualização do plano ReAct",
        "SYMBOLIC_STEP": "Passo de análise simbólica",
        "VISUALIZATION_UPDATE": "Atualização de visualização",
        "SOCRATIC_RESPONSE": "Resposta do tutor socrático",
        "ERROR_MESSAGE": "Mensagem de erro",
        "PROGRESS_UPDATE": "Atualização de progresso",
        "LEARNING_FEEDBACK": "Feedback pedagógico",
        "SESSION_STATE": "Estado da sessão"
    }
    
    @staticmethod
    def create_message(msg_type: str, payload: Dict[str, Any], 
                      session_id: str = None, timestamp: str = None) -> Dict[str, Any]:
        """Cria mensagem estruturada"""
        return {
            "type": msg_type,
            "payload": payload,
            "session_id": session_id,
            "timestamp": timestamp or datetime.now().isoformat(),
            "message_id": str(uuid.uuid4())
        }
        
    @staticmethod
    def validate_message(message: Dict[str, Any]) -> bool:
        """Valida estrutura da mensagem"""
        required_fields = ["type", "payload", "timestamp"]
        return all(field in message for field in required_fields)


class ConnectionManager:
    """
    Gerenciador de conexões WebSocket ativas.
    
    Mantém mapeamento entre conexões, sessões e usuários para
    entrega eficiente de mensagens.
    """
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocketServerProtocol] = {}
        self.session_connections: Dict[str, str] = {}  # session_id -> connection_id
        self.connection_sessions: Dict[str, str] = {}  # connection_id -> session_id
        self.connection_metadata: Dict[str, Dict[str, Any]] = {}
        
    async def connect(self, websocket: WebSocketServerProtocol, connection_id: str = None) -> str:
        """Registra nova conexão WebSocket"""
        if not connection_id:
            connection_id = str(uuid.uuid4())
            
        self.active_connections[connection_id] = websocket
        self.connection_metadata[connection_id] = {
            "connected_at": datetime.now(),
            "last_activity": datetime.now(),
            "message_count": 0
        }
        
        logger.info(f"Nova conexão registrada: {connection_id}")
        return connection_id
        
    async def disconnect(self, connection_id: str):
        """Remove conexão e limpa metadados"""
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
            
        if connection_id in self.connection_metadata:
            del self.connection_metadata[connection_id]
            
        # Limpar mapeamentos de sessão
        if connection_id in self.connection_sessions:
            session_id = self.connection_sessions[connection_id]
            del self.connection_sessions[connection_id]
            if session_id in self.session_connections:
                del self.session_connections[session_id]
                
        logger.info(f"Conexão removida: {connection_id}")
        
    async def associate_session(self, connection_id: str, session_id: str):
        """Associa conexão com sessão de análise"""
        self.session_connections[session_id] = connection_id
        self.connection_sessions[connection_id] = session_id
        
    async def send_to_connection(self, connection_id: str, message: Dict[str, Any]) -> bool:
        """Envia mensagem para conexão específica usando serialização simbólica"""
        if connection_id not in self.active_connections:
            return False
            
        try:
            websocket = self.active_connections[connection_id]
            
            # NOVO: Usar ponte simbólica para preparar mensagem
            safe_message = self.symbolic_bridge.prepare_for_websocket(message)
            message_json = json.dumps(safe_message, ensure_ascii=False)
            
            await websocket.send(message_json)
            
            # Atualizar metadados
            if connection_id in self.connection_metadata:
                self.connection_metadata[connection_id]["last_activity"] = datetime.now()
                self.connection_metadata[connection_id]["message_count"] += 1
                
            return True
        except Exception as e:
            logger.error(f"Erro enviando mensagem para {connection_id}: {e}")
            await self.disconnect(connection_id)
            return False
            
    async def send_to_session(self, session_id: str, message: Dict[str, Any]) -> bool:
        """Envia mensagem para sessão específica"""
        if session_id not in self.session_connections:
            return False
            
        connection_id = self.session_connections[session_id]
        return await self.send_to_connection(connection_id, message)
        
    async def broadcast(self, message: Dict[str, Any]) -> int:
        """Envia mensagem para todas as conexões ativas"""
        sent_count = 0
        
        for connection_id in list(self.active_connections.keys()):
            if await self.send_to_connection(connection_id, message):
                sent_count += 1
                
        return sent_count    async def cleanup_symbolic_cache(self):
        """Limpa cache de objetos simbólicos periodicamente"""
        cache_size = len(self.symbolic_bridge.symbolic_cache)
        if cache_size > 1000:  # Limite arbitrário
            # Manter apenas os 500 mais recentes
            cache_items = list(self.symbolic_bridge.symbolic_cache.items())
            self.symbolic_bridge.symbolic_cache = dict(cache_items[-500:])
            logger.info(f"Cache simbólico limpo: {cache_size} → {len(self.symbolic_bridge.symbolic_cache)}")
    
    async def get_symbolic_object(self, obj_id: str) -> Any:
        """Recupera objeto simbólico por ID"""
        return self.symbolic_bridge.get_symbolic_object(obj_id)
        return sent_count
        
    def get_connection_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas das conexões"""
        return {
            "total_connections": len(self.active_connections),
            "active_sessions": len(self.session_connections),
            "connections_metadata": self.connection_metadata
        }


class ControlLabWebSocketServer:
    """
    Servidor WebSocket principal do ControlLab.
    
    Implementa comunicação em tempo real otimizada para descarregamento cognitivo,
    integrando Analysis Maestro e AI Tutor Socrático.
    """
    
    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.connection_manager = ConnectionManager()
        
        # NOVO: Ponte simbólica para serialização segura
        self.symbolic_bridge = WebSocketSymbolicBridge()
        
        # Integração com backend
        if BACKEND_AVAILABLE:
            self.analysis_maestro = AnalysisMaestro()
            self.socratic_tutor = SocraticTutor()
        else:
            logger.warning("Backend ControlLab não disponível - modo demonstração")
            self.analysis_maestro = None
            self.socratic_tutor = None
            
        # Handlers de mensagens
        self.message_handlers = {
            "ANALYSIS_REQUEST": self._handle_analysis_request,
            "SOCRATIC_QUESTION": self._handle_socratic_question,
            "STUDENT_ANSWER": self._handle_student_answer,
            "SYSTEM_INPUT": self._handle_system_input,
            "SESSION_CREATE": self._handle_session_create,
            "SESSION_STATUS": self._handle_session_status,
            "NATURAL_LANGUAGE": self._handle_natural_language
        }
        
    async def _handle_connection(self, websocket: WebSocketServerProtocol, path: str):
        """Handler principal para novas conexões WebSocket"""
        connection_id = await self.connection_manager.connect(websocket)
        
        # Mensagem de boas-vindas
        welcome_message = MessageProtocol.create_message(
            "SESSION_STATE",
            {
                "status": "connected",
                "connection_id": connection_id,
                "server_capabilities": await self._get_server_capabilities(),
                "available_analyses": await self._get_available_analyses()
            }
        )
        
        await self.connection_manager.send_to_connection(connection_id, welcome_message)
        
        try:
            async for message in websocket:
                await self._process_message(connection_id, message)
                
        except Exception as e:
            logger.error(f"Erro na conexão {connection_id}: {e}")
        finally:
            await self.connection_manager.disconnect(connection_id)
            
    async def _process_message(self, connection_id: str, raw_message: str):
        """Processa mensagem recebida do frontend"""
        try:
            message = json.loads(raw_message)
            
            if not MessageProtocol.validate_message(message):
                await self._send_error(connection_id, "Formato de mensagem inválido")
                return
                
            msg_type = message["type"]
            payload = message["payload"]
            session_id = message.get("session_id")
            
            # Associar sessão se fornecida
            if session_id:
                await self.connection_manager.associate_session(connection_id, session_id)
                
            # Processar mensagem
            if msg_type in self.message_handlers:
                await self.message_handlers[msg_type](connection_id, payload, session_id)
            else:
                await self._send_error(connection_id, f"Tipo de mensagem não suportado: {msg_type}")
                
        except json.JSONDecodeError:
            await self._send_error(connection_id, "JSON inválido")
        except Exception as e:
            logger.error(f"Erro processando mensagem: {e}")
            await self._send_error(connection_id, f"Erro interno: {str(e)}")
            
    async def _handle_analysis_request(self, connection_id: str, payload: Dict[str, Any], session_id: str):
        """Handler para solicitações de análise técnica"""
        if not BACKEND_AVAILABLE:
            await self._send_error(connection_id, "Backend de análise não disponível")
            return
            
        analysis_type = payload.get("analysis_type")
        tf_string = payload.get("tf_string")
        
        if not analysis_type or not tf_string:
            await self._send_error(connection_id, "Parâmetros obrigatórios: analysis_type, tf_string")
            return
            
        try:
            # Enviar plano ReAct primeiro
            if analysis_type == "stability":
                plan = await self.analysis_maestro.create_stability_analysis_plan(tf_string)
                
                plan_message = MessageProtocol.create_message(
                    "AI_PLAN_UPDATE",
                    {
                        "plan": plan.to_dict(),
                        "analysis_type": analysis_type,
                        "system": tf_string
                    },
                    session_id
                )
                
                await self.connection_manager.send_to_connection(connection_id, plan_message)
                
                # Executar análise
                result = await self.analysis_maestro.execute_stability_analysis(session_id or "temp", tf_string)
                
                # LaTeX Enhancement
                if LATEX_INTEGRATION:
                    latex_enhancer = LaTeXResponseEnhancer()
                    result = latex_enhancer.enhance_response_with_latex(result)
                    result['show_latex'] = True
                
                # Enviar resultado final
                result_message = MessageProtocol.create_message(
                    "SYMBOLIC_STEP",
                    {
                        "title": "Análise de Estabilidade Completa",
                        "result": result,
                        "backend_function": "StabilityAnalysisEngine.comprehensive_analysis()"
                    },
                    session_id
                )
                
                await self.connection_manager.send_to_connection(connection_id, result_message)
                
            else:
                await self._send_error(connection_id, f"Tipo de análise não implementado: {analysis_type}")
                
        except Exception as e:
            await self._send_error(connection_id, f"Erro na análise: {str(e)}")
            
    async def _handle_socratic_question(self, connection_id: str, payload: Dict[str, Any], session_id: str):
        """Handler para perguntas socráticas"""
        if not BACKEND_AVAILABLE:
            await self._send_error(connection_id, "Tutor socrático não disponível")
            return
            
        student_id = payload.get("student_id", session_id or "anonymous")
        concept = payload.get("concept")
        context = payload.get("context", "")
        
        if not concept:
            await self._send_error(connection_id, "Conceito obrigatório para pergunta socrática")
            return
            
        try:
            question_result = await self.socratic_tutor.ask_socratic_question(
                student_id, concept, context
            )
            
            response_message = MessageProtocol.create_message(
                "SOCRATIC_RESPONSE",
                {
                    "response_type": "question",
                    "content": question_result
                },
                session_id
            )
            
            await self.connection_manager.send_to_connection(connection_id, response_message)
            
        except Exception as e:
            await self._send_error(connection_id, f"Erro no tutor socrático: {str(e)}")
            
    async def _handle_student_answer(self, connection_id: str, payload: Dict[str, Any], session_id: str):
        """Handler para respostas de estudantes"""
        if not BACKEND_AVAILABLE:
            await self._send_error(connection_id, "Tutor socrático não disponível")
            return
            
        student_id = payload.get("student_id", session_id or "anonymous")
        concept = payload.get("concept")
        answer = payload.get("answer")
        
        if not concept or not answer:
            await self._send_error(connection_id, "Conceito e resposta obrigatórios")
            return
            
        try:
            explanation_result = await self.socratic_tutor.provide_guided_explanation(
                student_id, concept, answer
            )
            
            feedback_message = MessageProtocol.create_message(
                "LEARNING_FEEDBACK",
                {
                    "feedback_type": "explanation",
                    "content": explanation_result
                },
                session_id
            )
            
            await self.connection_manager.send_to_connection(connection_id, feedback_message)
            
        except Exception as e:
            await self._send_error(connection_id, f"Erro processando resposta: {str(e)}")
            
    async def _handle_system_input(self, connection_id: str, payload: Dict[str, Any], session_id: str):
        """Handler para entrada de sistema"""
        tf_string = payload.get("tf_string")
        description = payload.get("description", "")
        
        if not tf_string:
            await self._send_error(connection_id, "Função de transferência obrigatória")
            return
            
        # Validar e parsear sistema
        try:
            # Aqui seria feita validação da função de transferência
            validation_result = {
                "valid": True,
                "tf_string": tf_string,
                "description": description,
                "parsed_at": datetime.now().isoformat()
            }
            
            # LaTeX Enhancement for system input
            if LATEX_INTEGRATION:
                latex_enhancer = LaTeXResponseEnhancer()
                validation_result = latex_enhancer.enhance_response_with_latex(validation_result)
                validation_result['show_latex'] = True
            
            response_message = MessageProtocol.create_message(
                "SESSION_STATE",
                {
                    "status": "system_updated",
                    "current_system": validation_result
                },
                session_id
            )
            
            await self.connection_manager.send_to_connection(connection_id, response_message)
            
        except Exception as e:
            await self._send_error(connection_id, f"Erro validando sistema: {str(e)}")
            
    async def _handle_session_create(self, connection_id: str, payload: Dict[str, Any], session_id: str):
        """Handler para criação de sessão"""
        new_session_id = payload.get("session_id", str(uuid.uuid4()))
        
        try:
            if BACKEND_AVAILABLE:
                session = await self.analysis_maestro.create_session(new_session_id)
                
                session_info = {
                    "session_id": new_session_id,
                    "created_at": session.created_at.isoformat(),
                    "status": "active"
                }
            else:
                session_info = {
                    "session_id": new_session_id,
                    "created_at": datetime.now().isoformat(),
                    "status": "demo_mode",
                    "note": "Backend não disponível"
                }
                
            await self.connection_manager.associate_session(connection_id, new_session_id)
            
            response_message = MessageProtocol.create_message(
                "SESSION_STATE",
                {
                    "status": "created",
                    "session_info": session_info
                },
                new_session_id
            )
            
            await self.connection_manager.send_to_connection(connection_id, response_message)
            
        except Exception as e:
            await self._send_error(connection_id, f"Erro criando sessão: {str(e)}")
            
    async def _handle_session_status(self, connection_id: str, payload: Dict[str, Any], session_id: str):
        """Handler para status da sessão"""
        try:
            if BACKEND_AVAILABLE and session_id:
                session = await self.analysis_maestro.get_session(session_id)
                if session:
                    status_info = session.get_session_summary()
                else:
                    status_info = {"error": "Sessão não encontrada"}
            else:
                status_info = {
                    "session_id": session_id,
                    "status": "demo_mode",
                    "backend_available": BACKEND_AVAILABLE
                }
                
            response_message = MessageProtocol.create_message(
                "SESSION_STATE",
                {
                    "status": "current_status",
                    "session_info": status_info
                },
                session_id
            )
            
            await self.connection_manager.send_to_connection(connection_id, response_message)
            
        except Exception as e:
            await self._send_error(connection_id, f"Erro obtendo status: {str(e)}")
            
    async def _handle_natural_language(self, connection_id: str, payload: Dict[str, Any], session_id: str):
        """Handler para comandos em linguagem natural"""
        if not BACKEND_AVAILABLE:
            await self._send_error(connection_id, "Processamento de linguagem natural não disponível")
            return
            
        command = payload.get("command")
        if not command:
            await self._send_error(connection_id, "Comando obrigatório")
            return
            
        try:
            result = await self.analysis_maestro.handle_natural_language_command(
                session_id or "temp", command
            )
            
            # LaTeX Enhancement for natural language results
            if LATEX_INTEGRATION:
                latex_enhancer = LaTeXResponseEnhancer()
                result = latex_enhancer.enhance_response_with_latex(result)
                result['show_latex'] = True
            
            response_message = MessageProtocol.create_message(
                "AI_PLAN_UPDATE",
                {
                    "command_processed": command,
                    "result": result
                },
                session_id
            )
            
            await self.connection_manager.send_to_connection(connection_id, response_message)
            
        except Exception as e:
            await self._send_error(connection_id, f"Erro processando comando: {str(e)}")
            
    async def _send_error(self, connection_id: str, error_message: str):
        """Envia mensagem de erro para o frontend"""
        error_msg = MessageProtocol.create_message(
            "ERROR_MESSAGE",
            {
                "error": error_message,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        await self.connection_manager.send_to_connection(connection_id, error_msg)
        
    async def _get_server_capabilities(self) -> Dict[str, Any]:
        """Retorna capacidades do servidor"""
        return {
            "backend_available": BACKEND_AVAILABLE,
            "websocket_available": WEBSOCKET_AVAILABLE,
            "supported_analyses": ["stability"] if BACKEND_AVAILABLE else [],
            "socratic_tutor": BACKEND_AVAILABLE,
            "version": "1.0.0"
        }
        
    async def _get_available_analyses(self) -> Dict[str, Any]:
        """Retorna análises disponíveis"""
        if BACKEND_AVAILABLE:
            return await self.analysis_maestro.get_available_analyses()
        else:
            return {
                "analyses": {},
                "backend_status": {
                    "controllab_available": False,
                    "message": "Backend em modo demonstração"
                }
            }
            
    async def start_server(self):
        """Inicia o servidor WebSocket"""
        if not WEBSOCKET_AVAILABLE:
            logger.error("Dependências WebSocket não disponíveis")
            return
            
        logger.info(f"Iniciando ControlLab WebSocket Server em {self.host}:{self.port}")
        
        try:
            server = await websockets.serve(
                self._handle_connection,
                self.host,
                self.port,
                ping_interval=30,
                ping_timeout=10
            )
            
            logger.info(f"✅ Servidor iniciado com sucesso!")
            logger.info(f"WebSocket URL: ws://{self.host}:{self.port}")
            
            # Manter servidor rodando
            await server.wait_closed()
            
        except Exception as e:
            logger.error(f"Erro iniciando servidor: {e}")
            
    def get_server_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do servidor"""
        return {
            "server": {
                "host": self.host,
                "port": self.port,
                "backend_available": BACKEND_AVAILABLE,
                "websocket_available": WEBSOCKET_AVAILABLE
            },
            "connections": self.connection_manager.get_connection_stats()
        }


# Funções de conveniência
def create_server(host: str = "localhost", port: int = 8765) -> ControlLabWebSocketServer:
    """Cria instância do servidor WebSocket"""
    return ControlLabWebSocketServer(host, port)


async def run_server(host: str = "localhost", port: int = 8765):
    """Executa servidor WebSocket"""
    server = create_server(host, port)
    await server.start_server()


# Exemplo de uso e teste
if __name__ == "__main__":
    async def demo():
        """Demonstração do WebSocket Server"""
        print("🌐 ControlLab WebSocket Server Demo")
        print("=" * 50)
        
        # Criar servidor
        server = create_server()
        
        # Mostrar estatísticas
        stats = server.get_server_stats()
        print(f"Configuração do Servidor:")
        print(json.dumps(stats, indent=2))
        
        print(f"\n📡 Para conectar cliente WebSocket:")
        print(f"URL: ws://{server.host}:{server.port}")
        
        if WEBSOCKET_AVAILABLE:
            print(f"\n🚀 Iniciando servidor...")
            print(f"Pressione Ctrl+C para parar")
            try:
                await server.start_server()
            except KeyboardInterrupt:
                print(f"\n🛑 Servidor parado pelo usuário")
        else:
            print(f"\n⚠️ WebSocket não disponível - instale dependências")
            
    # Executar demo
    asyncio.run(demo())
