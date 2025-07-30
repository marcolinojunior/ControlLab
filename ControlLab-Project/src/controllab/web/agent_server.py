#!/usr/bin/env python3
"""
ControlLab Agent Server - Módulo 8 Completo
Servidor integrado para agentes AI com execução Python e ControlLab

Usage:
    python agent_server.py [--port 8765] [--host localhost] [--debug]
"""

import asyncio
import websockets
import json
import sys
import argparse
import logging
from pathlib import Path
import webbrowser
import time
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
import os

# Importa a integração de agentes
try:
    from .agent_integration import WebSocketAgentServer, ControlLabAgentInterface
    from ..core.symbolic_tf import SymbolicTransferFunction
    AGENT_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Importação de agentes falhou: {e}")
    print("Executando em modo standalone...")
    AGENT_AVAILABLE = False
    
    # Classes básicas para modo standalone
    class WebSocketAgentServer:
        def __init__(self, host='localhost', port=8765):
            self.host = host
            self.port = port
            
        def start_server(self):
            print(f"Servidor básico iniciado em {self.host}:{self.port}")
            asyncio.get_event_loop().run_forever()


class ControlLabAgentApp:
    """
    Aplicação principal do ControlLab Agent.
    
    Integra servidor WebSocket, servidor HTTP para frontend,
    e interface de agentes AI.
    """
    
    def __init__(self, ws_port=8765, http_port=8080, host='localhost', debug=False):
        self.ws_port = ws_port
        self.http_port = http_port
        self.host = host
        self.debug = debug
        
        # Configura logging
        log_level = logging.DEBUG if debug else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('ControlLabAgent')
        
        # Componentes da aplicação
        self.ws_server = None
        self.http_server = None
        self.agent_interface = None
        
        # Status
        self.running = False
        
    def setup_components(self):
        """Configura componentes da aplicação."""
        try:
            # Servidor WebSocket para agentes
            if AGENT_AVAILABLE:
                self.ws_server = WebSocketAgentServer(self.host, self.ws_port)
                self.agent_interface = ControlLabAgentInterface()
                self.logger.info("✅ Componentes de agente configurados")
            else:
                self.ws_server = WebSocketAgentServer(self.host, self.ws_port)
                self.logger.warning("⚠️ Modo básico - funcionalidades limitadas")
                
        except Exception as e:
            self.logger.error(f"❌ Erro ao configurar componentes: {e}")
            raise
    
    def start_http_server(self):
        """Inicia servidor HTTP para frontend."""
        try:
            # Muda para diretório do frontend
            frontend_dir = Path(__file__).parent / 'frontend'
            os.chdir(frontend_dir)
            
            # Cria servidor HTTP
            handler = SimpleHTTPRequestHandler
            self.http_server = HTTPServer((self.host, self.http_port), handler)
            
            self.logger.info(f"🌐 Servidor HTTP iniciado em http://{self.host}:{self.http_port}")
            
            # Executa em thread separada
            http_thread = threading.Thread(
                target=self.http_server.serve_forever,
                daemon=True
            )
            http_thread.start()
            
        except Exception as e:
            self.logger.error(f"❌ Erro ao iniciar servidor HTTP: {e}")
            raise
    
    def start_websocket_server(self):
        """Inicia servidor WebSocket."""
        try:
            self.logger.info(f"🔌 Iniciando servidor WebSocket em ws://{self.host}:{self.ws_port}")
            
            # Usa a implementação disponível
            start_server = websockets.serve(
                self.ws_server.handle_client if hasattr(self.ws_server, 'handle_client') else self._basic_ws_handler,
                self.host,
                self.ws_port
            )
            
            return start_server
            
        except Exception as e:
            self.logger.error(f"❌ Erro ao configurar WebSocket: {e}")
            raise
    
    async def _basic_ws_handler(self, websocket, path):
        """Handler WebSocket básico para modo standalone."""
        self.logger.info(f"Cliente conectado: {websocket.remote_address}")
        
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    response = {
                        'type': 'error',
                        'success': False,
                        'error': 'ControlLab backend não disponível. Execute: pip install sympy control'
                    }
                    await websocket.send(json.dumps(response))
                except Exception as e:
                    error_response = {
                        'type': 'error',
                        'success': False,
                        'error': str(e)
                    }
                    await websocket.send(json.dumps(error_response))
                    
        except websockets.exceptions.ConnectionClosed:
            self.logger.info(f"Cliente desconectado: {websocket.remote_address}")
    
    def print_startup_info(self):
        """Imprime informações de inicialização."""
        print("\n" + "="*60)
        print("🤖 CONTROLLAB AGENT SERVER - MÓDULO 8 COMPLETO")
        print("="*60)
        print(f"🌐 Interface Web: http://{self.host}:{self.http_port}/agent_interface.html")
        print(f"🔌 WebSocket API: ws://{self.host}:{self.ws_port}")
        print(f"📊 Status Backend: {'✅ Completo' if AGENT_AVAILABLE else '⚠️ Limitado'}")
        print("="*60)
        
        if AGENT_AVAILABLE:
            print("🎯 FUNCIONALIDADES DISPONÍVEIS:")
            print("  • Análise de sistemas de controle")
            print("  • Projeto de controladores PID")
            print("  • Execução de código Python")
            print("  • Chat com IA Socrática")
            print("  • Visualizações interativas")
        else:
            print("⚠️ MODO LIMITADO:")
            print("  • Interface web básica disponível")
            print("  • Para funcionalidades completas, instale:")
            print("    pip install sympy control matplotlib")
            
        print("="*60)
        print("💡 COMANDOS ÚTEIS:")
        print("  • Ctrl+C para parar o servidor")
        print("  • Abra o navegador no endereço acima")
        print("  • Use o chat para interagir com a IA")
        print("="*60)
    
    def open_browser(self):
        """Abre navegador automaticamente."""
        url = f"http://{self.host}:{self.http_port}/agent_interface.html"
        try:
            webbrowser.open(url)
            self.logger.info(f"🌐 Navegador aberto em {url}")
        except Exception as e:
            self.logger.warning(f"⚠️ Não foi possível abrir navegador: {e}")
    
    def run(self, open_browser=True):
        """Executa a aplicação completa."""
        try:
            self.logger.info("🚀 Iniciando ControlLab Agent App...")
            
            # Configura componentes
            self.setup_components()
            
            # Inicia servidor HTTP
            self.start_http_server()
            
            # Aguarda um momento para estabilizar
            time.sleep(1)
            
            # Imprime informações
            self.print_startup_info()
            
            # Abre navegador
            if open_browser:
                threading.Timer(2.0, self.open_browser).start()
            
            # Inicia servidor WebSocket (bloqueia)
            start_server = self.start_websocket_server()
            
            # Event loop principal
            self.running = True
            loop = asyncio.get_event_loop()
            loop.run_until_complete(start_server)
            loop.run_forever()
            
        except KeyboardInterrupt:
            self.logger.info("🛑 Parando servidor...")
            self.stop()
        except Exception as e:
            self.logger.error(f"❌ Erro fatal: {e}")
            raise
    
    def stop(self):
        """Para a aplicação."""
        self.running = False
        
        if self.http_server:
            self.http_server.shutdown()
            self.logger.info("🌐 Servidor HTTP parado")
            
        self.logger.info("✅ ControlLab Agent App parado")


def main():
    """Função principal."""
    parser = argparse.ArgumentParser(
        description='ControlLab Agent Server - Sistema integrado de agentes AI'
    )
    parser.add_argument('--ws-port', type=int, default=8765, 
                       help='Porta do servidor WebSocket (padrão: 8765)')
    parser.add_argument('--http-port', type=int, default=8080,
                       help='Porta do servidor HTTP (padrão: 8080)')
    parser.add_argument('--host', default='localhost',
                       help='Host do servidor (padrão: localhost)')
    parser.add_argument('--debug', action='store_true',
                       help='Ativa modo debug')
    parser.add_argument('--no-browser', action='store_true',
                       help='Não abre navegador automaticamente')
    
    args = parser.parse_args()
    
    # Cria e executa aplicação
    app = ControlLabAgentApp(
        ws_port=args.ws_port,
        http_port=args.http_port,
        host=args.host,
        debug=args.debug
    )
    
    app.run(open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
