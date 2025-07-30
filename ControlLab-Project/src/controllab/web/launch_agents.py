#!/usr/bin/env python3
"""
ControlLab Module 8 - Agent Integration Launcher
Inicializador simplificado para demonstração dos agentes

Para uso completo, instale: pip install websockets sympy control matplotlib
"""

import os
import sys
import json
import webbrowser
import time
from pathlib import Path
import subprocess
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
import argparse


class ControlLabAgentLauncher:
    """
    Lançador simplificado para demonstração do Módulo 8.
    
    Este script:
    1. Inicia servidor HTTP para a interface web
    2. Tenta iniciar servidor WebSocket (se disponível)
    3. Abre navegador na interface
    4. Exibe status e instruções
    """
    
    def __init__(self, port=8080, auto_browser=True):
        self.port = port
        self.auto_browser = auto_browser
        self.project_root = Path(__file__).parent.parent.parent.parent
        self.frontend_dir = Path(__file__).parent / 'frontend'
        
    def check_dependencies(self):
        """Verifica dependências disponíveis."""
        deps = {
            'websockets': False,
            'sympy': False,
            'control': False,
            'matplotlib': False,
            'numpy': False
        }
        
        for dep in deps:
            try:
                __import__(dep)
                deps[dep] = True
            except ImportError:
                pass
                
        return deps
    
    def print_status(self, deps):
        """Imprime status do sistema."""
        print("\n" + "="*60)
        print("🤖 CONTROLLAB MÓDULO 8 - INTEGRAÇÃO COM AGENTES")
        print("="*60)
        
        print("📦 STATUS DAS DEPENDÊNCIAS:")
        for dep, available in deps.items():
            status = "✅" if available else "❌"
            print(f"  {status} {dep}")
        
        complete = all(deps.values())
        print(f"\n🎯 FUNCIONALIDADE: {'✅ COMPLETA' if complete else '⚠️ LIMITADA'}")
        
        print(f"\n🌐 INTERFACE WEB: http://localhost:{self.port}/agent_interface.html")
        
        if not complete:
            print("\n💡 Para funcionalidade completa, execute:")
            print("pip install websockets sympy control matplotlib numpy")
        
        print("="*60)
        return complete
    
    def start_http_server(self):
        """Inicia servidor HTTP para interface."""
        try:
            os.chdir(self.frontend_dir)
            
            handler = SimpleHTTPRequestHandler
            server = HTTPServer(('localhost', self.port), handler)
            
            print(f"🌐 Servidor HTTP iniciado na porta {self.port}")
            
            # Executa em thread separada
            server_thread = threading.Thread(
                target=server.serve_forever,
                daemon=True
            )
            server_thread.start()
            
            return server
            
        except Exception as e:
            print(f"❌ Erro ao iniciar servidor HTTP: {e}")
            return None
    
    def try_start_websocket(self):
        """Tenta iniciar servidor WebSocket."""
        try:
            # Importa e inicia servidor de agentes
            from .agent_integration import WebSocketAgentServer
            
            print("🔌 Tentando iniciar servidor WebSocket...")
            
            # Executa em subprocess para não bloquear
            script_path = Path(__file__).parent / 'agent_integration.py'
            proc = subprocess.Popen([
                sys.executable, str(script_path), '--server'
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            print("✅ Servidor WebSocket iniciado")
            return proc
            
        except ImportError:
            print("⚠️ WebSocket não disponível - instale 'websockets'")
            return None
        except Exception as e:
            print(f"❌ Erro ao iniciar WebSocket: {e}")
            return None
    
    def open_browser(self):
        """Abre navegador na interface."""
        if self.auto_browser:
            url = f"http://localhost:{self.port}/agent_interface.html"
            try:
                time.sleep(2)  # Aguarda servidor estabilizar
                webbrowser.open(url)
                print(f"🌐 Navegador aberto em {url}")
            except Exception as e:
                print(f"⚠️ Erro ao abrir navegador: {e}")
    
    def run(self):
        """Executa o lançador."""
        print("🚀 Iniciando ControlLab Agent Launcher...")
        
        # Verifica dependências
        deps = self.check_dependencies()
        complete = self.print_status(deps)
        
        # Inicia servidor HTTP
        http_server = self.start_http_server()
        if not http_server:
            return
        
        # Tenta iniciar WebSocket
        ws_process = self.try_start_websocket()
        
        # Abre navegador
        browser_thread = threading.Thread(target=self.open_browser, daemon=True)
        browser_thread.start()
        
        print("\n🎮 COMO USAR:")
        print("1. A interface web abrirá automaticamente")
        print("2. Digite comandos no chat (ex: 'Analise G(s) = 1/(s+1)')")
        print("3. Use Ctrl+C para parar")
        
        if not complete:
            print("\n⚠️ MODO DEMONSTRAÇÃO:")
            print("• Interface web disponível")
            print("• Funcionalidades limitadas sem dependências")
            print("• Instale dependências para uso completo")
        
        try:
            print("\n✅ Servidores rodando. Pressione Ctrl+C para parar.\n")
            
            # Mantém programa rodando
            while True:
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n🛑 Parando servidores...")
            
            if http_server:
                http_server.shutdown()
            
            if ws_process:
                ws_process.terminate()
                
            print("✅ ControlLab Agent Launcher parado")


def main():
    """Função principal."""
    parser = argparse.ArgumentParser(
        description='ControlLab Módulo 8 - Lançador de Agentes'
    )
    parser.add_argument('--port', type=int, default=8080,
                       help='Porta do servidor HTTP (padrão: 8080)')
    parser.add_argument('--no-browser', action='store_true',
                       help='Não abre navegador automaticamente')
    
    args = parser.parse_args()
    
    launcher = ControlLabAgentLauncher(
        port=args.port,
        auto_browser=not args.no_browser
    )
    launcher.run()


if __name__ == "__main__":
    main()
