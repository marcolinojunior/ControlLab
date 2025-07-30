#!/usr/bin/env python3
"""
ControlLab Advanced Agent Launcher
Sistema completo de tutoria inteligente similar ao SymboLab

Este launcher:
1. Ativa automaticamente o ambiente virtual
2. Verifica todas as dependências 
3. Inicia servidor avançado com reconhecimento inteligente
4. Abre interface web moderna
5. Fornece tutoria socrática especializada
"""

import os
import sys
import subprocess
import time
import webbrowser
import threading
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
import argparse


class ControlLabAdvancedLauncher:
    """
    Launcher avançado para o sistema ControlLab com IA.
    Similar ao SymboLab em funcionalidade e interface.
    """
    
    def __init__(self, ws_port=8765, http_port=8080, auto_browser=True):
        self.ws_port = ws_port
        self.http_port = http_port
        self.auto_browser = auto_browser
        self.project_root = Path(__file__).parent.parent.parent.parent
        self.venv_path = self.project_root / "venv"
        self.frontend_dir = Path(__file__).parent / "frontend"
        
    def print_header(self):
        """Imprime header estilizado."""
        print("\n" + "="*70)
        print("🎛️ CONTROLLAB ADVANCED AI TUTOR - SISTEMA COMPLETO")
        print("="*70)
        print("🤖 Tutoria Inteligente • 📊 Análise Automática • 🎯 Socrático")
        print("Similar ao SymboLab para Engenharia de Controle")
        print("="*70)
    
    def check_environment(self) -> bool:
        """Verifica e ativa ambiente virtual automaticamente."""
        print("\n🔍 VERIFICANDO AMBIENTE VIRTUAL...")
        
        # Verifica se venv existe
        if not self.venv_path.exists():
            print(f"❌ Ambiente virtual não encontrado em: {self.venv_path}")
            print("💡 Execute primeiro: python -m venv venv")
            return False
            
        # Ativa ambiente virtual automaticamente
        if sys.platform == "win32":
            activate_script = self.venv_path / "Scripts" / "activate.bat"
            python_exe = self.venv_path / "Scripts" / "python.exe"
        else:
            activate_script = self.venv_path / "bin" / "activate"
            python_exe = self.venv_path / "bin" / "python"
            
        if not python_exe.exists():
            print(f"❌ Python não encontrado em: {python_exe}")
            return False
            
        print(f"✅ Ambiente virtual encontrado: {self.venv_path}")
        print(f"✅ Python executável: {python_exe}")
        
        # Testa importações críticas
        return self.test_dependencies(python_exe)
    
    def test_dependencies(self, python_exe: Path) -> bool:
        """Testa dependências críticas."""
        print("\n📦 VERIFICANDO DEPENDÊNCIAS...")
        
        dependencies = {
            'sympy': 'Álgebra simbólica',
            'numpy': 'Arrays numéricos', 
            'scipy': 'Algoritmos científicos',
            'control': 'Sistemas de controle',
            'matplotlib': 'Gráficos básicos',
            'plotly': 'Visualização interativa',
            'websockets': 'Comunicação em tempo real'
        }
        
        missing_deps = []
        
        for dep, description in dependencies.items():
            try:
                result = subprocess.run(
                    [str(python_exe), "-c", f"import {dep}; print('{dep} OK')"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0:
                    print(f"  ✅ {dep:12} - {description}")
                else:
                    print(f"  ❌ {dep:12} - {description}")
                    missing_deps.append(dep)
            except Exception as e:
                print(f"  ❌ {dep:12} - Erro: {e}")
                missing_deps.append(dep)
        
        if missing_deps:
            print(f"\n⚠️ DEPENDÊNCIAS FALTANDO: {', '.join(missing_deps)}")
            print("💡 Para instalar todas:")
            print(f"   {python_exe} -m pip install sympy numpy scipy control matplotlib plotly websockets")
            
            # Pergunta se deve continuar
            response = input("\n🤔 Continuar mesmo assim? (s/N): ").strip().lower()
            return response in ['s', 'sim', 'y', 'yes']
        else:
            print("✅ Todas as dependências estão disponíveis!")
            return True
    
    def start_advanced_agent_server(self) -> subprocess.Popen:
        """Inicia servidor avançado de agentes."""
        print(f"\n🤖 INICIANDO SERVIDOR AVANÇADO DE IA...")
        
        # Caminho para o servidor avançado
        agent_script = Path(__file__).parent / "advanced_agent.py"
        python_exe = self.venv_path / "Scripts" / "python.exe"
        
        try:
            # Inicia servidor em subprocess
            proc = subprocess.Popen(
                [str(python_exe), str(agent_script)],
                cwd=str(self.project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Aguarda um momento para verificar se iniciou
            time.sleep(2)
            
            if proc.poll() is None:  # Ainda rodando
                print(f"✅ Servidor de IA iniciado na porta {self.ws_port}")
                print(f"🔗 WebSocket: ws://localhost:{self.ws_port}")
                return proc
            else:
                # Falhou ao iniciar
                stdout, stderr = proc.communicate()
                print(f"❌ Erro ao iniciar servidor:")
                print(f"STDOUT: {stdout}")
                print(f"STDERR: {stderr}")
                return None
                
        except Exception as e:
            print(f"❌ Erro ao executar servidor: {e}")
            return None
    
    def start_http_server(self) -> HTTPServer:
        """Inicia servidor HTTP para frontend."""
        print(f"\n🌐 INICIANDO SERVIDOR WEB...")
        
        try:
            # Muda para diretório frontend
            original_dir = os.getcwd()
            os.chdir(self.frontend_dir)
            
            # Cria servidor HTTP
            handler = SimpleHTTPRequestHandler
            server = HTTPServer(('localhost', self.http_port), handler)
            
            print(f"✅ Servidor web iniciado na porta {self.http_port}")
            print(f"🔗 Interface: http://localhost:{self.http_port}/advanced_interface.html")
            
            # Executa em thread separada
            server_thread = threading.Thread(
                target=server.serve_forever,
                daemon=True
            )
            server_thread.start()
            
            # Volta para diretório original
            os.chdir(original_dir)
            
            return server
            
        except Exception as e:
            print(f"❌ Erro ao iniciar servidor HTTP: {e}")
            return None
    
    def open_browser(self):
        """Abre navegador na interface avançada."""
        if self.auto_browser:
            url = f"http://localhost:{self.http_port}/advanced_interface.html"
            try:
                time.sleep(3)  # Aguarda servidores estabilizarem
                webbrowser.open(url)
                print(f"🌐 Navegador aberto em: {url}")
            except Exception as e:
                print(f"⚠️ Erro ao abrir navegador: {e}")
                print(f"🔗 Acesse manualmente: {url}")
    
    def print_usage_instructions(self):
        """Imprime instruções de uso."""
        print("\n" + "="*70)
        print("🎯 COMO USAR O CONTROLLAB AI TUTOR")
        print("="*70)
        
        print("\n📝 ENTRADA INTELIGENTE:")
        print("  • Digite funções de transferência: G(s) = 1/(s²+2*s+1)")
        print("  • Faça perguntas: 'Analise a estabilidade'")
        print("  • Solicite projetos: 'Projete um controlador PID'")
        
        print("\n🤖 CHAT SOCRÁTICO:")
        print("  • IA explica conceitos passo a passo")
        print("  • Metodologia socrática (perguntas orientadas)")
        print("  • Feedback em tempo real")
        
        print("\n📊 VISUALIZAÇÕES AUTOMÁTICAS:")
        print("  • Mapas de polos e zeros")
        print("  • Diagramas de Bode")
        print("  • Respostas temporais")
        print("  • Análises de estabilidade")
        
        print("\n⚡ ANÁLISES AUTOMÁTICAS:")
        print("  • Reconhecimento inteligente de expressões")
        print("  • Sugestões contextuais")
        print("  • Múltiplos métodos de análise")
        print("  • Validação cruzada")
        
        print("\n🎓 RECURSOS PEDAGÓGICOS:")
        print("  • Explicações transparentes (anti-caixa-preta)")
        print("  • Conexão entre teoria e prática")
        print("  • Exemplos interativos")
        print("  • Progressão guiada")
        
        print("="*70)
    
    def print_system_status(self, ws_server, http_server):
        """Imprime status final do sistema."""
        print("\n" + "="*70)
        print("✅ CONTROLLAB AI TUTOR - SISTEMA ATIVO")
        print("="*70)
        
        print(f"🤖 Servidor de IA:  {'✅ Ativo' if ws_server else '❌ Falhou'}")
        print(f"🌐 Servidor Web:    {'✅ Ativo' if http_server else '❌ Falhou'}")
        print(f"🔗 Interface:       http://localhost:{self.http_port}/advanced_interface.html")
        print(f"📡 WebSocket:       ws://localhost:{self.ws_port}")
        
        if ws_server and http_server:
            print("\n🎉 SISTEMA COMPLETAMENTE FUNCIONAL!")
            print("🚀 Acesse a interface web para começar")
        else:
            print("\n⚠️ SISTEMA COM LIMITAÇÕES")
            if not ws_server:
                print("   • Servidor de IA não disponível")
                print("   • Funcionalidade limitada apenas à interface")
            if not http_server:
                print("   • Servidor web não disponível")
                print("   • Acesse diretamente o arquivo HTML")
        
        print("\n💡 COMANDOS ÚTEIS:")
        print("   • Ctrl+C para parar servidores")
        print("   • F5 para recarregar interface")
        print("   • Ctrl+Shift+I para abrir DevTools")
        
        print("="*70)
    
    def run(self):
        """Executa o launcher completo."""
        self.print_header()
        
        # Verifica ambiente
        if not self.check_environment():
            print("\n❌ Falha na verificação do ambiente")
            print("💡 Configure o ambiente virtual primeiro")
            return False
        
        # Inicia servidores
        ws_server = self.start_advanced_agent_server()
        http_server = self.start_http_server()
        
        # Abre navegador
        browser_thread = threading.Thread(target=self.open_browser, daemon=True)
        browser_thread.start()
        
        # Mostra instruções
        self.print_usage_instructions()
        
        # Status final
        self.print_system_status(ws_server, http_server)
        
        # Mantém programa rodando
        try:
            print("\n⏳ Servidores rodando... Pressione Ctrl+C para parar")
            while True:
                time.sleep(1)
                
                # Verifica se servidores ainda estão ativos
                if ws_server and ws_server.poll() is not None:
                    print("⚠️ Servidor de IA parou inesperadamente")
                    ws_server = None
                    
        except KeyboardInterrupt:
            print("\n🛑 Parando servidores...")
            
            if ws_server:
                ws_server.terminate()
                try:
                    ws_server.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    ws_server.kill()
                    
            if http_server:
                http_server.shutdown()
                
            print("✅ ControlLab AI Tutor parado com sucesso")
            return True


def main():
    """Função principal."""
    parser = argparse.ArgumentParser(
        description='ControlLab Advanced AI Tutor - Sistema Completo de Tutoria'
    )
    parser.add_argument('--ws-port', type=int, default=8765,
                       help='Porta do servidor WebSocket (padrão: 8765)')
    parser.add_argument('--http-port', type=int, default=8080,
                       help='Porta do servidor HTTP (padrão: 8080)')
    parser.add_argument('--no-browser', action='store_true',
                       help='Não abre navegador automaticamente')
    
    args = parser.parse_args()
    
    # Cria e executa launcher
    launcher = ControlLabAdvancedLauncher(
        ws_port=args.ws_port,
        http_port=args.http_port,
        auto_browser=not args.no_browser
    )
    
    success = launcher.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
