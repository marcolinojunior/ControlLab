import http.server
import socketserver
import webbrowser
import os
import multiprocessing
import sys
import time

# Obtém o caminho absoluto para o diretório do projeto
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Define a porta para o servidor HTTP
PORT = 8000
# Define o diretório que contém os arquivos do frontend
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "src", "controllab", "web", "frontend")
# Define o caminho para o script do servidor WebSocket
WEBSOCKET_SCRIPT = os.path.join(PROJECT_ROOT, "src", "controllab", "web", "backend", "websocket_server.py")

def run_http_server():
    """Inicia o servidor HTTP para servir os arquivos do frontend."""
    os.chdir(FRONTEND_DIR)
    with socketserver.TCPServer(("", PORT), http.server.SimpleHTTPRequestHandler) as httpd:
        print(f"Servindo em http://localhost:{PORT}")
        httpd.serve_forever()

def run_websocket_server():
    """Inicia o servidor WebSocket."""
    # Usa sys.executable para garantir que estamos usando o mesmo interpretador Python
    os.system(f"{sys.executable} {WEBSOCKET_SCRIPT}")


if __name__ == "__main__":
    # Cria processos separados para os servidores HTTP e WebSocket
    http_process = multiprocessing.Process(target=run_http_server)
    websocket_process = multiprocessing.Process(target=run_websocket_server)

    # Inicia os processos
    http_process.start()
    websocket_process.start()

    # Dá um tempo para os servidores iniciarem
    time.sleep(2)

    # Abre o navegador na página da interface de Laplace
    webbrowser.open_new_tab(f"http://localhost:{PORT}/laplace_interface.html")

    try:
        # Mantém o script principal em execução
        # e aguarda os processos terminarem
        http_process.join()
        websocket_process.join()
    except KeyboardInterrupt:
        # Encerra os processos se o script principal for interrompido
        print("\nEncerrando servidores...")
        http_process.terminate()
        websocket_process.terminate()
        http_process.join()
        websocket_process.join()
        print("Servidores encerrados.")
