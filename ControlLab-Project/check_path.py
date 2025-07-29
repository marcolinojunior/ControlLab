import sys
import os

print("--- Python Path (sys.path) ---")
for p in sys.path:
    print(p)

print("\n--- Conteúdo do Diretório src/ ---")
try:
    print(os.listdir('src'))
except FileNotFoundError:
    print("Diretório 'src' não encontrado no local atual.")

print("\n--- Conteúdo do Diretório src/controllab/ ---")
try:
    print(os.listdir('src/controllab'))
except FileNotFoundError:
    print("Diretório 'src/controllab' não encontrado.")

print("\n--- Tentativa de Importação ---")
try:
    from controllab.numerical import simulation
    print("SUCESSO: 'controllab.numerical.simulation' importado com sucesso!")
except Exception as e:
    print(f"FALHA: A importação falhou com o erro: {e}")
