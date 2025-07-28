# debug_imports.py
import pdb

print("--- A iniciar a depuração de importações do ControlLab ---")
print("O depurador vai parar agora. Use o comando 's' para entrar na importação.")

# Inicia o depurador
pdb.set_trace()

# A linha que está a causar o problema.
# Vamos entrar nela e ver o que acontece.
import controllab

print("--- Importação concluída com sucesso! ---")
