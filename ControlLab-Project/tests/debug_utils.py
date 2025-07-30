# DENTRO DE: tests/debug_utils.py
import time
from contextlib import contextmanager

def trace_import(func):
    """
    Este é o nosso "gancho" decorador. Ele imprime uma mensagem antes e
    depois de qualquer função que "embrulharmos" com ele.
    """
    def wrapper(*args, **kwargs):
        print(f"  -> A entrar em: {func.__name__} (do ficheiro {func.__module__})...")
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"  ✅ SUCESSO ao sair de: {func.__name__} (demorou {end_time - start_time:.4f}s)")
        return result
    return wrapper

@contextmanager
def test_section(name):
    """
    Este é o nosso "gancho" para testes. Ele imprime um cabeçalho
    no início de um teste e uma mensagem de sucesso no final.
    """
    print("\n" + "="*60)
    print(f"INICIANDO TESTE DA SECÇÃO: {name}")
    print("="*60)
    try:
        yield
    finally:
        print("\n" + "="*60)
        print(f"✅ SECÇÃO DE TESTE '{name}' CONCLUÍDA SEM BLOQUEIOS.")
        print("="*60)
