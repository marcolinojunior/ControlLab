#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script simplificado para gerar estrutura de diretórios no formato markdown
Focado especificamente em criar a seção "## Estrutura de Diretórios"
"""

import os
from pathlib import Path

def gerar_estrutura_markdown_simples(diretorio_raiz=".", incluir_pycache=False):
    """
    Gera estrutura de diretórios em formato markdown simples
    
    Args:
        diretorio_raiz (str): Diretório para analisar
        incluir_pycache (bool): Se deve incluir __pycache__
    
    Returns:
        str: Estrutura em formato markdown
    """
    
    def deve_ignorar(nome):
        """Determina se deve ignorar um arquivo/diretório"""
        ignorar = {'.git', '.vscode', 'node_modules', '.pytest_cache', '.mypy_cache'}
        
        if not incluir_pycache:
            ignorar.add('__pycache__')
            
        return nome in ignorar or nome.startswith('.')
    
    def obter_estrutura(caminho, nivel=0):
        """Obtém estrutura recursivamente"""
        items = []
        indent = "\t" * nivel
        
        try:
            entradas = sorted(os.listdir(caminho))
        except PermissionError:
            return []
        
        # Separar diretórios e arquivos
        diretorios = []
        arquivos = []
        
        for entrada in entradas:
            if deve_ignorar(entrada):
                continue
                
            caminho_completo = os.path.join(caminho, entrada)
            if os.path.isdir(caminho_completo):
                diretorios.append(entrada)
            else:
                arquivos.append(entrada)
        
        # Adicionar diretórios
        for diretorio in diretorios:
            items.append(f"{indent}{diretorio}/")
            sub_caminho = os.path.join(caminho, diretorio)
            sub_items = obter_estrutura(sub_caminho, nivel + 1)
            items.extend(sub_items)
        
        # Adicionar arquivos
        for arquivo in arquivos:
            items.append(f"{indent}{arquivo}")
        
        return items
    
    # Gerar estrutura
    nome_raiz = os.path.basename(os.path.abspath(diretorio_raiz))
    estrutura = obter_estrutura(diretorio_raiz)
    
    # Montar resultado
    resultado = "## Estrutura de Diretórios\n\n```\n"
    resultado += f"{nome_raiz}/\n"
    for item in estrutura:
        resultado += f"{item}\n"
    resultado += "```\n"
    
    return resultado

def main():
    """Função principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Gera estrutura de diretórios simples")
    parser.add_argument("diretorio", nargs='?', default=".", help="Diretório (padrão: atual)")
    parser.add_argument("-o", "--output", help="Arquivo de saída")
    parser.add_argument("--pycache", action="store_true", help="Incluir __pycache__")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.diretorio):
        print(f"Erro: Diretório '{args.diretorio}' não encontrado!")
        return 1
    
    try:
        estrutura = gerar_estrutura_markdown_simples(
            args.diretorio, 
            incluir_pycache=args.pycache
        )
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(estrutura)
            print(f"Estrutura salva em: {args.output}")
        else:
            print(estrutura)
            
        return 0
        
    except Exception as e:
        print(f"Erro: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
