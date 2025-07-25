#!/usr/bin/env python3
"""
Módulo de Visualização de Passos - ControlLab
=============================================

Este módulo implementa funcionalidades para visualizar os passos pedagógicos
das transformações matemáticas realizadas no módulo de modelagem.

Funcionalidades:
- Visualização de passos da transformada de Laplace
- Visualização de passos da expansão em frações parciais
- Formatação em LaTeX para notebooks Jupyter
- Exportação para HTML/PDF
"""

import os
import warnings
from typing import Dict, List, Any, Optional, Union

try:
    import sympy as sp
    from sympy import latex, pretty
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False

try:
    from IPython.display import display, Markdown, HTML, Latex
    IPYTHON_AVAILABLE = True
except ImportError:
    IPYTHON_AVAILABLE = False


class StepVisualizationHistory:
    """
    Classe para armazenar e formatar histórico de visualização
    """
    
    def __init__(self):
        self.steps = []
        self.title = ""
        self.description = ""
    
    def add_step(self, step_title: str, expression_before, expression_after, 
                 explanation: str, step_type: str = "transformation"):
        """Adiciona um passo ao histórico"""
        step = {
            'title': step_title,
            'before': expression_before,
            'after': expression_after,
            'explanation': explanation,
            'type': step_type,
            'step_number': len(self.steps) + 1
        }
        self.steps.append(step)
    
    def add_text_step(self, title: str, content: str, step_type: str = "explanation"):
        """Adiciona um passo apenas textual"""
        step = {
            'title': title,
            'content': content,
            'type': step_type,
            'step_number': len(self.steps) + 1
        }
        self.steps.append(step)
    
    def get_formatted_steps(self, format_type: str = "text") -> str:
        """Obtém os passos formatados"""
        if format_type == "latex":
            return self._format_latex()
        elif format_type == "html":
            return self._format_html()
        elif format_type == "markdown":
            return self._format_markdown()
        else:
            return self._format_text()
    
    def _format_text(self) -> str:
        """Formatação em texto simples"""
        output = []
        if self.title:
            output.append(f"=== {self.title} ===")
        if self.description:
            output.append(f"{self.description}\n")
        
        for step in self.steps:
            output.append(f"Passo {step['step_number']}: {step['title']}")
            
            if step['type'] == 'transformation':
                if SYMPY_AVAILABLE:
                    output.append(f"  Antes:  {pretty(step['before'])}")
                    output.append(f"  Depois: {pretty(step['after'])}")
                else:
                    output.append(f"  Antes:  {step['before']}")
                    output.append(f"  Depois: {step['after']}")
                output.append(f"  Explicação: {step['explanation']}")
            else:
                output.append(f"  {step.get('content', step.get('explanation', ''))}")
            
            output.append("")
        
        return "\n".join(output)
    
    def _format_latex(self) -> str:
        """Formatação em LaTeX"""
        if not SYMPY_AVAILABLE:
            return self._format_text()
        
        output = []
        if self.title:
            output.append(f"\\section{{{self.title}}}")
        if self.description:
            output.append(f"{self.description}\n")
        
        for step in self.steps:
            output.append(f"\\subsection{{Passo {step['step_number']}: {step['title']}}}")
            
            if step['type'] == 'transformation':
                output.append(f"Antes: ${latex(step['before'])}$")
                output.append(f"Depois: ${latex(step['after'])}$")
                output.append(f"Explicação: {step['explanation']}")
            else:
                output.append(step.get('content', step.get('explanation', '')))
            
            output.append("")
        
        return "\n".join(output)
    
    def _format_html(self) -> str:
        """Formatação em HTML"""
        output = []
        output.append("<div class='step-visualization'>")
        
        if self.title:
            output.append(f"<h2>{self.title}</h2>")
        if self.description:
            output.append(f"<p>{self.description}</p>")
        
        for step in self.steps:
            output.append(f"<div class='step'>")
            output.append(f"<h3>Passo {step['step_number']}: {step['title']}</h3>")
            
            if step['type'] == 'transformation':
                if SYMPY_AVAILABLE:
                    output.append(f"<p><strong>Antes:</strong> {latex(step['before'])}</p>")
                    output.append(f"<p><strong>Depois:</strong> {latex(step['after'])}</p>")
                else:
                    output.append(f"<p><strong>Antes:</strong> {step['before']}</p>")
                    output.append(f"<p><strong>Depois:</strong> {step['after']}</p>")
                output.append(f"<p><strong>Explicação:</strong> {step['explanation']}</p>")
            else:
                output.append(f"<p>{step.get('content', step.get('explanation', ''))}</p>")
            
            output.append("</div>")
        
        output.append("</div>")
        return "\n".join(output)
    
    def _format_markdown(self) -> str:
        """Formatação em Markdown"""
        output = []
        if self.title:
            output.append(f"## {self.title}")
        if self.description:
            output.append(f"{self.description}\n")
        
        for step in self.steps:
            output.append(f"### Passo {step['step_number']}: {step['title']}")
            
            if step['type'] == 'transformation':
                if SYMPY_AVAILABLE:
                    output.append(f"**Antes:** $${latex(step['before'])}$$")
                    output.append(f"**Depois:** $${latex(step['after'])}$$")
                else:
                    output.append(f"**Antes:** `{step['before']}`")
                    output.append(f"**Depois:** `{step['after']}`")
                output.append(f"**Explicação:** {step['explanation']}")
            else:
                output.append(step.get('content', step.get('explanation', '')))
            
            output.append("")
        
        return "\n".join(output)


def show_laplace_steps(transformation_history, format_type: str = "auto"):
    """
    Exibe os passos da transformada de Laplace
    
    Args:
        transformation_history: Histórico de transformação
        format_type: Tipo de formatação ('text', 'latex', 'html', 'markdown', 'auto')
    """
    if not hasattr(transformation_history, 'steps'):
        print("Histórico de transformação não disponível ou vazio")
        return
    
    # Determinar formato automaticamente
    if format_type == "auto":
        if IPYTHON_AVAILABLE:
            format_type = "markdown"
        else:
            format_type = "text"
    
    # Criar visualização
    viz = StepVisualizationHistory()
    viz.title = "Transformada de Laplace - Passos Detalhados"
    viz.description = "Sequência de transformações aplicadas:"
    
    # Copiar passos do histórico original
    for i, step in enumerate(transformation_history.steps):
        if isinstance(step, dict):
            viz.add_step(
                step_title=step.get('operation', f'Operação {i+1}'),
                expression_before=step.get('input', ''),
                expression_after=step.get('result', ''),
                explanation=step.get('explanation', ''),
                step_type='transformation'
            )
        else:
            viz.add_text_step(f'Passo {i+1}', str(step))
    
    # Exibir resultado
    formatted_output = viz.get_formatted_steps(format_type)
    
    if IPYTHON_AVAILABLE and format_type in ['markdown', 'latex']:
        if format_type == 'latex':
            display(Latex(formatted_output))
        else:
            display(Markdown(formatted_output))
    else:
        print(formatted_output)


def show_partial_fraction_steps(expansion_process, format_type: str = "auto"):
    """
    Exibe os passos da expansão em frações parciais
    
    Args:
        expansion_process: Processo de expansão
        format_type: Tipo de formatação
    """
    if not hasattr(expansion_process, 'steps') and not hasattr(expansion_process, 'history'):
        print("Processo de expansão não disponível ou vazio")
        return
    
    # Determinar formato automaticamente
    if format_type == "auto":
        if IPYTHON_AVAILABLE:
            format_type = "markdown"
        else:
            format_type = "text"
    
    # Criar visualização
    viz = StepVisualizationHistory()
    viz.title = "Expansão em Frações Parciais - Passos Detalhados"
    viz.description = "Processo de decomposição em frações parciais:"
    
    # Obter histórico
    history = getattr(expansion_process, 'history', expansion_process)
    if hasattr(history, 'steps'):
        steps = history.steps
    else:
        steps = history if isinstance(history, list) else [history]
    
    # Processar passos
    for i, step in enumerate(steps):
        if isinstance(step, dict):
            viz.add_step(
                step_title=step.get('operation', f'Expansão {i+1}'),
                expression_before=step.get('original', ''),
                expression_after=step.get('expanded', ''),
                explanation=step.get('method', ''),
                step_type='transformation'
            )
        else:
            viz.add_text_step(f'Passo {i+1}', str(step))
    
    # Exibir resultado
    formatted_output = viz.get_formatted_steps(format_type)
    
    if IPYTHON_AVAILABLE and format_type in ['markdown', 'latex']:
        if format_type == 'latex':
            display(Latex(formatted_output))
        else:
            display(Markdown(formatted_output))
    else:
        print(formatted_output)


def export_to_html(visualization_history, filename: str, include_css: bool = True):
    """
    Exporta visualização para arquivo HTML
    
    Args:
        visualization_history: Histórico de visualização
        filename: Nome do arquivo
        include_css: Incluir CSS básico
    """
    html_content = visualization_history.get_formatted_steps("html")
    
    if include_css:
        css = """
        <style>
        .step-visualization {
            font-family: 'Segoe UI', Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        .step {
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 15px;
            margin: 15px 0;
            background-color: #f9f9f9;
        }
        .step h3 {
            color: #2c3e50;
            margin-top: 0;
        }
        .step p {
            margin: 8px 0;
        }
        </style>
        """
        html_content = f"<html><head>{css}</head><body>{html_content}</body></html>"
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"Exportado para: {filename}")
    except Exception as e:
        print(f"Erro ao exportar para HTML: {e}")


def export_to_pdf(visualization_history, filename: str):
    """
    Exporta visualização para PDF (requer pdfkit ou similar)
    
    Args:
        visualization_history: Histórico de visualização
        filename: Nome do arquivo
    """
    try:
        import pdfkit
        
        html_content = visualization_history.get_formatted_steps("html")
        pdfkit.from_string(html_content, filename)
        print(f"Exportado para PDF: {filename}")
        
    except ImportError:
        print("pdfkit não disponível. Use: pip install pdfkit")
        print("Exportando para HTML como alternativa...")
        html_filename = filename.replace('.pdf', '.html')
        export_to_html(visualization_history, html_filename)
    except Exception as e:
        print(f"Erro ao exportar para PDF: {e}")


def create_jupyter_visualization(transformation_history, title: str = "Passos de Transformação"):
    """
    Cria visualização otimizada para Jupyter notebooks
    
    Args:
        transformation_history: Histórico de transformação
        title: Título da visualização
    """
    if not IPYTHON_AVAILABLE:
        print("IPython não disponível. Use show_laplace_steps() para visualização básica.")
        return
    
    viz = StepVisualizationHistory()
    viz.title = title
    
    # Processar histórico
    if hasattr(transformation_history, 'steps'):
        for step in transformation_history.steps:
            if isinstance(step, dict):
                viz.add_step(
                    step_title=step.get('operation', 'Transformação'),
                    expression_before=step.get('input', ''),
                    expression_after=step.get('result', ''),
                    explanation=step.get('explanation', ''),
                    step_type='transformation'
                )
    
    # Exibir com formatação rica
    display(Markdown(viz.get_formatted_steps("markdown")))


# Classe de fallback quando bibliotecas não estão disponíveis
class FallbackStepVisualization:
    """Classe de fallback para visualização de passos"""
    
    def __init__(self):
        warnings.warn("Funcionalidades de visualização limitadas - instale SymPy e IPython para funcionalidade completa")
    
    def show_laplace_steps(self, transformation_history, format_type="text"):
        print("=== Passos da Transformada de Laplace ===")
        if hasattr(transformation_history, 'steps'):
            for i, step in enumerate(transformation_history.steps):
                print(f"Passo {i+1}: {step}")
        else:
            print("Histórico não disponível")
    
    def show_partial_fraction_steps(self, expansion_process, format_type="text"):
        print("=== Passos da Expansão em Frações Parciais ===")
        if hasattr(expansion_process, 'steps'):
            for i, step in enumerate(expansion_process.steps):
                print(f"Passo {i+1}: {step}")
        else:
            print("Processo não disponível")


# Instanciar fallback se necessário
if not SYMPY_AVAILABLE or not IPYTHON_AVAILABLE:
    # Criar instância de fallback
    _fallback = FallbackStepVisualization()
    
    if 'show_laplace_steps' not in locals():
        show_laplace_steps = _fallback.show_laplace_steps
    
    if 'show_partial_fraction_steps' not in locals():
        show_partial_fraction_steps = _fallback.show_partial_fraction_steps
