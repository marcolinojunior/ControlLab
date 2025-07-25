#!/usr/bin/env python3
"""
Sistema de Histórico Pedagógico - ControlLab
Registra e formata operações para fins educacionais
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import sympy as sp

class OperationStep:
    """Representa um passo de operação no histórico"""
    
    def __init__(self, operation: str, description: str, before: Any, after: Any, metadata: Optional[Dict] = None):
        self.operation = operation
        self.description = description
        self.before = before
        self.after = after
        self.timestamp = datetime.now()
        self.metadata = metadata or {}
    
    def __str__(self):
        return f"[{self.operation}] {self.description}"

class OperationHistory:
    """
    Sistema de histórico para rastreamento pedagógico de operações simbólicas
    """
    
    def __init__(self, max_steps: int = 100):
        self.steps: List[OperationStep] = []
        self.max_steps = max_steps
    
    def add_step(self, operation: str, description: str, before: Any, after: Any, metadata: Optional[Dict] = None):
        """Adiciona um passo ao histórico"""
        step = OperationStep(operation, description, before, after, metadata)
        self.steps.append(step)
        
        # Limita o número de passos
        if len(self.steps) > self.max_steps:
            self.steps.pop(0)
    
    def get_formatted_steps(self, format_type: str = "text") -> str:
        """
        Retorna os passos formatados
        
        Args:
            format_type: 'text', 'latex', 'html'
        """
        if not self.steps:
            return "Nenhuma operação registrada."
        
        if format_type == "text":
            return self._format_text()
        elif format_type == "latex":
            return self._format_latex()
        elif format_type == "html":
            return self._format_html()
        else:
            return self._format_text()
    
    def _format_text(self) -> str:
        """Formatação em texto simples"""
        lines = ["=== Histórico de Operações ==="]
        
        for i, step in enumerate(self.steps, 1):
            lines.append(f"\n{i}. {step.operation}: {step.description}")
            lines.append(f"   Antes:  {step.before}")
            lines.append(f"   Depois: {step.after}")
            
            if step.metadata:
                lines.append(f"   Info:   {step.metadata}")
        
        return "\n".join(lines)
    
    def _format_latex(self) -> str:
        """Formatação em LaTeX"""
        lines = [r"\begin{enumerate}"]
        
        for step in self.steps:
            lines.append(f"\\item \\textbf{{{step.operation}}}: {step.description}")
            
            # Converte expressões simbólicas para LaTeX
            before_latex = self._to_latex_safe(step.before)
            after_latex = self._to_latex_safe(step.after)
            
            lines.append(f"\\begin{{align}}")
            lines.append(f"\\text{{Antes: }} &\\quad {before_latex} \\\\")
            lines.append(f"\\text{{Depois: }} &\\quad {after_latex}")
            lines.append(f"\\end{{align}}")
        
        lines.append(r"\end{enumerate}")
        return "\n".join(lines)
    
    def _format_html(self) -> str:
        """Formatação em HTML"""
        lines = ["<div class='operation-history'>"]
        lines.append("<h3>Histórico de Operações</h3>")
        lines.append("<ol>")
        
        for step in self.steps:
            lines.append(f"<li><strong>{step.operation}</strong>: {step.description}")
            lines.append("<div class='step-details'>")
            lines.append(f"<p><em>Antes:</em> {step.before}</p>")
            lines.append(f"<p><em>Depois:</em> {step.after}</p>")
            
            if step.metadata:
                lines.append(f"<p><em>Info:</em> {step.metadata}</p>")
            
            lines.append("</div></li>")
        
        lines.append("</ol></div>")
        return "\n".join(lines)
    
    def _to_latex_safe(self, expr) -> str:
        """Converte expressão para LaTeX de forma segura"""
        try:
            if hasattr(expr, '__str__') and hasattr(sp, 'latex'):
                # Tenta converter para SymPy se for string matemática
                if isinstance(expr, str) and any(op in expr for op in ['+', '-', '*', '/', '^', 's']):
                    try:
                        sympy_expr = sp.sympify(expr)
                        return sp.latex(sympy_expr)
                    except:
                        return str(expr)
                elif hasattr(expr, 'latex'):
                    return expr.latex()
                else:
                    return sp.latex(expr) if hasattr(expr, 'free_symbols') else str(expr)
            return str(expr)
        except:
            return str(expr)
    
    def clear(self):
        """Limpa o histórico"""
        self.steps.clear()
    
    def get_last_step(self) -> Optional[OperationStep]:
        """Retorna o último passo"""
        return self.steps[-1] if self.steps else None
    
    def get_steps_by_operation(self, operation: str) -> List[OperationStep]:
        """Retorna passos filtrados por tipo de operação"""
        return [step for step in self.steps if step.operation == operation]
    
    def get_formatted_history(self) -> List[str]:
        """Retorna histórico formatado como lista de strings"""
        formatted = []
        for i, step in enumerate(self.steps, 1):
            formatted.append(f"{i}. {step.operation}: {step.description}")
            if step.before:
                formatted.append(f"   Antes: {step.before}")
            if step.after:
                formatted.append(f"   Depois: {step.after}")
            if step.metadata:
                formatted.append(f"   Info: {step.metadata}")
        return formatted
    
    def get_latex_history(self) -> List[str]:
        """Retorna histórico formatado em LaTeX"""
        latex_steps = []
        for i, step in enumerate(self.steps, 1):
            latex_steps.append(f"\\item {step.operation}: {step.description}")
            if step.before:
                latex_steps.append(f"   Antes: ${self._to_latex_safe(step.before)}$")
            if step.after:
                latex_steps.append(f"   Depois: ${self._to_latex_safe(step.after)}$")
        return latex_steps
    
    def export_to_dict(self) -> List[Dict]:
        """Exporta histórico para dicionário"""
        return [
            {
                'operation': step.operation,
                'description': step.description,
                'before': str(step.before),
                'after': str(step.after),
                'timestamp': step.timestamp.isoformat(),
                'metadata': step.metadata
            }
            for step in self.steps
        ]


class HistoryManager:
    """
    Gerenciador de histórico para interface numérica
    Compatível com OperationHistory mas com interface simplificada
    """
    
    def __init__(self, max_steps: int = 100):
        self.history = OperationHistory(max_steps)
    
    def add_step(self, operation: str, description: str, before: Any, after: Any, metadata: Optional[Dict] = None):
        """Adiciona um passo ao histórico"""
        self.history.add_step(operation, description, before, after, metadata)
    
    def get_full_history(self) -> List[Dict]:
        """Retorna histórico completo como lista de dicionários"""
        return self.history.export_to_dict()
    
    def clear_history(self):
        """Limpa o histórico"""
        self.history.clear()
    
    def get_formatted_history(self) -> List[str]:
        """Retorna histórico formatado"""
        return self.history.get_formatted_history()
