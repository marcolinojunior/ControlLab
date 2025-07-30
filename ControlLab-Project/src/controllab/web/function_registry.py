"""
ControlLab Function Registry - Sistema de Reconhecimento de Funções
Integra documentação completa de funções com o agente inteligente

Este módulo:
1. Carrega documentação completa dos módulos 1-7
2. Cria registry de funções disponíveis  
3. Permite ao agente descobrir e usar funções automaticamente
4. Fornece contexto pedagógico para cada função
"""

import json
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import importlib


@dataclass
class FunctionInfo:
    """Informação completa sobre uma função do ControlLab."""
    name: str
    module: str
    file_path: str
    line_number: int
    signature: str
    purpose: str
    theoretical_context: str
    inputs: Dict[str, str]
    outputs: Dict[str, str]
    examples: List[str]
    related_functions: List[str]
    educational_notes: List[str]


class ControlLabFunctionRegistry:
    """
    Registry completo de funções do ControlLab.
    Baseado na documentação completa dos módulos 1-7.
    """
    
    def __init__(self):
        self.functions = {}
        self.modules = {}
        self.categories = {}
        self.load_function_database()
    
    def load_function_database(self):
        """Carrega base de dados de funções da documentação."""
        
        # Módulo Analysis - Análise de Estabilidade
        self.register_analysis_functions()
        
        # Módulo Core - Núcleo Simbólico  
        self.register_core_functions()
        
        # Módulo Design - Projeto de Controladores
        self.register_design_functions()
        
        # Módulo Modeling - Modelagem e Transformadas
        self.register_modeling_functions()
        
        # Módulo Numerical - Interface Numérica
        self.register_numerical_functions()
        
        # Módulo Visualization - Visualização
        self.register_visualization_functions()
    
    def register_analysis_functions(self):
        """Registra funções do módulo de análise de estabilidade."""
        
        # Função principal de análise
        self.register_function(FunctionInfo(
            name="analyze_stability",
            module="controllab.analysis.stability_analysis",
            file_path="src/controllab/analysis/stability_analysis.py",
            line_number=644,
            signature="analyze_stability(tf_obj, show_steps: bool = True) -> ComprehensiveStabilityReport",
            purpose="Realiza análise completa de estabilidade usando múltiplos métodos (Routh-Hurwitz, Root Locus, Margens)",
            theoretical_context="""
            Implementa os conceitos fundamentais do Capítulo 6 - ESTABILIDADE. 
            A estabilidade é uma propriedade fundamental que determina se o sistema 
            converge (estável), diverge (instável), ou permanece limítrofe (marginalmente estável).
            
            Métodos implementados:
            - Critério de Routh-Hurwitz (6.2-6.4): Método algébrico
            - Lugar Geométrico das Raízes (Caps 8-9): Análise gráfica  
            - Resposta em Frequência (Cap 10): Diagramas de Bode e Nyquist
            """,
            inputs={
                "tf_obj": "Função de transferência ou sistema a ser analisado",
                "show_steps": "Exibir todos os passos pedagógicos (padrão: True)"
            },
            outputs={
                "ComprehensiveStabilityReport": "Relatório completo com análise multi-método, conclusões e validação cruzada"
            },
            examples=[
                "analyze_stability(SymbolicTransferFunction([1], [1, 2, 1]))",
                "report = analyze_stability(tf); print(report.get_executive_summary())"
            ],
            related_functions=["quick_stability_check", "compare_systems_stability"],
            educational_notes=[
                "Sistema é estável se todos os polos estão no semiplano esquerdo",
                "Métodos diferentes devem concordar para validação",
                "Relatório inclui interpretação física dos resultados"
            ]
        ))
        
        # Verificação rápida
        self.register_function(FunctionInfo(
            name="quick_stability_check",
            module="controllab.analysis.stability_analysis", 
            file_path="src/controllab/analysis/stability_analysis.py",
            line_number=650,
            signature="quick_stability_check(tf_obj) -> bool",
            purpose="Verificação rápida de estabilidade usando método mais eficiente",
            theoretical_context="Usa Routh-Hurwitz como método prioritário para análise eficiente",
            inputs={"tf_obj": "Função de transferência"},
            outputs={"bool": "True se estável, False se instável"},
            examples=["is_stable = quick_stability_check(transfer_function)"],
            related_functions=["analyze_stability"],
            educational_notes=["Método mais rápido para verificação inicial"]
        ))
    
    def register_core_functions(self):
        """Registra funções do núcleo simbólico."""
        
        self.register_function(FunctionInfo(
            name="SymbolicTransferFunction",
            module="controllab.core.symbolic_tf",
            file_path="src/controllab/core/symbolic_tf.py", 
            line_number=1,
            signature="SymbolicTransferFunction(numerator, denominator)",
            purpose="Representa função de transferência com capacidade simbólica e numérica",
            theoretical_context="""
            Implementa representação híbrida de funções de transferência G(s) = N(s)/D(s).
            Combina álgebra simbólica (SymPy) com capacidades numéricas (python-control).
            Fundamental para análise transparente em sistemas de controle.
            """,
            inputs={
                "numerator": "Lista de coeficientes do numerador (ordem decrescente)",
                "denominator": "Lista de coeficientes do denominador (ordem decrescente)"
            },
            outputs={
                "SymbolicTransferFunction": "Objeto com métodos simbólicos e numéricos"
            },
            examples=[
                "G = SymbolicTransferFunction([1], [1, 2, 1])",
                "poles = G.poles(); zeros = G.zeros()"
            ],
            related_functions=["poles", "zeros", "is_stable", "step_response"],
            educational_notes=[
                "Mantém representação simbólica e numérica simultaneamente",
                "Permite análise transparente sem 'caixa-preta'",
                "Base para todos os outros módulos"
            ]
        ))
    
    def register_design_functions(self):
        """Registra funções de projeto de controladores."""
        
        self.register_function(FunctionInfo(
            name="PIDTuner",
            module="controllab.design.pid_tuning",
            file_path="src/controllab/design/pid_tuning.py",
            line_number=1,
            signature="PIDTuner()",
            purpose="Projeto e sintonia de controladores PID com múltiplos métodos",
            theoretical_context="""
            Implementa métodos clássicos e modernos de sintonia PID:
            - Ziegler-Nichols (resposta ao degrau e oscilação crítica)
            - Posicionamento de polos
            - Otimização por critérios de performance
            - Anti-windup e limitações práticas
            """,
            inputs={},
            outputs={"PIDTuner": "Objeto para projeto de controladores PID"},
            examples=[
                "tuner = PIDTuner()",
                "kp, ki, kd = tuner.tune_ziegler_nichols(plant)"
            ],
            related_functions=["tune_ziegler_nichols", "pole_placement", "optimize_pid"],
            educational_notes=[
                "Múltiplos métodos para diferentes situações",
                "Inclui considerações práticas (anti-windup)",
                "Validação automática de estabilidade"
            ]
        ))
    
    def register_modeling_functions(self):
        """Registra funções de modelagem e transformadas."""
        
        self.register_function(FunctionInfo(
            name="LaplaceTransformer",
            module="controllab.modeling.laplace_transform",
            file_path="src/controllab/modeling/laplace_transform.py",
            line_number=1,
            signature="LaplaceTransformer()",
            purpose="Transformadas de Laplace e análise no domínio s",
            theoretical_context="""
            Implementa transformadas de Laplace para modelagem de sistemas:
            - Transformada direta e inversa
            - Frações parciais
            - Teoremas (valor inicial, final, convolução)
            - Conexão entre domínio tempo e frequência
            """,
            inputs={},
            outputs={"LaplaceTransformer": "Objeto para transformadas de Laplace"},
            examples=[
                "transformer = LaplaceTransformer()",
                "F_s = transformer.transform('t**2 * exp(-t)')"
            ],
            related_functions=["transform", "inverse_transform", "partial_fractions"],
            educational_notes=[
                "Base matemática para análise de sistemas",
                "Conexão entre representações temporais e frequenciais",
                "Essencial para compreensão teórica"
            ]
        ))
    
    def register_numerical_functions(self):
        """Registra funções da interface numérica."""
        
        self.register_function(FunctionInfo(
            name="NumericalInterface",
            module="controllab.numerical.interface",
            file_path="src/controllab/numerical/interface.py",
            line_number=1,
            signature="NumericalInterface()",
            purpose="Interface entre representações simbólicas e numéricas",
            theoretical_context="""
            Ponte entre SymPy (simbólico) e python-control (numérico).
            Permite computações eficientes mantendo transparência simbólica.
            29 métodos especializados para conversão e validação.
            """,
            inputs={},
            outputs={"NumericalInterface": "Interface para computações híbridas"},
            examples=[
                "interface = NumericalInterface()",
                "control_tf = interface.to_control_tf(symbolic_tf)"
            ],
            related_functions=["to_control_tf", "to_symbolic", "validate_conversion"],
            educational_notes=[
                "Combina melhor dos dois mundos (simbólico + numérico)",
                "Validação automática de conversões",
                "Performance otimizada para computações pesadas"
            ]
        ))
    
    def register_visualization_functions(self):
        """Registra funções de visualização."""
        
        self.register_function(FunctionInfo(
            name="InteractivePlotGenerator",
            module="controllab.visualization.interactive_plots",
            file_path="src/controllab/visualization/interactive_plots.py", 
            line_number=1,
            signature="InteractivePlotGenerator()",
            purpose="Geração de visualizações interativas para análise pedagógica",
            theoretical_context="""
            Cria visualizações educacionais interativas:
            - Diagramas de Bode com explicações
            - Mapas de polos-zeros com interpretação
            - Respostas temporais com análise
            - Root Locus com parâmetros variáveis
            """,
            inputs={},
            outputs={"InteractivePlotGenerator": "Gerador de plots interativos"},
            examples=[
                "plotter = InteractivePlotGenerator()",
                "plotter.bode_plot(transfer_function, interactive=True)"
            ],
            related_functions=["bode_plot", "pole_zero_map", "step_response_plot"],
            educational_notes=[
                "Visualizações explicativas não apenas ilustrativas",
                "Interatividade promove exploração",
                "Conexão visual com conceitos teóricos"
            ]
        ))
    
    def register_function(self, func_info: FunctionInfo):
        """Registra uma função no registry."""
        self.functions[func_info.name] = func_info
        
        # Organiza por módulo
        if func_info.module not in self.modules:
            self.modules[func_info.module] = []
        self.modules[func_info.module].append(func_info.name)
        
        # Organiza por categoria
        category = func_info.module.split('.')[-1]  # analysis, core, design, etc.
        if category not in self.categories:
            self.categories[category] = []
        self.categories[category].append(func_info.name)
    
    def find_function(self, query: str) -> List[FunctionInfo]:
        """
        Encontra funções baseado em query.
        
        Args:
            query: Termo de busca (nome, propósito, contexto teórico)
            
        Returns:
            Lista de funções que correspondem à query
        """
        query_lower = query.lower()
        matches = []
        
        for func_info in self.functions.values():
            # Busca no nome
            if query_lower in func_info.name.lower():
                matches.append(func_info)
                continue
                
            # Busca no propósito
            if query_lower in func_info.purpose.lower():
                matches.append(func_info)
                continue
                
            # Busca no contexto teórico
            if query_lower in func_info.theoretical_context.lower():
                matches.append(func_info)
                continue
                
            # Busca nas notas educacionais
            for note in func_info.educational_notes:
                if query_lower in note.lower():
                    matches.append(func_info)
                    break
        
        return matches
    
    def get_function_by_capability(self, capability: str) -> List[FunctionInfo]:
        """
        Encontra funções por capacidade específica.
        
        Args:
            capability: Capacidade desejada (ex: "stability", "pid", "visualization")
            
        Returns:
            Lista de funções com essa capacidade
        """
        capability_lower = capability.lower()
        
        # Mapeamento de capacidades para palavras-chave
        capability_keywords = {
            'stability': ['stability', 'stable', 'estabilidade', 'routh', 'hurwitz'],
            'pid': ['pid', 'controller', 'controlador', 'tuning', 'sintonia'],
            'analysis': ['analysis', 'analyze', 'analise', 'pole', 'zero'],
            'visualization': ['plot', 'graph', 'visualization', 'visualização', 'bode'],
            'modeling': ['model', 'laplace', 'transform', 'modelo'],
            'design': ['design', 'projeto', 'controller', 'compensator']
        }
        
        keywords = capability_keywords.get(capability_lower, [capability_lower])
        matches = []
        
        for keyword in keywords:
            matches.extend(self.find_function(keyword))
        
        # Remove duplicatas mantendo ordem
        seen = set()
        unique_matches = []
        for match in matches:
            if match.name not in seen:
                seen.add(match.name)
                unique_matches.append(match)
        
        return unique_matches
    
    def get_function_context(self, function_name: str) -> Optional[FunctionInfo]:
        """Retorna contexto completo de uma função."""
        return self.functions.get(function_name)
    
    def get_category_functions(self, category: str) -> List[FunctionInfo]:
        """Retorna todas as funções de uma categoria."""
        function_names = self.categories.get(category, [])
        return [self.functions[name] for name in function_names]
    
    def generate_function_call_example(self, function_name: str, context: Dict[str, Any] = None) -> str:
        """
        Gera exemplo de chamada de função baseado no contexto.
        
        Args:
            function_name: Nome da função
            context: Contexto atual (sistema analisado, etc.)
            
        Returns:
            String com exemplo de código
        """
        func_info = self.get_function_context(function_name)
        if not func_info:
            return f"# Função {function_name} não encontrada"
        
        # Usa exemplos pré-definidos se disponíveis
        if func_info.examples:
            example = func_info.examples[0]
            
            # Adapta exemplo ao contexto atual
            if context and 'current_system' in context:
                example = example.replace('transfer_function', 'current_system')
                example = example.replace('tf', 'current_system')
            
            return f"# {func_info.purpose}\n{example}"
        
        # Gera exemplo básico baseado na assinatura
        return f"# {func_info.purpose}\n# {func_info.signature}"
    
    def get_educational_guidance(self, function_name: str) -> str:
        """Retorna orientação pedagógica para uma função."""
        func_info = self.get_function_context(function_name)
        if not func_info:
            return "Função não encontrada"
        
        guidance = f"""
🎓 **Contexto Educacional: {func_info.name}**

📚 **Propósito:**
{func_info.purpose}

🔬 **Fundamento Teórico:**
{func_info.theoretical_context.strip()}

💡 **Notas Pedagógicas:**
"""
        for note in func_info.educational_notes:
            guidance += f"• {note}\n"
        
        guidance += f"""
🔗 **Funções Relacionadas:**
{', '.join(func_info.related_functions)}

📝 **Exemplo de Uso:**
```python
{func_info.examples[0] if func_info.examples else 'Ver documentação'}
```
        """
        
        return guidance.strip()


# Instância global do registry
FUNCTION_REGISTRY = ControlLabFunctionRegistry()


def get_available_functions() -> Dict[str, List[str]]:
    """Retorna funções disponíveis organizadas por categoria."""
    return FUNCTION_REGISTRY.categories


def find_functions_for_task(task_description: str) -> List[FunctionInfo]:
    """
    Encontra funções adequadas para uma tarefa específica.
    
    Args:
        task_description: Descrição da tarefa (ex: "analisar estabilidade")
        
    Returns:
        Lista de funções adequadas com contexto completo
    """
    return FUNCTION_REGISTRY.find_function(task_description)


def generate_code_for_task(task_description: str, context: Dict[str, Any] = None) -> str:
    """
    Gera código Python para executar uma tarefa.
    
    Args:
        task_description: Descrição da tarefa
        context: Contexto atual (sistema, variáveis, etc.)
        
    Returns:
        Código Python para executar a tarefa
    """
    functions = find_functions_for_task(task_description)
    
    if not functions:
        return f"# Nenhuma função encontrada para: {task_description}"
    
    # Usa a função mais relevante (primeira da lista)
    main_function = functions[0]
    
    code = f"""
# Tarefa: {task_description}
# Usando: {main_function.name} - {main_function.purpose}

from {main_function.module} import {main_function.name}

"""
    
    # Adiciona exemplo de uso
    code += FUNCTION_REGISTRY.generate_function_call_example(
        main_function.name, context
    )
    
    return code.strip()


if __name__ == "__main__":
    # Teste do registry
    registry = ControlLabFunctionRegistry()
    
    print("📚 Funções disponíveis por categoria:")
    for category, functions in registry.categories.items():
        print(f"  {category}: {len(functions)} funções")
    
    print("\n🔍 Teste de busca:")
    stability_functions = registry.find_function("stability")
    for func in stability_functions:
        print(f"  • {func.name}: {func.purpose}")
    
    print("\n🎓 Orientação pedagógica para analyze_stability:")
    print(registry.get_educational_guidance("analyze_stability"))
