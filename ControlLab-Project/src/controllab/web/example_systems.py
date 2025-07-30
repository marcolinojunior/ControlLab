"""
ControlLab Web - Sistema de Templates e Exemplos

Biblioteca de sistemas de exemplo e templates para facilitar o aprendizado,
integrada com o backend ControlLab existente.

Classes implementadas:
- ExampleSystems: Biblioteca de sistemas clássicos
- VirtualKeyboardTemplates: Templates para entrada de equações
- TutorialManager: Gerenciamento de tutoriais guiados
"""

import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

# Integração com backend ControlLab
try:
    from ..core.symbolic_tf import SymbolicTransferFunction
    from ..analysis.stability_analysis import StabilityAnalysisEngine
    from ..core.controller_design import PIDController, LeadLagCompensator
    BACKEND_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Backend ControlLab não disponível para templates: {e}")
    BACKEND_AVAILABLE = False


class ExampleSystems:
    """
    Biblioteca de sistemas de exemplo para educação em controle.
    
    Contém sistemas clássicos organizados por categoria pedagógica,
    com análises pré-computadas e explicações teóricas.
    """
    
    def __init__(self):
        self.systems = self._initialize_example_systems()
        self.categories = self._get_categories()
        
    def _initialize_example_systems(self) -> Dict[str, Dict[str, Any]]:
        """Inicializa biblioteca de sistemas de exemplo"""
        
        systems = {
            # Sistemas de Primeira Ordem
            "first_order_basic": {
                "name": "Sistema de Primeira Ordem Básico",
                "transfer_function": "1/(s+1)",
                "description": "Sistema mais simples possível - resposta exponencial",
                "category": "primeira_ordem",
                "difficulty": 1,
                "learning_objectives": [
                    "Compreender resposta exponencial",
                    "Conceito de constante de tempo",
                    "Estabilidade inerente de primeira ordem"
                ],
                "theory_reference": "Cap. 4.2 - Sistemas de Primeira Ordem",
                "expected_characteristics": {
                    "poles": [-1],
                    "zeros": [],
                    "stable": True,
                    "time_constant": 1.0,
                    "settling_time": 4.0,
                    "overshoot": 0.0
                },
                "variations": {
                    "slow": "1/(s+0.5)",
                    "fast": "1/(s+5)",
                    "with_gain": "K/(s+1)"
                }
            },
            
            "first_order_with_zero": {
                "name": "Primeira Ordem com Zero",
                "transfer_function": "(s+2)/(s+1)",
                "description": "Sistema de primeira ordem com zero - introduz derivativo",
                "category": "primeira_ordem",
                "difficulty": 2,
                "learning_objectives": [
                    "Efeito do zero na resposta",
                    "Ação derivativa",
                    "Overshoot em primeira ordem"
                ],
                "theory_reference": "Cap. 5.3 - Efeito de Zeros",
                "expected_characteristics": {
                    "poles": [-1],
                    "zeros": [-2],
                    "stable": True,
                    "overshoot_expected": True
                }
            },
            
            # Sistemas de Segunda Ordem
            "second_order_underdamped": {
                "name": "Segunda Ordem Subamortecido",
                "transfer_function": "1/(s^2+2*s+2)",
                "description": "Sistema clássico com oscilação - ζ < 1",
                "category": "segunda_ordem",
                "difficulty": 3,
                "learning_objectives": [
                    "Resposta oscilatória",
                    "Conceitos de ζ e ωn",
                    "Overshoot e tempo de acomodação",
                    "Polos complexos conjugados"
                ],
                "theory_reference": "Cap. 4.4 - Sistemas de Segunda Ordem",
                "expected_characteristics": {
                    "poles": [-1+1j, -1-1j],
                    "zeros": [],
                    "stable": True,
                    "damping_ratio": 0.707,
                    "natural_frequency": 1.414,
                    "overshoot": 4.3,  # %
                    "settling_time": 4.0
                }
            },
            
            "second_order_critically_damped": {
                "name": "Segunda Ordem Criticamente Amortecido",
                "transfer_function": "1/(s^2+2*s+1)",
                "description": "Resposta mais rápida sem overshoot - ζ = 1",
                "category": "segunda_ordem", 
                "difficulty": 3,
                "learning_objectives": [
                    "Amortecimento crítico",
                    "Resposta sem overshoot",
                    "Polos duplos",
                    "Otimização de resposta"
                ],
                "expected_characteristics": {
                    "poles": [-1, -1],  # Polos duplos
                    "stable": True,
                    "damping_ratio": 1.0,
                    "overshoot": 0.0
                }
            },
            
            "second_order_overdamped": {
                "name": "Segunda Ordem Superamortecido",
                "transfer_function": "1/(s^2+3*s+2)",
                "description": "Resposta lenta sem oscilação - ζ > 1",
                "category": "segunda_ordem",
                "difficulty": 3,
                "learning_objectives": [
                    "Superamortecimento",
                    "Dois polos reais distintos",
                    "Resposta lenta",
                    "Trade-off velocidade vs overshoot"
                ],
                "expected_characteristics": {
                    "poles": [-1, -2],
                    "stable": True,
                    "damping_ratio": 1.5,
                    "overshoot": 0.0
                }
            },
            
            # Sistemas Instáveis
            "unstable_simple": {
                "name": "Sistema Instável Simples",
                "transfer_function": "1/(s-1)",
                "description": "Sistema de primeira ordem instável - polo no semiplano direito",
                "category": "instabilidade",
                "difficulty": 4,
                "learning_objectives": [
                    "Conceito de instabilidade",
                    "Polo no semiplano direito",
                    "Resposta divergente",
                    "Necessidade de compensação"
                ],
                "theory_reference": "Cap. 6.1 - Conceitos de Estabilidade",
                "expected_characteristics": {
                    "poles": [1],
                    "stable": False,
                    "response_type": "exponentially_growing"
                },
                "safety_note": "Sistema instável - apenas para análise teórica"
            },
            
            "marginally_stable": {
                "name": "Sistema Marginalmente Estável",
                "transfer_function": "1/(s^2+1)",
                "description": "Polos no eixo imaginário - oscilação sustentada",
                "category": "instabilidade",
                "difficulty": 4,
                "learning_objectives": [
                    "Estabilidade marginal",
                    "Polos no eixo jω",
                    "Oscilação sustentada",
                    "Limite entre estável e instável"
                ],
                "expected_characteristics": {
                    "poles": [1j, -1j],
                    "stable": False,  # Marginalmente instável
                    "response_type": "sustained_oscillation"
                }
            },
            
            # Sistemas de Controle Clássicos
            "dc_motor": {
                "name": "Motor DC com Carga",
                "transfer_function": "K/(s*(s+10))",
                "description": "Modelo simplificado de motor DC - sistema tipo 1",
                "category": "aplicacoes",
                "difficulty": 5,
                "learning_objectives": [
                    "Sistema tipo 1",
                    "Erro de regime nulo para degrau",
                    "Polo na origem",
                    "Aplicação prática"
                ],
                "theory_reference": "Cap. 7 - Erro de Regime",
                "expected_characteristics": {
                    "poles": [0, -10],
                    "stable": True,  # Marginalmente estável
                    "system_type": 1,
                    "steady_state_error_step": 0
                },
                "applications": ["Controle de posição", "Servo sistemas"],
                "variations": {
                    "with_compensation": "K*(s+5)/(s*(s+10))",
                    "higher_order": "K/(s*(s+10)*(s+50))"
                }
            },
            
            "aircraft_longitudinal": {
                "name": "Dinâmica Longitudinal de Aeronave",
                "transfer_function": "1.5*(s+0.5)/(s^2-0.1*s+1.2)",
                "description": "Modelo simplificado de controle de altitude",
                "category": "aplicacoes",
                "difficulty": 6,
                "learning_objectives": [
                    "Sistema de alta ordem",
                    "Dinâmica complexa",
                    "Zero e polos complexos",
                    "Controle aeroespacial"
                ],
                "expected_characteristics": {
                    "poles": [0.05+1.095j, 0.05-1.095j],  # Complexos com parte real positiva
                    "zeros": [-0.5],
                    "stable": False,  # Instável sem controle
                    "requires_feedback": True
                }
            },
            
            # Sistemas com Atraso
            "system_with_delay": {
                "name": "Sistema com Atraso de Transporte",
                "transfer_function": "exp(-2*s)/(s+1)",
                "description": "Sistema de primeira ordem com atraso - aproximação de Padé",
                "category": "sistemas_especiais",
                "difficulty": 7,
                "learning_objectives": [
                    "Atraso de transporte",
                    "Aproximação de Padé",
                    "Efeito na estabilidade",
                    "Limitação de bandwidth"
                ],
                "theory_reference": "Cap. 11 - Sistemas com Atraso",
                "pade_approximation": "(-s+1)/(s+1) * 1/(s+1)",  # Padé 1ª ordem para e^(-2s)
                "expected_characteristics": {
                    "stable_without_delay": True,
                    "stability_reduced_with_delay": True
                }
            }
        }
        
        return systems
        
    def _get_categories(self) -> Dict[str, Dict[str, Any]]:
        """Retorna categorias de sistemas organizadas pedagogicamente"""
        
        return {
            "primeira_ordem": {
                "name": "Sistemas de Primeira Ordem",
                "description": "Sistemas mais simples - resposta exponencial",
                "difficulty": 1,
                "prerequisite_concepts": ["transformada_laplace", "funcao_transferencia"],
                "learning_sequence": 1
            },
            "segunda_ordem": {
                "name": "Sistemas de Segunda Ordem", 
                "description": "Sistemas com dinâmica oscilatória - conceitos fundamentais",
                "difficulty": 2,
                "prerequisite_concepts": ["primeira_ordem", "polos_complexos"],
                "learning_sequence": 2
            },
            "instabilidade": {
                "name": "Sistemas Instáveis",
                "description": "Exemplos de instabilidade - compreensão conceitual",
                "difficulty": 3,
                "prerequisite_concepts": ["estabilidade", "localizacao_polos"],
                "learning_sequence": 3
            },
            "aplicacoes": {
                "name": "Aplicações Práticas",
                "description": "Sistemas reais de engenharia",
                "difficulty": 4,
                "prerequisite_concepts": ["controle_realimentado", "especificacoes"],
                "learning_sequence": 4
            },
            "sistemas_especiais": {
                "name": "Sistemas Especiais",
                "description": "Casos avançados - atrasos, não-linearidades",
                "difficulty": 5,
                "prerequisite_concepts": ["analise_avancada"],
                "learning_sequence": 5
            }
        }
        
    def get_system(self, system_id: str) -> Optional[Dict[str, Any]]:
        """Retorna sistema específico"""
        return self.systems.get(system_id)
        
    def get_systems_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Retorna sistemas de uma categoria"""
        category_systems = []
        
        for system_id, system in self.systems.items():
            if system.get("category") == category:
                system_copy = system.copy()
                system_copy["id"] = system_id
                category_systems.append(system_copy)
                
        # Ordenar por dificuldade
        category_systems.sort(key=lambda x: x.get("difficulty", 0))
        return category_systems
        
    def get_systems_by_difficulty(self, max_difficulty: int = 10) -> List[Dict[str, Any]]:
        """Retorna sistemas até certo nível de dificuldade"""
        filtered_systems = []
        
        for system_id, system in self.systems.items():
            if system.get("difficulty", 0) <= max_difficulty:
                system_copy = system.copy()
                system_copy["id"] = system_id
                filtered_systems.append(system_copy)
                
        # Ordenar por dificuldade
        filtered_systems.sort(key=lambda x: x.get("difficulty", 0))
        return filtered_systems
        
    def search_systems(self, query: str) -> List[Dict[str, Any]]:
        """Busca sistemas por palavra-chave"""
        results = []
        query_lower = query.lower()
        
        for system_id, system in self.systems.items():
            # Buscar em nome, descrição e objetivos
            searchable_text = (
                system.get("name", "").lower() + " " +
                system.get("description", "").lower() + " " +
                " ".join(system.get("learning_objectives", []))
            ).lower()
            
            if query_lower in searchable_text:
                system_copy = system.copy()
                system_copy["id"] = system_id
                results.append(system_copy)
                
        return results
        
    def get_learning_path(self, target_system: str) -> List[str]:
        """Sugere sequência de aprendizagem para chegar ao sistema alvo"""
        target = self.systems.get(target_system)
        if not target:
            return []
            
        target_difficulty = target.get("difficulty", 0)
        target_category = target.get("category", "")
        
        # Buscar sistemas preparatórios
        prep_systems = []
        
        for system_id, system in self.systems.items():
            if (system.get("difficulty", 0) < target_difficulty and 
                system.get("category") == target_category):
                prep_systems.append(system_id)
                
        # Ordenar por dificuldade
        prep_systems.sort(key=lambda x: self.systems[x].get("difficulty", 0))
        prep_systems.append(target_system)
        
        return prep_systems
        
    def get_variations(self, system_id: str) -> Dict[str, str]:
        """Retorna variações de um sistema"""
        system = self.systems.get(system_id)
        if system:
            return system.get("variations", {})
        return {}
        
    def get_all_categories(self) -> Dict[str, Dict[str, Any]]:
        """Retorna todas as categorias"""
        return self.categories
        
    def get_system_statistics(self) -> Dict[str, Any]:
        """Retorna estatísticas da biblioteca"""
        stats = {
            "total_systems": len(self.systems),
            "categories": len(self.categories),
            "by_category": {},
            "by_difficulty": {},
            "has_variations": 0
        }
        
        # Por categoria
        for category in self.categories.keys():
            stats["by_category"][category] = len(self.get_systems_by_category(category))
            
        # Por dificuldade
        for system in self.systems.values():
            difficulty = system.get("difficulty", 0)
            stats["by_difficulty"][difficulty] = stats["by_difficulty"].get(difficulty, 0) + 1
            
        # Com variações
        for system in self.systems.values():
            if system.get("variations"):
                stats["has_variations"] += 1
                
        return stats


class VirtualKeyboardTemplates:
    """
    Templates para teclado virtual contextual.
    
    Fornece estruturas pré-definidas para facilitar entrada de
    equações complexas em função de transferência e controle.
    """
    
    def __init__(self):
        self.templates = self._initialize_templates()
        
    def _initialize_templates(self) -> Dict[str, Dict[str, Any]]:
        """Inicializa templates de teclado virtual"""
        
        return {
            # Templates básicos de função de transferência
            "tf_basic": {
                "name": "Função de Transferência Básica",
                "template": "K/(s+a)",
                "description": "Template mais simples - primeira ordem",
                "placeholders": {
                    "K": "Ganho",
                    "a": "Polo"
                },
                "example": "2/(s+3)",
                "category": "basico"
            },
            
            "tf_second_order": {
                "name": "Segunda Ordem",
                "template": "K/(s^2+2*zeta*wn*s+wn^2)",
                "description": "Sistema de segunda ordem canônico",
                "placeholders": {
                    "K": "Ganho",
                    "zeta": "Razão de amortecimento",
                    "wn": "Frequência natural"
                },
                "example": "1/(s^2+2*0.7*1*s+1^2)",
                "category": "segunda_ordem"
            },
            
            "tf_with_zeros": {
                "name": "Com Zeros",
                "template": "K*(s+z)/(s+p)",
                "description": "Sistema com zero e polo",
                "placeholders": {
                    "K": "Ganho",
                    "z": "Zero",
                    "p": "Polo"
                },
                "example": "5*(s+2)/(s+10)",
                "category": "zeros_polos"
            },
            
            "tf_higher_order": {
                "name": "Ordem Superior",
                "template": "K/((s+p1)*(s+p2)*(s+p3))",
                "description": "Sistema de terceira ordem",
                "placeholders": {
                    "K": "Ganho",
                    "p1": "Polo 1",
                    "p2": "Polo 2", 
                    "p3": "Polo 3"
                },
                "example": "10/((s+1)*(s+5)*(s+20))",
                "category": "ordem_superior"
            },
            
            # Templates de controladores
            "pid_controller": {
                "name": "Controlador PID",
                "template": "Kp + Ki/s + Kd*s",
                "description": "Controlador PID clássico",
                "placeholders": {
                    "Kp": "Ganho proporcional",
                    "Ki": "Ganho integral",
                    "Kd": "Ganho derivativo"
                },
                "example": "2 + 5/s + 0.1*s",
                "category": "controladores"
            },
            
            "lead_compensator": {
                "name": "Compensador de Avanço",
                "template": "K*(s+z)/(s+p)",
                "description": "Compensador lead (z < p)",
                "placeholders": {
                    "K": "Ganho",
                    "z": "Zero (menor)",
                    "p": "Polo (maior)"
                },
                "example": "3*(s+1)/(s+10)",
                "category": "controladores",
                "constraint": "z < p para avanço"
            },
            
            "lag_compensator": {
                "name": "Compensador de Atraso",
                "template": "K*(s+z)/(s+p)",
                "description": "Compensador lag (z > p)",
                "placeholders": {
                    "K": "Ganho",
                    "z": "Zero (maior)",
                    "p": "Polo (menor)"
                },
                "example": "0.5*(s+10)/(s+1)",
                "category": "controladores",
                "constraint": "z > p para atraso"
            },
            
            # Templates de sistemas especiais
            "integrator": {
                "name": "Integrador",
                "template": "K/s",
                "description": "Integrador puro - polo na origem",
                "placeholders": {
                    "K": "Ganho"
                },
                "example": "1/s",
                "category": "especiais"
            },
            
            "double_integrator": {
                "name": "Integrador Duplo",
                "template": "K/s^2",
                "description": "Dois polos na origem",
                "placeholders": {
                    "K": "Ganho"
                },
                "example": "1/s^2",
                "category": "especiais"
            },
            
            "type_1_system": {
                "name": "Sistema Tipo 1",
                "template": "K/(s*(s+a))",
                "description": "Um polo na origem + dinâmica",
                "placeholders": {
                    "K": "Ganho",
                    "a": "Polo dinâmico"
                },
                "example": "10/(s*(s+2))",
                "category": "tipos_sistema"
            },
            
            # Templates com delay (usando aproximação de Padé)
            "system_with_delay": {
                "name": "Sistema com Atraso",
                "template": "K*(-T*s+2)/(T*s+2)/(s+a)",
                "description": "Atraso aproximado por Padé 1ª ordem",
                "placeholders": {
                    "K": "Ganho",
                    "T": "Constante de atraso",
                    "a": "Polo do sistema"
                },
                "example": "1*(-0.5*s+2)/(0.5*s+2)/(s+1)",
                "category": "especiais",
                "note": "Aproximação de Padé para e^(-T*s)"
            }
        }
        
    def get_template(self, template_id: str) -> Optional[Dict[str, Any]]:
        """Retorna template específico"""
        return self.templates.get(template_id)
        
    def get_templates_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Retorna templates de uma categoria"""
        category_templates = []
        
        for template_id, template in self.templates.items():
            if template.get("category") == category:
                template_copy = template.copy()
                template_copy["id"] = template_id
                category_templates.append(template_copy)
                
        return category_templates
        
    def get_all_categories(self) -> List[str]:
        """Retorna todas as categorias de templates"""
        categories = set()
        for template in self.templates.values():
            categories.add(template.get("category", "outros"))
        return sorted(list(categories))
        
    def search_templates(self, query: str) -> List[Dict[str, Any]]:
        """Busca templates por palavra-chave"""
        results = []
        query_lower = query.lower()
        
        for template_id, template in self.templates.items():
            searchable_text = (
                template.get("name", "").lower() + " " +
                template.get("description", "").lower()
            )
            
            if query_lower in searchable_text:
                template_copy = template.copy()
                template_copy["id"] = template_id
                results.append(template_copy)
                
        return results
        
    def substitute_template(self, template_id: str, values: Dict[str, str]) -> Optional[str]:
        """Substitui placeholders no template com valores fornecidos"""
        template = self.templates.get(template_id)
        if not template:
            return None
            
        result = template["template"]
        placeholders = template.get("placeholders", {})
        
        for placeholder, description in placeholders.items():
            if placeholder in values:
                result = result.replace(placeholder, values[placeholder])
                
        return result
        
    def get_template_suggestions(self, partial_input: str) -> List[Dict[str, Any]]:
        """Sugere templates baseado em entrada parcial"""
        suggestions = []
        
        # Detectar padrões na entrada
        if "pid" in partial_input.lower():
            suggestions.extend(self.get_templates_by_category("controladores"))
        elif "/" in partial_input and "s" in partial_input:
            suggestions.extend(self.get_templates_by_category("basico"))
        elif "^2" in partial_input:
            suggestions.extend(self.get_templates_by_category("segunda_ordem"))
            
        return suggestions[:5]  # Limitar a 5 sugestões


class TutorialManager:
    """
    Gerenciador de tutoriais guiados integrado com backend ControlLab.
    
    Usa sistemas de exemplo e tutoria socrática para criar
    experiências de aprendizagem estruturadas.
    """
    
    def __init__(self):
        self.example_systems = ExampleSystems()
        self.keyboard_templates = VirtualKeyboardTemplates()
        
        # Tutoriais disponíveis
        self.tutorials = self._initialize_tutorials()
        
        # Integração com backend se disponível
        if BACKEND_AVAILABLE:
            self.stability_engine = StabilityAnalysisEngine()
        else:
            self.stability_engine = None
            
    def _initialize_tutorials(self) -> Dict[str, Dict[str, Any]]:
        """Inicializa tutoriais guiados"""
        
        return {
            "intro_stability": {
                "name": "Introdução à Análise de Estabilidade",
                "description": "Tutorial básico de estabilidade usando Routh-Hurwitz",
                "difficulty": 1,
                "duration_minutes": 15,
                "learning_objectives": [
                    "Compreender conceito de estabilidade",
                    "Aplicar critério de Routh-Hurwitz",
                    "Interpretar tabela de Routh"
                ],
                "prerequisite_concepts": ["funcao_transferencia", "polos"],
                "systems_used": ["first_order_basic", "second_order_underdamped", "unstable_simple"],
                "steps": [
                    {
                        "title": "Conceitos Fundamentais",
                        "type": "theory",
                        "content": "Um sistema é estável se todos os polos estão no semiplano esquerdo",
                        "interactive_elements": ["concept_quiz"]
                    },
                    {
                        "title": "Sistema Estável Simples",
                        "type": "analysis",
                        "system": "first_order_basic",
                        "analysis_type": "stability",
                        "guided_questions": [
                            "Onde está localizado o polo deste sistema?",
                            "O que isso significa para a estabilidade?"
                        ]
                    },
                    {
                        "title": "Aplicando Routh-Hurwitz",
                        "type": "hands_on",
                        "system": "second_order_underdamped",
                        "task": "Construir tabela de Routh",
                        "validation": "automated"
                    }
                ]
            },
            
            "controller_design_intro": {
                "name": "Introdução ao Projeto de Controladores",
                "description": "Primeiros passos no design de compensadores",
                "difficulty": 3,
                "duration_minutes": 25,
                "learning_objectives": [
                    "Compreender necessidade de compensação",
                    "Projetar compensador básico",
                    "Avaliar desempenho do sistema compensado"
                ],
                "prerequisite_concepts": ["estabilidade", "root_locus", "especificacoes"],
                "systems_used": ["dc_motor"],
                "steps": [
                    {
                        "title": "Análise do Sistema Não-Compensado",
                        "type": "analysis",
                        "system": "dc_motor",
                        "analysis_type": "complete"
                    },
                    {
                        "title": "Especificações de Desempenho",
                        "type": "theory",
                        "content": "Definir overshoot, tempo de acomodação, erro de regime"
                    },
                    {
                        "title": "Design do Compensador",
                        "type": "design",
                        "controller_type": "lead",
                        "guided": True
                    }
                ]
            }
        }
        
    def get_tutorial(self, tutorial_id: str) -> Optional[Dict[str, Any]]:
        """Retorna tutorial específico"""
        return self.tutorials.get(tutorial_id)
        
    def get_tutorials_by_difficulty(self, max_difficulty: int = 10) -> List[Dict[str, Any]]:
        """Retorna tutoriais até certo nível de dificuldade"""
        filtered_tutorials = []
        
        for tutorial_id, tutorial in self.tutorials.items():
            if tutorial.get("difficulty", 0) <= max_difficulty:
                tutorial_copy = tutorial.copy()
                tutorial_copy["id"] = tutorial_id
                filtered_tutorials.append(tutorial_copy)
                
        return filtered_tutorials
        
    def suggest_tutorial_path(self, student_level: str = "beginner") -> List[str]:
        """Sugere sequência de tutoriais baseada no nível do estudante"""
        
        level_mapping = {
            "beginner": 2,
            "intermediate": 4,
            "advanced": 10
        }
        
        max_difficulty = level_mapping.get(student_level, 2)
        available_tutorials = self.get_tutorials_by_difficulty(max_difficulty)
        
        # Ordenar por dificuldade
        available_tutorials.sort(key=lambda x: x.get("difficulty", 0))
        
        return [t["id"] for t in available_tutorials]
        
    async def execute_tutorial_step(self, tutorial_id: str, step_index: int, 
                                  student_input: Any = None) -> Dict[str, Any]:
        """Executa passo específico de um tutorial"""
        
        tutorial = self.tutorials.get(tutorial_id)
        if not tutorial:
            return {"error": "Tutorial não encontrado"}
            
        steps = tutorial.get("steps", [])
        if step_index >= len(steps):
            return {"error": "Passo inválido"}
            
        step = steps[step_index]
        step_type = step.get("type")
        
        result = {
            "tutorial_id": tutorial_id,
            "step_index": step_index,
            "step_title": step.get("title"),
            "step_type": step_type,
            "timestamp": datetime.now().isoformat()
        }
        
        if step_type == "theory":
            result["content"] = step.get("content")
            result["interactive_elements"] = step.get("interactive_elements", [])
            
        elif step_type == "analysis":
            system_id = step.get("system")
            if system_id and BACKEND_AVAILABLE:
                system = self.example_systems.get_system(system_id)
                if system:
                    # Executar análise usando backend
                    try:
                        tf_string = system["transfer_function"]
                        analysis_result = self.stability_engine.comprehensive_analysis(tf_string)
                        result["analysis_result"] = analysis_result
                        result["system_info"] = system
                        result["guided_questions"] = step.get("guided_questions", [])
                    except Exception as e:
                        result["error"] = f"Erro na análise: {str(e)}"
                        
        elif step_type == "hands_on":
            result["task"] = step.get("task")
            result["system"] = step.get("system")
            result["validation_type"] = step.get("validation", "manual")
            
            if student_input and step.get("validation") == "automated":
                # Validar resposta do estudante
                validation_result = await self._validate_student_response(
                    step, student_input
                )
                result["validation_result"] = validation_result
                
        elif step_type == "design":
            result["controller_type"] = step.get("controller_type")
            result["guided"] = step.get("guided", False)
            
            # Fornecer templates de controlador
            controller_templates = self.keyboard_templates.get_templates_by_category("controladores")
            result["available_templates"] = controller_templates
            
        return result
        
    async def _validate_student_response(self, step: Dict[str, Any], 
                                       student_input: Any) -> Dict[str, Any]:
        """Valida resposta do estudante em passo hands-on"""
        
        validation = {
            "correct": False,
            "feedback": "",
            "hints": [],
            "next_steps": []
        }
        
        # Implementação básica - expandível
        task = step.get("task", "").lower()
        
        if "routh" in task and isinstance(student_input, dict):
            # Validar construção de tabela de Routh
            if "table" in student_input:
                validation["correct"] = True
                validation["feedback"] = "Tabela de Routh construída corretamente!"
            else:
                validation["feedback"] = "Tabela incompleta. Revise os coeficientes."
                validation["hints"] = [
                    "Organize coeficientes por ordem decrescente de potência",
                    "Use fórmula de recorrência para linhas subsequentes"
                ]
                
        return validation
        
    def get_tutorial_progress(self, tutorial_id: str, completed_steps: List[int]) -> Dict[str, Any]:
        """Calcula progresso em um tutorial"""
        
        tutorial = self.tutorials.get(tutorial_id)
        if not tutorial:
            return {"error": "Tutorial não encontrado"}
            
        total_steps = len(tutorial.get("steps", []))
        completed_count = len(completed_steps)
        
        progress = {
            "tutorial_id": tutorial_id,
            "tutorial_name": tutorial.get("name"),
            "total_steps": total_steps,
            "completed_steps": completed_count,
            "progress_percentage": (completed_count / max(total_steps, 1)) * 100,
            "next_step": None,
            "estimated_time_remaining": 0
        }
        
        # Próximo passo
        if completed_count < total_steps:
            progress["next_step"] = completed_count
            
        # Tempo estimado restante
        duration_minutes = tutorial.get("duration_minutes", 0)
        if completed_count > 0:
            time_per_step = duration_minutes / total_steps
            remaining_steps = total_steps - completed_count
            progress["estimated_time_remaining"] = remaining_steps * time_per_step
            
        return progress


# Funções de conveniência
def get_example_systems() -> ExampleSystems:
    """Retorna instância da biblioteca de sistemas"""
    return ExampleSystems()


def get_virtual_keyboard() -> VirtualKeyboardTemplates:
    """Retorna instância dos templates de teclado"""
    return VirtualKeyboardTemplates()


def get_tutorial_manager() -> TutorialManager:
    """Retorna instância do gerenciador de tutoriais"""
    return TutorialManager()


# Exemplo de uso e teste
if __name__ == "__main__":
    import asyncio
    
    async def demo():
        """Demonstração do sistema de templates e exemplos"""
        print("📚 ControlLab Web - Templates & Examples Demo")
        print("=" * 50)
        
        # Biblioteca de sistemas
        examples = get_example_systems()
        
        print("Sistemas Disponíveis:")
        stats = examples.get_system_statistics()
        print(json.dumps(stats, indent=2))
        
        # Sistema específico
        first_order = examples.get_system("first_order_basic")
        print(f"\nSistema de Primeira Ordem:")
        print(f"Nome: {first_order['name']}")
        print(f"TF: {first_order['transfer_function']}")
        print(f"Objetivos: {first_order['learning_objectives']}")
        
        # Templates de teclado
        keyboard = get_virtual_keyboard()
        
        print(f"\nTemplates de Teclado:")
        categories = keyboard.get_all_categories()
        print(f"Categorias: {categories}")
        
        pid_template = keyboard.get_template("pid_controller")
        print(f"Template PID: {pid_template['template']}")
        
        # Substitución de template
        substituted = keyboard.substitute_template("tf_basic", {"K": "5", "a": "2"})
        print(f"Template substituído: {substituted}")
        
        # Gerenciador de tutoriais
        tutorial_mgr = get_tutorial_manager()
        
        print(f"\nTutoriais Disponíveis:")
        beginner_tutorials = tutorial_mgr.suggest_tutorial_path("beginner")
        print(f"Para iniciantes: {beginner_tutorials}")
        
        # Executar passo de tutorial
        if BACKEND_AVAILABLE:
            step_result = await tutorial_mgr.execute_tutorial_step("intro_stability", 0)
            print(f"Resultado do passo: {list(step_result.keys())}")
        else:
            print("Backend não disponível - tutorial limitado")
            
        print(f"\n✅ Demo de templates concluída!")
        
    # Executar demo
    asyncio.run(demo())
