"""
ControlLab Web - AI Tutor Socrático

Implementa um tutor virtual que usa metodologia socrática para ensino de controle,
integrado com o backend ControlLab existente.

Classes implementadas:
- SocraticTutor: Tutor principal com metodologia socrática
- ConceptGraph: Grafo de conceitos para navegação pedagógica  
- LearningSession: Sessão de aprendizagem personalizada
"""

import asyncio
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import random

# Importações do backend ControlLab
try:
    from ..analysis.stability_analysis import StabilityAnalysisEngine
    from ..core.symbolic_tf import SymbolicTransferFunction
    from ..core.history import HistoryManager
    from .analysis_maestro import AnalysisMaestro, ReActPlan
    
    BACKEND_AVAILABLE = True
except ImportError as e:
    print(f"Aviso: Backend ControlLab não disponível para AI Tutor: {e}")
    BACKEND_AVAILABLE = False


class ConceptGraph:
    """
    Grafo de conceitos de Engenharia de Controle para navegação pedagógica.
    
    Representa relações entre conceitos fundamentais e permite
    navegação inteligente baseada no conhecimento do estudante.
    """
    
    def __init__(self):
        self.concepts = {
            # Conceitos fundamentais
            "sistema_linear": {
                "name": "Sistema Linear",
                "description": "Sistema que obedece ao princípio da superposição",
                "prerequisites": [],
                "leads_to": ["funcao_transferencia", "resposta_temporal"],
                "difficulty": 1,
                "theory_chapter": "Cap. 2"
            },
            "funcao_transferencia": {
                "name": "Função de Transferência", 
                "description": "Razão entre transformada de Laplace da saída e entrada",
                "prerequisites": ["sistema_linear", "transformada_laplace"],
                "leads_to": ["polos_zeros", "estabilidade"],
                "difficulty": 2,
                "theory_chapter": "Cap. 5"
            },
            "transformada_laplace": {
                "name": "Transformada de Laplace",
                "description": "Transformação matemática do domínio tempo para frequência complexa",
                "prerequisites": [],
                "leads_to": ["funcao_transferencia", "resposta_frequencia"],
                "difficulty": 2,
                "theory_chapter": "Cap. 3"
            },
            "polos_zeros": {
                "name": "Polos e Zeros",
                "description": "Raízes do denominador (polos) e numerador (zeros) da função de transferência",
                "prerequisites": ["funcao_transferencia"],
                "leads_to": ["estabilidade", "resposta_temporal"],
                "difficulty": 3,
                "theory_chapter": "Cap. 5"
            },
            "estabilidade": {
                "name": "Estabilidade",
                "description": "Propriedade de um sistema convergir para valor finito",
                "prerequisites": ["polos_zeros"],
                "leads_to": ["routh_hurwitz", "root_locus", "margens_estabilidade"],
                "difficulty": 4,
                "theory_chapter": "Cap. 6"
            },
            "routh_hurwitz": {
                "name": "Critério de Routh-Hurwitz",
                "description": "Método algébrico para determinar estabilidade",
                "prerequisites": ["estabilidade"],
                "leads_to": ["analise_parametrica"],
                "difficulty": 4,
                "theory_chapter": "Cap. 6.2-6.4"
            },
            "root_locus": {
                "name": "Lugar Geométrico das Raízes",
                "description": "Análise gráfica da localização dos polos em função de parâmetros",
                "prerequisites": ["estabilidade", "polos_zeros"],
                "leads_to": ["projeto_controladores"],
                "difficulty": 5,
                "theory_chapter": "Cap. 8-9"
            },
            "resposta_frequencia": {
                "name": "Resposta em Frequência",
                "description": "Comportamento do sistema para entradas senoidais",
                "prerequisites": ["funcao_transferencia"],
                "leads_to": ["diagrama_bode", "diagrama_nyquist"],
                "difficulty": 4,
                "theory_chapter": "Cap. 10"
            },
            "diagrama_bode": {
                "name": "Diagrama de Bode",
                "description": "Representação gráfica de magnitude e fase vs frequência",
                "prerequisites": ["resposta_frequencia"],
                "leads_to": ["margens_estabilidade"],
                "difficulty": 4,
                "theory_chapter": "Cap. 10.3"
            },
            "margens_estabilidade": {
                "name": "Margens de Estabilidade",
                "description": "Margens de fase e ganho para quantificar robustez",
                "prerequisites": ["diagrama_bode", "estabilidade"],
                "leads_to": ["projeto_controladores"],
                "difficulty": 5,
                "theory_chapter": "Cap. 10.5"
            },
            "projeto_controladores": {
                "name": "Projeto de Controladores",
                "description": "Design de compensadores para atender especificações",
                "prerequisites": ["root_locus", "margens_estabilidade"],
                "leads_to": ["controlador_pid", "compensador_avanço"],
                "difficulty": 6,
                "theory_chapter": "Cap. 11-12"
            }
        }
        
        # Perguntas socráticas por conceito
        self.socratic_questions = {
            "estabilidade": [
                "O que significa dizer que um sistema é estável?",
                "Como você relaciona a localização dos polos com estabilidade?",
                "Por que todos os polos devem estar no semiplano esquerdo?",
                "Qual a diferença entre estabilidade matemática e física?"
            ],
            "routh_hurwitz": [
                "Por que verificamos primeiro as condições necessárias?",
                "O que representa cada linha da tabela de Routh?",
                "Como interpretamos mudanças de sinal na primeira coluna?",
                "Quando usamos o método de Routh vs. cálculo direto de raízes?"
            ],
            "funcao_transferencia": [
                "Por que modelamos sistemas com funções de transferência?",
                "Como a função de transferência se relaciona com a equação diferencial?",
                "Qual o significado físico dos polos e zeros?",
                "Quando a função de transferência é válida?"
            ]
        }
        
    def get_concept(self, concept_id: str) -> Optional[Dict[str, Any]]:
        """Retorna informações de um conceito"""
        return self.concepts.get(concept_id)
        
    def get_prerequisites(self, concept_id: str) -> List[str]:
        """Retorna pré-requisitos de um conceito"""
        concept = self.concepts.get(concept_id, {})
        return concept.get("prerequisites", [])
        
    def get_next_concepts(self, concept_id: str) -> List[str]:
        """Retorna conceitos que podem ser explorados após este"""
        concept = self.concepts.get(concept_id, {})
        return concept.get("leads_to", [])
        
    def get_socratic_questions(self, concept_id: str) -> List[str]:
        """Retorna perguntas socráticas para um conceito"""
        return self.socratic_questions.get(concept_id, [])
        
    def suggest_learning_path(self, known_concepts: List[str], target_concept: str) -> List[str]:
        """Sugere caminho de aprendizagem do conhecimento atual até o alvo"""
        # Implementação básica - pode ser expandida com algoritmos de grafos
        path = []
        
        target_info = self.concepts.get(target_concept, {})
        prerequisites = target_info.get("prerequisites", [])
        
        # Adicionar pré-requisitos não conhecidos
        for prereq in prerequisites:
            if prereq not in known_concepts:
                path.append(prereq)
                
        # Adicionar conceito alvo
        if target_concept not in known_concepts:
            path.append(target_concept)
            
        return path


class LearningSession:
    """
    Sessão de aprendizagem personalizada com acompanhamento do progresso.
    
    Mantém estado do aprendizado do estudante e adapta metodologia.
    """
    
    def __init__(self, student_id: str):
        self.student_id = student_id
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        
        # Estado do aprendizado
        self.known_concepts = set()  # Conceitos dominados
        self.learning_concepts = set()  # Conceitos em aprendizado
        self.struggling_concepts = set()  # Conceitos com dificuldade
        
        # Histórico de interações
        self.interaction_history = []
        self.current_topic = None
        self.learning_style = "balanced"  # visual, analytical, practical, balanced
        
        # Métricas de engajamento
        self.total_questions_asked = 0
        self.correct_answers = 0
        self.hints_requested = 0
        self.completion_rate = 0.0
        
    def add_interaction(self, interaction_type: str, content: str, result: str = ""):
        """Adiciona interação ao histórico"""
        self.last_activity = datetime.now()
        
        interaction = {
            "timestamp": datetime.now(),
            "type": interaction_type,  # question, answer, explanation, hint
            "content": content,
            "result": result,
            "topic": self.current_topic
        }
        
        self.interaction_history.append(interaction)
        
        # Limitar histórico
        if len(self.interaction_history) > 100:
            self.interaction_history = self.interaction_history[-100:]
            
    def update_concept_mastery(self, concept_id: str, mastery_level: str):
        """Atualiza nível de domínio de um conceito"""
        # Remove de outros conjuntos
        self.known_concepts.discard(concept_id)
        self.learning_concepts.discard(concept_id)
        self.struggling_concepts.discard(concept_id)
        
        # Adiciona no conjunto apropriado
        if mastery_level == "mastered":
            self.known_concepts.add(concept_id)
        elif mastery_level == "learning":
            self.learning_concepts.add(concept_id)
        elif mastery_level == "struggling":
            self.struggling_concepts.add(concept_id)
            
    def get_learning_summary(self) -> Dict[str, Any]:
        """Retorna resumo do aprendizado"""
        total_concepts = len(self.known_concepts) + len(self.learning_concepts) + len(self.struggling_concepts)
        
        return {
            "student_id": self.student_id,
            "session_duration": (datetime.now() - self.created_at).total_seconds() / 3600,
            "concepts_mastered": len(self.known_concepts),
            "concepts_learning": len(self.learning_concepts), 
            "concepts_struggling": len(self.struggling_concepts),
            "total_interactions": len(self.interaction_history),
            "completion_rate": len(self.known_concepts) / max(total_concepts, 1),
            "current_topic": self.current_topic,
            "learning_style": self.learning_style
        }


class SocraticTutor:
    """
    Tutor Virtual usando metodologia socrática integrado com backend ControlLab.
    
    Características:
    - Metodologia socrática: perguntas guiam descoberta
    - Integração com análises do backend
    - Adaptação ao nível do estudante
    - Explicações step-by-step personalizadas
    """
    
    def __init__(self):
        self.concept_graph = ConceptGraph()
        self.learning_sessions = {}  # student_id -> LearningSession
        
        # Integração com backend
        if BACKEND_AVAILABLE:
            self.analysis_maestro = AnalysisMaestro()
        else:
            self.analysis_maestro = None
            
        # Templates de respostas por estilo de aprendizagem
        self.response_templates = {
            "visual": {
                "stability": "Imagine os polos como pontos no plano complexo. Se todos estão à esquerda do eixo imaginário...",
                "routh": "Vamos construir a tabela de Routh visualmente, linha por linha..."
            },
            "analytical": {
                "stability": "Matematicamente, estabilidade requer que todas as raízes da equação característica tenham parte real negativa...",
                "routh": "O critério de Routh-Hurwitz estabelece condições algébricas necessárias e suficientes..."
            },
            "practical": {
                "stability": "Na prática, um sistema instável significa que sua saída cresce indefinidamente...",
                "routh": "O método de Routh é uma ferramenta prática para evitar resolver equações de alto grau..."
            }
        }
        
    async def create_learning_session(self, student_id: str) -> LearningSession:
        """Cria nova sessão de aprendizagem"""
        session = LearningSession(student_id)
        self.learning_sessions[student_id] = session
        return session
        
    async def get_learning_session(self, student_id: str) -> Optional[LearningSession]:
        """Recupera sessão de aprendizagem existente"""
        return self.learning_sessions.get(student_id)
        
    async def ask_socratic_question(self, student_id: str, concept: str, 
                                  context: str = "") -> Dict[str, Any]:
        """
        Faz pergunta socrática sobre um conceito.
        
        A metodologia socrática usa perguntas para guiar o estudante
        à descoberta do conhecimento ao invés de dar respostas diretas.
        """
        session = await self.get_learning_session(student_id)
        if not session:
            session = await self.create_learning_session(student_id)
            
        # Obter perguntas para o conceito
        questions = self.concept_graph.get_socratic_questions(concept)
        
        if not questions:
            return {
                "type": "error",
                "message": f"Conceito '{concept}' não encontrado",
                "available_concepts": list(self.concept_graph.concepts.keys())
            }
            
        # Selecionar pergunta baseada no histórico
        used_questions = [i["content"] for i in session.interaction_history 
                         if i["type"] == "question" and i["topic"] == concept]
        
        available_questions = [q for q in questions if q not in used_questions]
        
        if not available_questions:
            # Todas as perguntas foram usadas, reciclar
            available_questions = questions
            
        selected_question = random.choice(available_questions)
        
        # Personalizar pergunta baseada no estilo de aprendizagem
        learning_style = session.learning_style
        
        # Adicionar contexto se fornecido
        full_question = selected_question
        if context:
            full_question = f"Considerando que {context}, {selected_question.lower()}"
            
        # Registrar interação
        session.add_interaction("question", full_question, concept)
        session.current_topic = concept
        session.total_questions_asked += 1
        
        return {
            "type": "socratic_question",
            "concept": concept,
            "question": full_question,
            "learning_style": learning_style,
            "concept_info": self.concept_graph.get_concept(concept),
            "session_stats": session.get_learning_summary()
        }
        
    async def provide_guided_explanation(self, student_id: str, concept: str, 
                                       student_answer: str = "") -> Dict[str, Any]:
        """
        Fornece explicação guiada baseada na resposta do estudante.
        
        Usa metodologia socrática para construir compreensão gradualmente.
        """
        session = await self.get_learning_session(student_id)
        if not session:
            session = await self.create_learning_session(student_id)
            
        concept_info = self.concept_graph.get_concept(concept)
        if not concept_info:
            return {"type": "error", "message": f"Conceito '{concept}' não encontrado"}
            
        learning_style = session.learning_style
        
        # Avaliar resposta do estudante (implementação básica)
        answer_quality = self._evaluate_answer(concept, student_answer)
        
        # Gerar explicação baseada no estilo de aprendizagem e qualidade da resposta
        explanation = self._generate_explanation(concept, learning_style, answer_quality, student_answer)
        
        # Registrar interação
        session.add_interaction("answer", student_answer, answer_quality)
        session.add_interaction("explanation", explanation["content"], concept)
        
        # Atualizar domínio do conceito
        if answer_quality == "excellent":
            session.update_concept_mastery(concept, "mastered")
            session.correct_answers += 1
        elif answer_quality == "good":
            session.update_concept_mastery(concept, "learning")
        else:
            session.update_concept_mastery(concept, "struggling")
            
        return {
            "type": "guided_explanation",
            "concept": concept,
            "answer_quality": answer_quality,
            "explanation": explanation,
            "next_suggestions": self._suggest_next_steps(session, concept, answer_quality),
            "session_stats": session.get_learning_summary()
        }
        
    async def integrate_with_analysis(self, student_id: str, analysis_type: str, 
                                    tf_string: str) -> Dict[str, Any]:
        """
        Integra análise técnica com tutoria socrática.
        
        Combina resultados do backend ControlLab com explicações pedagógicas.
        """
        if not BACKEND_AVAILABLE:
            return {
                "type": "error",
                "message": "Backend ControlLab não disponível para análise integrada"
            }
            
        session = await self.get_learning_session(student_id)
        if not session:
            session = await self.create_learning_session(student_id)
            
        # Executar análise usando o maestro
        if analysis_type == "stability":
            analysis_result = await self.analysis_maestro.execute_stability_analysis(
                f"tutor_{student_id}", tf_string
            )
            
            # Converter resultado técnico em explicação pedagógica
            pedagogical_explanation = await self._create_pedagogical_explanation(
                session, "estabilidade", analysis_result
            )
            
            return {
                "type": "integrated_analysis",
                "analysis_type": analysis_type,
                "technical_result": analysis_result,
                "pedagogical_explanation": pedagogical_explanation,
                "socratic_followup": await self.ask_socratic_question(
                    student_id, "estabilidade", 
                    f"analisamos o sistema {tf_string}"
                )
            }
            
        else:
            return {
                "type": "error",
                "message": f"Tipo de análise '{analysis_type}' não implementado ainda"
            }
            
    def _evaluate_answer(self, concept: str, answer: str) -> str:
        """
        Avalia qualidade da resposta do estudante.
        
        Implementação básica - pode ser expandida com NLP.
        """
        if not answer or len(answer.strip()) < 10:
            return "insufficient"
            
        answer_lower = answer.lower()
        
        # Palavras-chave por conceito
        concept_keywords = {
            "estabilidade": ["polo", "esquerdo", "direito", "convergir", "divergir", "finito"],
            "routh_hurwitz": ["tabela", "coeficiente", "sinal", "mudança", "linha"],
            "funcao_transferencia": ["laplace", "entrada", "saída", "razão", "transferência"]
        }
        
        keywords = concept_keywords.get(concept, [])
        keyword_matches = sum(1 for keyword in keywords if keyword in answer_lower)
        
        if keyword_matches >= len(keywords) * 0.7:
            return "excellent"
        elif keyword_matches >= len(keywords) * 0.4:
            return "good"
        elif keyword_matches >= len(keywords) * 0.2:
            return "partial"
        else:
            return "needs_guidance"
            
    def _generate_explanation(self, concept: str, learning_style: str, 
                            answer_quality: str, student_answer: str) -> Dict[str, Any]:
        """Gera explicação personalizada baseada no estilo de aprendizagem"""
        
        concept_info = self.concept_graph.get_concept(concept)
        base_explanation = concept_info.get("description", "")
        
        # Personalizar por estilo
        style_templates = self.response_templates.get(learning_style, {})
        style_explanation = style_templates.get(concept, base_explanation)
        
        # Ajustar baseado na qualidade da resposta
        if answer_quality == "excellent":
            feedback = "Excelente! Você demonstra compreensão sólida do conceito."
            extension = "Vamos explorar aplicações mais avançadas..."
        elif answer_quality == "good":
            feedback = "Boa resposta! Você está no caminho certo."
            extension = "Vamos refinar alguns pontos..."
        elif answer_quality == "partial":
            feedback = "Você tocou em pontos importantes."
            extension = "Vamos expandir sua compreensão..."
        else:
            feedback = "Vamos construir a compreensão juntos."
            extension = "Começando pelos fundamentos..."
            
        return {
            "feedback": feedback,
            "content": style_explanation,
            "extension": extension,
            "theory_reference": concept_info.get("theory_chapter", ""),
            "learning_style": learning_style
        }
        
    def _suggest_next_steps(self, session: LearningSession, concept: str, 
                          answer_quality: str) -> List[str]:
        """Sugere próximos passos baseados no progresso"""
        suggestions = []
        
        if answer_quality in ["excellent", "good"]:
            # Sugerir conceitos mais avançados
            next_concepts = self.concept_graph.get_next_concepts(concept)
            for next_concept in next_concepts:
                if next_concept not in session.known_concepts:
                    concept_info = self.concept_graph.get_concept(next_concept)
                    suggestions.append(f"Explorar: {concept_info['name']}")
                    
        else:
            # Sugerir revisão de pré-requisitos
            prerequisites = self.concept_graph.get_prerequisites(concept)
            for prereq in prerequisites:
                if prereq not in session.known_concepts:
                    concept_info = self.concept_graph.get_concept(prereq)
                    suggestions.append(f"Revisar: {concept_info['name']}")
                    
        # Sugerir prática
        suggestions.append(f"Praticar: Resolver exercícios de {concept}")
        
        return suggestions[:3]  # Limitar a 3 sugestões
        
    async def _create_pedagogical_explanation(self, session: LearningSession, 
                                            concept: str, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Converte resultado técnico em explicação pedagógica"""
        
        if concept == "estabilidade" and analysis_result.get("status") == "COMPLETED":
            final_result = analysis_result.get("final_result", {})
            is_stable = final_result.get("is_stable", False)
            
            # Explicação baseada no resultado
            if is_stable:
                explanation = {
                    "conclusion": "O sistema é ESTÁVEL! 🎉",
                    "reasoning": "Todos os critérios de estabilidade foram satisfeitos.",
                    "learning_points": [
                        "Todos os polos estão no semiplano esquerdo",
                        "A tabela de Routh não apresenta mudanças de sinal",
                        "O sistema converge para um valor finito"
                    ],
                    "visualization_hint": "No gráfico, todos os polos aparecem à esquerda do eixo imaginário"
                }
            else:
                explanation = {
                    "conclusion": "O sistema é INSTÁVEL ⚠️",
                    "reasoning": "Pelo menos um critério de estabilidade foi violado.",
                    "learning_points": [
                        "Existe(m) polo(s) no semiplano direito",
                        "A tabela de Routh apresenta mudança(s) de sinal",
                        "O sistema tende a divergir"
                    ],
                    "visualization_hint": "No gráfico, pelo menos um polo aparece à direita do eixo imaginário"
                }
                
            return explanation
            
        return {"message": "Explicação pedagógica não disponível para este resultado"}
        
    async def get_learning_progress(self, student_id: str) -> Dict[str, Any]:
        """Retorna progresso detalhado do estudante"""
        session = await self.get_learning_session(student_id)
        if not session:
            return {"error": "Sessão não encontrada"}
            
        summary = session.get_learning_summary()
        
        # Adicionar análise detalhada
        progress_analysis = {
            "strengths": [],
            "areas_for_improvement": [],
            "recommended_path": []
        }
        
        # Analisar conceitos dominados
        for concept_id in session.known_concepts:
            concept_info = self.concept_graph.get_concept(concept_id)
            progress_analysis["strengths"].append(concept_info["name"])
            
        # Analisar dificuldades
        for concept_id in session.struggling_concepts:
            concept_info = self.concept_graph.get_concept(concept_id)
            progress_analysis["areas_for_improvement"].append(concept_info["name"])
            
        # Sugerir caminho de aprendizagem
        if session.current_topic:
            next_concepts = self.concept_graph.get_next_concepts(session.current_topic)
            for concept_id in next_concepts:
                if concept_id not in session.known_concepts:
                    concept_info = self.concept_graph.get_concept(concept_id)
                    progress_analysis["recommended_path"].append(concept_info["name"])
                    
        return {
            "summary": summary,
            "detailed_analysis": progress_analysis,
            "recent_interactions": session.interaction_history[-5:] if session.interaction_history else []
        }


# Função de conveniência
def create_tutor() -> SocraticTutor:
    """Cria e retorna instância do Socratic Tutor"""
    return SocraticTutor()


# Exemplo de uso e teste
if __name__ == "__main__":
    async def demo():
        """Demonstração do AI Tutor Socrático"""
        print("🎓 ControlLab Web - AI Tutor Socrático Demo")
        print("=" * 50)
        
        # Criar tutor
        tutor = create_tutor()
        
        # Criar sessão de aprendizagem
        session = await tutor.create_learning_session("demo_student")
        print(f"Sessão criada para: {session.student_id}")
        
        # Fazer pergunta socrática sobre estabilidade
        question_result = await tutor.ask_socratic_question("demo_student", "estabilidade")
        print(f"\nPergunta Socrática:")
        print(f"Conceito: {question_result.get('concept')}")
        print(f"Pergunta: {question_result.get('question')}")
        
        # Simular resposta do estudante
        student_answer = "Um sistema é estável quando todos os polos estão no semiplano esquerdo"
        
        explanation_result = await tutor.provide_guided_explanation(
            "demo_student", "estabilidade", student_answer
        )
        
        print(f"\nExplicação Guiada:")
        print(f"Qualidade da resposta: {explanation_result.get('answer_quality')}")
        print(f"Feedback: {explanation_result['explanation']['feedback']}")
        print(f"Próximos passos: {explanation_result.get('next_suggestions')}")
        
        # Progresso do estudante
        progress = await tutor.get_learning_progress("demo_student")
        print(f"\nProgresso do Estudante:")
        print(json.dumps(progress, indent=2, default=str))
        
        print("\n✅ Demo do Tutor concluída!")
        
    # Executar demo
    asyncio.run(demo())
