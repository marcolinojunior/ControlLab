"""
ControlLab Web - Analysis Maestro

Orquestrador principal da aplicação web que integra todas as análises do backend ControlLab
seguindo o padrão ReAct (Reasoning + Acting) para externalização do raciocínio.

Classes implementadas:
- AnalysisMaestro: Orquestrador principal
- AnalysisSession: Gerenciamento de sessão
- ReActPlan: Estrutura de planejamento ReAct
"""

import asyncio
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import sympy as sp

# Importações do backend ControlLab existente
try:
    from ..analysis.stability_analysis import StabilityAnalysisEngine, analyze_stability
    from ..analysis.frequency_response import FrequencyAnalyzer, FrequencyResponse
    from ..analysis.root_locus import RootLocusAnalyzer
    from ..core.symbolic_tf import SymbolicTransferFunction
    from ..core.controller_design import PIDController, LeadLagCompensator
    from ..core.history import HistoryManager
    from ..numerical import get_available_backends, check_numerical_dependencies
    
    BACKEND_AVAILABLE = True
except ImportError as e:
    print(f"Aviso: Backend ControlLab não totalmente disponível: {e}")
    BACKEND_AVAILABLE = False


class ReActPlan:
    """
    Estrutura de planejamento ReAct para externalização do raciocínio.
    
    Implementa o padrão ReAct (Reasoning + Acting) que externaliza o processo de
    pensamento antes da execução, reduzindo carga cognitiva para usuários com TDAH.
    """
    
    def __init__(self, goal: str):
        self.goal = goal
        self.steps = []
        self.current_step = 0
        self.created_at = datetime.now()
        self.status = "PLANNED"  # PLANNED, EXECUTING, COMPLETED, FAILED
        
    def add_step(self, action: str, reasoning: str, expected_result: str, 
                 backend_function: str = "", theory_reference: str = ""):
        """Adiciona um passo ao plano ReAct"""
        step = {
            "step_number": len(self.steps) + 1,
            "action": action,
            "reasoning": reasoning, 
            "expected_result": expected_result,
            "backend_function": backend_function,
            "theory_reference": theory_reference,
            "status": "PENDING",  # PENDING, EXECUTING, COMPLETED, FAILED
            "result": None,
            "execution_time": None
        }
        self.steps.append(step)
        
    def get_current_step(self):
        """Retorna o passo atual para execução"""
        if self.current_step < len(self.steps):
            return self.steps[self.current_step]
        return None
        
    def complete_step(self, result: Any, execution_time: float):
        """Marca passo atual como completo e avança"""
        if self.current_step < len(self.steps):
            self.steps[self.current_step]["status"] = "COMPLETED"
            self.steps[self.current_step]["result"] = result
            self.steps[self.current_step]["execution_time"] = execution_time
            self.current_step += 1
            
        if self.current_step >= len(self.steps):
            self.status = "COMPLETED"
            
    def fail_step(self, error: str):
        """Marca passo atual como falhado"""
        if self.current_step < len(self.steps):
            self.steps[self.current_step]["status"] = "FAILED"
            self.steps[self.current_step]["result"] = f"ERRO: {error}"
            self.status = "FAILED"
            
    def to_dict(self):
        """Converte plano para dicionário JSON-serializável"""
        return {
            "goal": self.goal,
            "steps": self.steps,
            "current_step": self.current_step,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "total_steps": len(self.steps)
        }


class AnalysisSession:
    """
    Gerenciamento de sessão de análise com persistência de contexto.
    
    Mantém estado da sessão para continuidade entre interações,
    essencial para usuários com TDAH que podem perder contexto.
    """
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        
        # Estado da sessão
        self.current_system = None  # Sistema sendo analisado
        self.analysis_history = []  # Histórico de análises
        self.active_plans = {}      # Planos ReAct ativos
        self.context_variables = {} # Variáveis de contexto
        
        # Cache de resultados para evitar recálculos
        self.results_cache = {}
        
        # Histórico pedagógico
        if BACKEND_AVAILABLE:
            self.history_manager = HistoryManager()
        else:
            self.history_manager = None
            
    def update_activity(self):
        """Atualiza timestamp da última atividade"""
        self.last_activity = datetime.now()
        
    def set_current_system(self, tf_string: str, description: str = ""):
        """Define o sistema atual para análise"""
        self.update_activity()
        
        try:
            # Parse da função de transferência
            s = sp.Symbol('s')
            
            # Conversão simplificada (pode ser expandida)
            if BACKEND_AVAILABLE:
                # Usar parser do backend quando disponível
                self.current_system = {
                    "tf_string": tf_string,
                    "description": description,
                    "parsed_at": datetime.now(),
                    "valid": True
                }
            else:
                # Fallback básico
                self.current_system = {
                    "tf_string": tf_string,
                    "description": description,
                    "parsed_at": datetime.now(),
                    "valid": False,
                    "error": "Backend não disponível"
                }
                
        except Exception as e:
            self.current_system = {
                "tf_string": tf_string,
                "description": description,
                "parsed_at": datetime.now(),
                "valid": False,
                "error": str(e)
            }
            
    def add_analysis(self, analysis_type: str, result: Any):
        """Adiciona resultado de análise ao histórico"""
        self.update_activity()
        
        analysis_entry = {
            "timestamp": datetime.now(),
            "type": analysis_type,
            "result": result,
            "system": self.current_system
        }
        
        self.analysis_history.append(analysis_entry)
        
        # Limitar histórico para evitar sobrecarga
        if len(self.analysis_history) > 50:
            self.analysis_history = self.analysis_history[-50:]
            
    def get_session_summary(self) -> Dict[str, Any]:
        """Retorna resumo da sessão atual"""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "current_system": self.current_system,
            "total_analyses": len(self.analysis_history),
            "active_plans": len(self.active_plans),
            "backend_available": BACKEND_AVAILABLE
        }


class AnalysisMaestro:
    """
    Orquestrador principal da aplicação web ControlLab.
    
    Integra todas as análises do backend existente seguindo a filosofia
    "Anti-Caixa-Preta" com transparência total do processo matemático.
    
    Características:
    - Padrão ReAct para externalização do raciocínio
    - Integração com classes existentes do backend
    - Sistema de sessões para continuidade
    - Cache inteligente para performance
    """
    
    def __init__(self):
        self.sessions = {}  # session_id -> AnalysisSession
        self.active_connections = {}  # WebSocket connections
        
        # Inicialização dos analisadores do backend
        if BACKEND_AVAILABLE:
            self.stability_engine = StabilityAnalysisEngine()
            self.frequency_analyzer = FrequencyAnalyzer()
            self.root_locus_analyzer = RootLocusAnalyzer()
            
            # Verificar dependências numéricas
            self.numerical_backends = get_available_backends()
            self.numerical_dependencies = check_numerical_dependencies()
        else:
            print("⚠️ Backend ControlLab não disponível - modo demonstração")
            self.stability_engine = None
            self.frequency_analyzer = None
            self.root_locus_analyzer = None
            self.numerical_backends = {}
            self.numerical_dependencies = {}
            
    async def create_session(self, session_id: str) -> AnalysisSession:
        """Cria nova sessão de análise"""
        session = AnalysisSession(session_id)
        self.sessions[session_id] = session
        return session
        
    async def get_session(self, session_id: str) -> Optional[AnalysisSession]:
        """Recupera sessão existente"""
        return self.sessions.get(session_id)
        
    async def cleanup_sessions(self, max_age_hours: int = 24):
        """Remove sessões antigas para limpeza de memória"""
        current_time = datetime.now()
        expired_sessions = []
        
        for session_id, session in self.sessions.items():
            age = (current_time - session.last_activity).total_seconds() / 3600
            if age > max_age_hours:
                expired_sessions.append(session_id)
                
        for session_id in expired_sessions:
            del self.sessions[session_id]
            
        return len(expired_sessions)
        
    async def create_stability_analysis_plan(self, tf_string: str) -> ReActPlan:
        """
        Cria plano ReAct para análise de estabilidade completa.
        
        Integra com o backend existente: StabilityAnalysisEngine.comprehensive_analysis()
        """
        plan = ReActPlan(f"Análise Completa de Estabilidade: {tf_string}")
        
        # Passo 1: Validação e parsing
        plan.add_step(
            action="Validar e parsear função de transferência",
            reasoning="Primeiro precisamos garantir que a função de transferência está bem formada e extrair o polinômio característico",
            expected_result="Função de transferência válida e polinômio característico extraído",
            backend_function="SymbolicTransferFunction.from_string()",
            theory_reference="Cap. 5 - Modelagem de Sistemas"
        )
        
        # Passo 2: Condições necessárias de estabilidade
        plan.add_step(
            action="Verificar condições necessárias de estabilidade",
            reasoning="Antes de aplicar Routh-Hurwitz, verificamos se todos os coeficientes são positivos e estão presentes",
            expected_result="Validação das condições necessárias (todos coeficientes positivos)",
            backend_function="RouthHurwitzAnalyzer.check_necessary_conditions()",
            theory_reference="Cap. 6.2 - Condições Necessárias"
        )
        
        # Passo 3: Análise de Routh-Hurwitz
        plan.add_step(
            action="Construir tabela de Routh-Hurwitz",
            reasoning="Aplicamos o critério algébrico de Routh-Hurwitz para determinar estabilidade sem calcular raízes",
            expected_result="Tabela de Routh completa e análise de sinais",
            backend_function="RouthHurwitzAnalyzer.build_routh_table()",
            theory_reference="Cap. 6.2-6.4 - Critério de Routh-Hurwitz"
        )
        
        # Passo 4: Análise de Root Locus (se disponível)
        if BACKEND_AVAILABLE:
            plan.add_step(
                action="Análise de Root Locus",
                reasoning="Validamos resultado de Routh com análise gráfica da localização dos polos",
                expected_result="Localização dos polos e concordância com análise de Routh",
                backend_function="RootLocusAnalyzer.analyze()",
                theory_reference="Cap. 8-9 - Lugar Geométrico das Raízes"
            )
            
        # Passo 5: Análise de margens de estabilidade (se numérico disponível)
        if self.numerical_dependencies.get('SCIPY_AVAILABLE', False):
            plan.add_step(
                action="Calcular margens de estabilidade",
                reasoning="Complementamos análise algébrica com margens de fase e ganho usando resposta em frequência",
                expected_result="Margens de fase e ganho calculadas",
                backend_function="FrequencyAnalyzer.calculate_stability_margins()",
                theory_reference="Cap. 10 - Resposta em Frequência"
            )
            
        # Passo 6: Relatório de validação cruzada
        plan.add_step(
            action="Validação cruzada entre métodos",
            reasoning="Comparamos resultados dos diferentes métodos para garantir consistência",
            expected_result="Relatório de concordância entre métodos",
            backend_function="StabilityAnalysisEngine.cross_validate_results()",
            theory_reference="Metodologia multi-critério"
        )
        
        return plan
        
    async def execute_stability_analysis(self, session_id: str, tf_string: str) -> Dict[str, Any]:
        """
        Executa análise de estabilidade completa usando o backend existente.
        
        Retorna resultados streaming para a interface web.
        """
        session = await self.get_session(session_id)
        if not session:
            session = await self.create_session(session_id)
            
        # Criar plano ReAct
        plan = await self.create_stability_analysis_plan(tf_string)
        session.active_plans["stability"] = plan
        
        # Definir sistema atual
        session.set_current_system(tf_string, "Análise de Estabilidade")
        
        results = {
            "plan": plan.to_dict(),
            "steps_results": [],
            "final_result": None,
            "status": "EXECUTING"
        }
        
        if not BACKEND_AVAILABLE:
            results["status"] = "ERROR"
            results["error"] = "Backend ControlLab não disponível"
            return results
            
        try:
            # Executar análise usando o backend existente
            start_time = datetime.now()
            
            # Análise completa usando StabilityAnalysisEngine
            stability_result = self.stability_engine.comprehensive_analysis(
                tf_string,
                show_all_steps=True,
                include_validation=True
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Estruturar resultado para a web
            final_result = {
                "is_stable": getattr(stability_result, 'is_stable', False),
                "method_results": {
                    "routh_hurwitz": getattr(stability_result, 'routh_result', None),
                    "root_locus": getattr(stability_result, 'root_locus_result', None),
                    "frequency": getattr(stability_result, 'frequency_result', None)
                },
                "executive_summary": stability_result.get_executive_summary() if hasattr(stability_result, 'get_executive_summary') else "Análise completa",
                "detailed_analysis": stability_result.get_detailed_analysis() if hasattr(stability_result, 'get_detailed_analysis') else "",
                "educational_notes": stability_result.get_educational_section() if hasattr(stability_result, 'get_educational_section') else "",
                "execution_time": execution_time
            }
            
            # Completar plano
            plan.complete_step(final_result, execution_time)
            
            # Adicionar ao histórico da sessão
            session.add_analysis("stability", final_result)
            
            results["final_result"] = final_result
            results["status"] = "COMPLETED"
            
        except Exception as e:
            error_msg = f"Erro na análise de estabilidade: {str(e)}"
            plan.fail_step(error_msg)
            results["status"] = "ERROR"
            results["error"] = error_msg
            
        return results
        
    async def handle_natural_language_command(self, session_id: str, command: str) -> Dict[str, Any]:
        """
        Processa comando em linguagem natural e determina análise apropriada.
        
        Esta função implementará NLP básico para interpretar comandos como:
        - "analise a estabilidade de K/(s*(s+4))"
        - "desenhe o lugar das raízes"
        - "calcule margens de estabilidade"
        """
        session = await self.get_session(session_id)
        if not session:
            session = await self.create_session(session_id)
            
        # NLP básico usando palavras-chave
        command_lower = command.lower()
        
        # Detectar tipo de análise
        if any(word in command_lower for word in ['estabilidade', 'routh', 'estável', 'instável']):
            # Extrair função de transferência do comando
            tf_match = self._extract_transfer_function(command)
            if tf_match:
                return await self.execute_stability_analysis(session_id, tf_match)
            else:
                return {
                    "status": "ERROR",
                    "error": "Não foi possível extrair função de transferência do comando",
                    "suggestion": "Tente: 'analise a estabilidade de K/(s*(s+4))'"
                }
                
        elif any(word in command_lower for word in ['root locus', 'lugar', 'raízes', 'polos']):
            return {
                "status": "NOT_IMPLEMENTED",
                "message": "Análise de Root Locus será implementada na próxima versão"
            }
            
        elif any(word in command_lower for word in ['bode', 'frequência', 'margens']):
            return {
                "status": "NOT_IMPLEMENTED", 
                "message": "Análise de resposta em frequência será implementada na próxima versão"
            }
            
        else:
            return {
                "status": "UNKNOWN_COMMAND",
                "message": "Comando não reconhecido",
                "available_commands": [
                    "analise a estabilidade de [função]",
                    "root locus de [função]",
                    "resposta em frequência de [função]"
                ]
            }
            
    def _extract_transfer_function(self, text: str) -> Optional[str]:
        """
        Extrai função de transferência de texto em linguagem natural.
        
        Implementação básica - pode ser expandida com regex mais sofisticado.
        """
        import re
        
        # Padrões básicos para funções de transferência
        patterns = [
            r'([KkGg]?\s*/\s*\([^)]+\))',  # K/(s+1)
            r'(\([^)]+\)\s*/\s*\([^)]+\))',  # (s+1)/(s+2)
            r'(\d+\s*/\s*\([^)]+\))',  # 1/(s+1)
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
                
        return None
        
    async def get_available_analyses(self) -> Dict[str, Any]:
        """Retorna análises disponíveis baseadas no backend"""
        analyses = {
            "stability": {
                "name": "Análise de Estabilidade",
                "description": "Critério de Routh-Hurwitz, Root Locus, Margens",
                "available": BACKEND_AVAILABLE,
                "methods": ["routh_hurwitz", "root_locus", "frequency_margins"]
            },
            "frequency_response": {
                "name": "Resposta em Frequência", 
                "description": "Diagramas de Bode, Nyquist, Margens",
                "available": BACKEND_AVAILABLE and self.numerical_dependencies.get('SCIPY_AVAILABLE', False),
                "methods": ["bode", "nyquist", "margins"]
            },
            "controller_design": {
                "name": "Projeto de Controladores",
                "description": "PID, Lead-Lag, State Space",
                "available": BACKEND_AVAILABLE,
                "methods": ["pid", "lead_lag", "state_space"]
            },
            "time_response": {
                "name": "Resposta Temporal",
                "description": "Step, Impulse, Ramp Response",
                "available": self.numerical_dependencies.get('SCIPY_AVAILABLE', False),
                "methods": ["step", "impulse", "ramp"]
            }
        }
        
        return {
            "analyses": analyses,
            "backend_status": {
                "controllab_available": BACKEND_AVAILABLE,
                "numerical_backends": self.numerical_backends,
                "dependencies": self.numerical_dependencies
            }
        }
        
    async def get_system_status(self) -> Dict[str, Any]:
        """Status geral do sistema para monitoramento"""
        return {
            "timestamp": datetime.now().isoformat(),
            "active_sessions": len(self.sessions),
            "backend_available": BACKEND_AVAILABLE,
            "numerical_backends": self.numerical_backends,
            "memory_usage": {
                "sessions": len(self.sessions),
                "total_analyses": sum(len(s.analysis_history) for s in self.sessions.values())
            }
        }


# Função de conveniência para criar instância
def create_maestro() -> AnalysisMaestro:
    """Cria e retorna instância do Analysis Maestro"""
    return AnalysisMaestro()


# Exemplo de uso e teste
if __name__ == "__main__":
    async def demo():
        """Demonstração básica do Analysis Maestro"""
        print("🎭 ControlLab Web - Analysis Maestro Demo")
        print("=" * 50)
        
        # Criar maestro
        maestro = create_maestro()
        
        # Status do sistema
        status = await maestro.get_system_status()
        print(f"Status do Sistema: {json.dumps(status, indent=2)}")
        
        # Análises disponíveis
        analyses = await maestro.get_available_analyses()
        print(f"\nAnálises Disponíveis: {json.dumps(analyses, indent=2)}")
        
        # Criar sessão de teste
        session = await maestro.create_session("demo_session")
        print(f"\nSessão criada: {session.session_id}")
        
        # Teste de comando em linguagem natural
        if BACKEND_AVAILABLE:
            result = await maestro.handle_natural_language_command(
                "demo_session", 
                "analise a estabilidade de 1/(s*(s+4))"
            )
            print(f"\nResultado da análise: {json.dumps(result, indent=2, default=str)}")
        else:
            print("\n⚠️ Backend não disponível - teste limitado")
            
        print("\n✅ Demo concluída!")
        
    # Executar demo se chamado diretamente
    asyncio.run(demo())
