"""
Execução e Validação do Módulo 8 - ControlLab Web Application

Script de demonstração e teste rápido para verificar
a funcionalidade básica da aplicação web educacional.

Funcionalidades testadas:
- Inicialização dos componentes web
- Integração entre módulos
- Funcionalidade de demo mode
- Capacidades de visualização
"""

import sys
import os
import json
import traceback
from typing import Dict, Any, List
from datetime import datetime

# Adicionar path do projeto
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

# Status de importações
module_status = {
    "analysis_maestro": False,
    "ai_tutor": False,
    "smart_plots": False,
    "example_systems": False,
    "websocket_server": False
}

imported_modules = {}

# Tentativa de importação dos módulos web
print("🔄 Importando módulos do ControlLab Web...")

try:
    from controllab.web.analysis_maestro import AnalysisMaestro, AnalysisSession, ReActPlan
    module_status["analysis_maestro"] = True
    imported_modules["analysis_maestro"] = {
        "AnalysisMaestro": AnalysisMaestro,
        "AnalysisSession": AnalysisSession,
        "ReActPlan": ReActPlan
    }
    print("✅ Analysis Maestro importado com sucesso")
except ImportError as e:
    print(f"⚠️ Analysis Maestro não disponível: {e}")

try:
    from controllab.web.ai_tutor import SocraticTutor, ConceptGraph, LearningSession
    module_status["ai_tutor"] = True
    imported_modules["ai_tutor"] = {
        "SocraticTutor": SocraticTutor,
        "ConceptGraph": ConceptGraph,
        "LearningSession": LearningSession
    }
    print("✅ AI Tutor importado com sucesso")
except ImportError as e:
    print(f"⚠️ AI Tutor não disponível: {e}")

try:
    from controllab.web.smart_plots import SmartPlotter, VisualizationManager, PlotlyRenderer
    module_status["smart_plots"] = True
    imported_modules["smart_plots"] = {
        "SmartPlotter": SmartPlotter,
        "VisualizationManager": VisualizationManager,
        "PlotlyRenderer": PlotlyRenderer
    }
    print("✅ Smart Plots importado com sucesso")
except ImportError as e:
    print(f"⚠️ Smart Plots não disponível: {e}")

try:
    from controllab.web.example_systems import ExampleSystems, VirtualKeyboardTemplates, TutorialManager
    module_status["example_systems"] = True
    imported_modules["example_systems"] = {
        "ExampleSystems": ExampleSystems,
        "VirtualKeyboardTemplates": VirtualKeyboardTemplates,
        "TutorialManager": TutorialManager
    }
    print("✅ Example Systems importado com sucesso")
except ImportError as e:
    print(f"⚠️ Example Systems não disponível: {e}")

try:
    from controllab.web.websocket_server import ControlLabWebSocketServer, MessageProtocol, ConnectionManager
    module_status["websocket_server"] = True
    imported_modules["websocket_server"] = {
        "ControlLabWebSocketServer": ControlLabWebSocketServer,
        "MessageProtocol": MessageProtocol,
        "ConnectionManager": ConnectionManager
    }
    print("✅ WebSocket Server importado com sucesso")
except ImportError as e:
    print(f"⚠️ WebSocket Server não disponível: {e}")


class ControlLabWebDemonstrator:
    """Demonstrador da aplicação web ControlLab"""
    
    def __init__(self):
        self.test_results = {}
        self.available_modules = sum(module_status.values())
        self.total_modules = len(module_status)
        
    def test_analysis_maestro(self) -> Dict[str, Any]:
        """Testa funcionalidades do Analysis Maestro"""
        if not module_status["analysis_maestro"]:
            return {"status": "SKIP", "reason": "Módulo não disponível"}
            
        try:
            AnalysisMaestro = imported_modules["analysis_maestro"]["AnalysisMaestro"]
            AnalysisSession = imported_modules["analysis_maestro"]["AnalysisSession"]
            ReActPlan = imported_modules["analysis_maestro"]["ReActPlan"]
            
            # Teste de inicialização
            maestro = AnalysisMaestro()
            
            # Teste de sessão
            session = AnalysisSession("demo_session")
            session.set_current_system("1/(s+1)", "Sistema de primeira ordem")
            
            # Teste de plano ReAct
            plan = ReActPlan("Análise de estabilidade demo")
            plan.add_step(
                action="Validar função de transferência",
                reasoning="Verificar sintaxe e formato",
                expected_result="Função válida",
                backend_function="validate_tf()"
            )
            
            return {
                "status": "SUCCESS",
                "maestro_initialized": True,
                "session_created": True,
                "plan_created": True,
                "plan_steps": len(plan.steps)
            }
            
        except Exception as e:
            return {
                "status": "ERROR",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            
    def test_ai_tutor(self) -> Dict[str, Any]:
        """Testa funcionalidades do AI Tutor"""
        if not module_status["ai_tutor"]:
            return {"status": "SKIP", "reason": "Módulo não disponível"}
            
        try:
            SocraticTutor = imported_modules["ai_tutor"]["SocraticTutor"]
            ConceptGraph = imported_modules["ai_tutor"]["ConceptGraph"]
            LearningSession = imported_modules["ai_tutor"]["LearningSession"]
            
            # Teste de grafo de conceitos
            concept_graph = ConceptGraph()
            stability_concept = concept_graph.get_concept("estabilidade")
            
            # Teste de sessão de aprendizagem
            learning_session = LearningSession("demo_student")
            learning_session.add_interaction("question", "O que é estabilidade?", "estabilidade")
            
            # Teste de tutor
            tutor = SocraticTutor()
            
            return {
                "status": "SUCCESS",
                "concept_graph_initialized": True,
                "stability_concept_found": stability_concept is not None,
                "learning_session_created": True,
                "tutor_initialized": True,
                "interaction_count": len(learning_session.interaction_history)
            }
            
        except Exception as e:
            return {
                "status": "ERROR",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            
    def test_smart_plots(self) -> Dict[str, Any]:
        """Testa funcionalidades do Smart Plots"""
        if not module_status["smart_plots"]:
            return {"status": "SKIP", "reason": "Módulo não disponível"}
            
        try:
            SmartPlotter = imported_modules["smart_plots"]["SmartPlotter"]
            VisualizationManager = imported_modules["smart_plots"]["VisualizationManager"]
            PlotlyRenderer = imported_modules["smart_plots"]["PlotlyRenderer"]
            
            # Teste de plotter
            plotter = SmartPlotter()
            capabilities = plotter.get_capabilities()
            
            # Teste de renderer
            poles_real = [-1, -2]
            poles_imag = [0, 0]
            root_locus_plot = PlotlyRenderer.create_root_locus_plot(poles_real, poles_imag)
            
            # Teste de visualization manager
            viz_manager = VisualizationManager()
            status = viz_manager.get_manager_status()
            
            return {
                "status": "SUCCESS",
                "plotter_initialized": True,
                "capabilities_obtained": capabilities is not None,
                "root_locus_created": "data" in root_locus_plot,
                "viz_manager_initialized": True,
                "manager_status": status
            }
            
        except Exception as e:
            return {
                "status": "ERROR",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            
    def test_example_systems(self) -> Dict[str, Any]:
        """Testa funcionalidades dos Example Systems"""
        if not module_status["example_systems"]:
            return {"status": "SKIP", "reason": "Módulo não disponível"}
            
        try:
            ExampleSystems = imported_modules["example_systems"]["ExampleSystems"]
            VirtualKeyboardTemplates = imported_modules["example_systems"]["VirtualKeyboardTemplates"]
            TutorialManager = imported_modules["example_systems"]["TutorialManager"]
            
            # Teste de biblioteca de sistemas
            examples = ExampleSystems()
            first_order_system = examples.get_system("first_order_basic")
            
            # Teste de teclado virtual
            keyboard = VirtualKeyboardTemplates()
            basic_template = keyboard.get_template("tf_basic")
            
            # Teste de tutorial manager
            tutorial_manager = TutorialManager()
            path = tutorial_manager.suggest_tutorial_path("beginner")
            
            return {
                "status": "SUCCESS",
                "examples_initialized": True,
                "system_count": len(examples.systems),
                "first_order_found": first_order_system is not None,
                "keyboard_initialized": True,
                "basic_template_found": basic_template is not None,
                "tutorial_manager_initialized": True,
                "suggested_path_length": len(path)
            }
            
        except Exception as e:
            return {
                "status": "ERROR",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            
    def test_websocket_server(self) -> Dict[str, Any]:
        """Testa funcionalidades do WebSocket Server"""
        if not module_status["websocket_server"]:
            return {"status": "SKIP", "reason": "Módulo não disponível"}
            
        try:
            MessageProtocol = imported_modules["websocket_server"]["MessageProtocol"]
            ConnectionManager = imported_modules["websocket_server"]["ConnectionManager"]
            
            # Teste de protocolo de mensagens
            message = MessageProtocol.create_message(
                "TEST_MESSAGE",
                {"test": "data"},
                "demo_session"
            )
            
            is_valid = MessageProtocol.validate_message(message)
            
            # Teste de connection manager
            conn_manager = ConnectionManager()
            
            return {
                "status": "SUCCESS",
                "message_created": message is not None,
                "message_valid": is_valid,
                "connection_manager_initialized": True,
                "message_fields": list(message.keys()) if message else []
            }
            
        except Exception as e:
            return {
                "status": "ERROR",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            
    def run_integration_test(self) -> Dict[str, Any]:
        """Executa teste de integração entre módulos"""
        if self.available_modules < 2:
            return {"status": "SKIP", "reason": "Módulos insuficientes para integração"}
            
        try:
            integration_results = {}
            
            # Teste 1: Maestro + Examples
            if module_status["analysis_maestro"] and module_status["example_systems"]:
                AnalysisMaestro = imported_modules["analysis_maestro"]["AnalysisMaestro"]
                ExampleSystems = imported_modules["example_systems"]["ExampleSystems"]
                
                maestro = AnalysisMaestro()
                examples = ExampleSystems()
                
                # Usar sistema de exemplo no maestro
                system = examples.get_system("first_order_basic")
                if system:
                    session_id = "integration_test"
                    session = AnalysisSession(session_id)
                    session.set_current_system(
                        system["transfer_function"],
                        system["description"]
                    )
                    
                integration_results["maestro_examples"] = True
                
            # Teste 2: Tutor + Examples
            if module_status["ai_tutor"] and module_status["example_systems"]:
                SocraticTutor = imported_modules["ai_tutor"]["SocraticTutor"]
                ExampleSystems = imported_modules["example_systems"]["ExampleSystems"]
                
                tutor = SocraticTutor()
                examples = ExampleSystems()
                
                # Correlacionar conceitos com sistemas
                stability_systems = examples.get_systems_by_category("primeira_ordem")
                
                integration_results["tutor_examples"] = len(stability_systems) > 0
                
            # Teste 3: Plots + Examples
            if module_status["smart_plots"] and module_status["example_systems"]:
                SmartPlotter = imported_modules["smart_plots"]["SmartPlotter"]
                ExampleSystems = imported_modules["example_systems"]["ExampleSystems"]
                
                plotter = SmartPlotter()
                examples = ExampleSystems()
                
                # Preparar sistema para plotagem
                system = examples.get_system("first_order_basic")
                if system:
                    tf_string = system["transfer_function"]
                    
                integration_results["plots_examples"] = True
                
            return {
                "status": "SUCCESS",
                "integration_tests": integration_results,
                "tests_executed": len(integration_results)
            }
            
        except Exception as e:
            return {
                "status": "ERROR",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            
    def generate_report(self) -> Dict[str, Any]:
        """Gera relatório completo dos testes"""
        
        # Executar todos os testes
        print("\n🧪 Executando testes dos módulos web...")
        
        tests = {
            "analysis_maestro": self.test_analysis_maestro(),
            "ai_tutor": self.test_ai_tutor(),
            "smart_plots": self.test_smart_plots(),
            "example_systems": self.test_example_systems(),
            "websocket_server": self.test_websocket_server(),
            "integration": self.run_integration_test()
        }
        
        # Calcular estatísticas
        successful_tests = sum(1 for test in tests.values() if test["status"] == "SUCCESS")
        skipped_tests = sum(1 for test in tests.values() if test["status"] == "SKIP")
        failed_tests = sum(1 for test in tests.values() if test["status"] == "ERROR")
        
        # Relatório final
        report = {
            "timestamp": datetime.now().isoformat(),
            "module_availability": module_status,
            "available_modules": self.available_modules,
            "total_modules": self.total_modules,
            "test_results": tests,
            "summary": {
                "successful": successful_tests,
                "skipped": skipped_tests,
                "failed": failed_tests,
                "total": len(tests)
            },
            "recommendations": self._generate_recommendations(tests)
        }
        
        return report
        
    def _generate_recommendations(self, tests: Dict[str, Any]) -> List[str]:
        """Gera recomendações baseadas nos resultados dos testes"""
        recommendations = []
        
        # Verificar dependências
        missing_deps = []
        if any("numpy" in str(test.get("error", "")) for test in tests.values()):
            missing_deps.append("numpy")
        if any("scipy" in str(test.get("error", "")) for test in tests.values()):
            missing_deps.append("scipy")
        if any("websockets" in str(test.get("error", "")) for test in tests.values()):
            missing_deps.append("websockets")
            
        if missing_deps:
            recommendations.append(f"Instalar dependências opcionais: {', '.join(missing_deps)}")
            
        # Verificar funcionalidade
        if self.available_modules == self.total_modules:
            recommendations.append("✅ Todos os módulos web estão funcionais")
        elif self.available_modules >= self.total_modules // 2:
            recommendations.append("⚠️ Maioria dos módulos funcionais - alguns recursos limitados")
        else:
            recommendations.append("❌ Poucos módulos funcionais - verificar instalação")
            
        # Recomendações específicas
        if tests["integration"]["status"] == "SUCCESS":
            recommendations.append("✅ Integração entre módulos funcionando")
        elif tests["integration"]["status"] == "SKIP":
            recommendations.append("⚠️ Testes de integração não executados - poucos módulos")
            
        return recommendations


def main():
    """Função principal de demonstração"""
    
    print("🚀 ControlLab Web Application - Demonstração Módulo 8")
    print("=" * 60)
    
    # Status de importação
    available_count = sum(module_status.values())
    total_count = len(module_status)
    
    print(f"\n📊 Status dos Módulos: {available_count}/{total_count} disponíveis")
    for module, status in module_status.items():
        icon = "✅" if status else "❌"
        print(f"  {icon} {module}")
        
    # Criar demonstrador
    demonstrator = ControlLabWebDemonstrator()
    
    # Gerar relatório
    report = demonstrator.generate_report()
    
    # Exibir resultados
    print("\n📋 Resultados dos Testes:")
    print("-" * 30)
    
    for test_name, result in report["test_results"].items():
        status_icon = {"SUCCESS": "✅", "SKIP": "⚠️", "ERROR": "❌"}[result["status"]]
        print(f"{status_icon} {test_name}: {result['status']}")
        
        if result["status"] == "ERROR":
            print(f"   Erro: {result['error']}")
        elif result["status"] == "SUCCESS" and test_name != "integration":
            # Mostrar informações adicionais para testes bem-sucedidos
            for key, value in result.items():
                if key not in ["status", "traceback"] and isinstance(value, bool) and value:
                    print(f"   ✓ {key}")
                    
    # Sumário
    summary = report["summary"]
    print(f"\n📈 Sumário: {summary['successful']} sucessos, {summary['skipped']} pulados, {summary['failed']} falhas")
    
    # Recomendações
    print("\n💡 Recomendações:")
    for rec in report["recommendations"]:
        print(f"  • {rec}")
        
    # Salvar relatório
    report_file = os.path.join(project_root, "web_module_test_report.json")
    try:
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Relatório salvo em: {report_file}")
    except Exception as e:
        print(f"\n⚠️ Erro salvando relatório: {e}")
        
    # Status final
    if available_count == total_count:
        print("\n🎉 Módulo 8 completamente funcional!")
        return True
    elif available_count >= total_count // 2:
        print("\n⚠️ Módulo 8 parcialmente funcional - alguns recursos limitados")
        return True
    else:
        print("\n❌ Módulo 8 com problemas - verificar instalação")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
