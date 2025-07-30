#!/usr/bin/env python3
"""
Integração do Módulo 4 - Modelagem com Outros Módulos ControlLab
===============================================================

Este módulo implementa a integração entre o módulo de modelagem (Módulo 4)
e os outros módulos existentes do ControlLab, criando um ambiente educacional
unificado e coeso.

Funcionalidades de Integração:
- Conexão com core (symbolic_tf, symbolic_ss)
- Integração com analysis (estabilidade, resposta)  
- Compatibilidade com numerical (simulação numérica)
- Interface com visualization (gráficos educativos)
- Pipeline educacional completo
"""

import sys
import os
import warnings
from typing import Dict, List, Any, Optional, Union, Tuple

# Adicionar src ao path para importações
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, '..', '..')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

try:
    import sympy as sp
    from sympy import symbols, Function, simplify, expand
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False

# Importações do ControlLab Core
try:
    from controllab.core import (
        SymbolicTransferFunction,
        SymbolicStateSpace,
        OperationHistory,
        OperationStep,
        create_laplace_variable,
        extract_poles_zeros,
        symbolic_stability_analysis
    )
    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False

# Importações do Módulo de Modelagem
try:
    from controllab.modeling import (
        LaplaceTransformer,
        PartialFractionExpander,
        MechanicalSystem,
        ElectricalSystem,
        ThermalSystem,
        tf_to_ss,
        ss_to_tf,
        controllable_canonical,
        validate_system_properties,
        show_laplace_steps
    )
    MODELING_AVAILABLE = True
except ImportError:
    MODELING_AVAILABLE = False

# Importações do Módulo Numerical
try:
    from controllab.numerical import (
        create_asymptotic_bode,
        compile_transfer_function
    )
    NUMERICAL_AVAILABLE = True
except ImportError:
    NUMERICAL_AVAILABLE = False


class EducationalPipeline:
    """
    Pipeline educacional que integra todos os módulos do ControlLab
    para criar uma experiência de aprendizado completa e coesa.
    """
    
    def __init__(self):
        self.history = OperationHistory() if CORE_AVAILABLE else []
        self.s = create_laplace_variable() if CORE_AVAILABLE else symbols('s')
        self.current_system = None
        self.learning_path = []
        self.educational_context = "general"
    
    def start_learning_session(self, topic: str, level: str = "beginner"):
        """
        Inicia uma sessão de aprendizado integrada
        
        Args:
            topic: Tópico de estudo ("laplace", "modeling", "control", "stability")
            level: Nível do estudante ("beginner", "intermediate", "advanced")
        """
        self.educational_context = topic
        self.learning_path = []
        
        print(f"🎓 SESSÃO DE APRENDIZADO INICIADA")
        print(f"📚 Tópico: {topic.title()}")
        print(f"📊 Nível: {level.title()}")
        print(f"🔧 Módulos disponíveis:")
        
        available_modules = []
        if CORE_AVAILABLE:
            available_modules.append("✅ Core (Simbólico)")
        if MODELING_AVAILABLE:
            available_modules.append("✅ Modeling (Modelagem)")
        if NUMERICAL_AVAILABLE:
            available_modules.append("✅ Numerical (Numérico)")
        
        for module in available_modules:
            print(f"   {module}")
        
        print(f"🚀 Pipeline educacional pronto!\n")
    
    def model_physical_system(self, system_type: str, parameters: Dict, show_derivation: bool = True):
        """
        Modela um sistema físico usando integração completa dos módulos
        
        Args:
            system_type: Tipo do sistema ("mechanical", "electrical", "thermal")
            parameters: Parâmetros do sistema
            show_derivation: Mostrar derivação passo-a-passo
        
        Returns:
            Sistema modelado com histórico educacional
        """
        if not MODELING_AVAILABLE:
            raise RuntimeError("Módulo de modelagem não disponível")
        
        print(f"🏗️ MODELAGEM DE SISTEMA FÍSICO: {system_type.upper()}")
        print("=" * 50)
        
        # Etapa 1: Criar sistema físico
        if system_type.lower() == "mechanical":
            system = MechanicalSystem(**parameters)
            print("📋 Sistema Massa-Mola-Amortecedor criado")
        elif system_type.lower() == "electrical":
            system = ElectricalSystem(**parameters)
            print("⚡ Circuito Elétrico RLC criado")
        elif system_type.lower() == "thermal":
            system = ThermalSystem(**parameters)
            print("🌡️ Sistema Térmico criado")
        else:
            raise ValueError(f"Tipo de sistema não suportado: {system_type}")
        
        # Etapa 2: Derivar equações diferenciais
        print("\n🔬 DERIVAÇÃO DAS EQUAÇÕES:")
        ode_result = system.derive_equations(show_steps=show_derivation)
        
        # Etapa 3: Aplicar transformada de Laplace
        print("\n🔄 TRANSFORMADA DE LAPLACE:")
        laplace_result = system.apply_laplace_modeling(show_steps=show_derivation)
        
        # Etapa 4: Converter para representações padrão do ControlLab
        if CORE_AVAILABLE:
            print("\n🔗 INTEGRAÇÃO COM NÚCLEO SIMBÓLICO:")
            
            # Criar SymbolicTransferFunction do core
            tf_core = SymbolicTransferFunction(
                numerator=laplace_result['transfer_function'].as_numer_denom()[0],
                denominator=laplace_result['transfer_function'].as_numer_denom()[1],
                variable=self.s
            )
            
            print(f"✅ SymbolicTransferFunction criada: {tf_core}")
            
            # Extrair pólos e zeros usando core
            poles_zeros = extract_poles_zeros(tf_core.get_expression())
            print(f"📍 Pólos: {poles_zeros.get('poles', [])}")
            print(f"🎯 Zeros: {poles_zeros.get('zeros', [])}")
            
            self.current_system = {
                'physical_system': system,
                'transfer_function': tf_core,
                'state_space': None,
                'poles_zeros': poles_zeros,
                'modeling_history': laplace_result.get('history', [])
            }
        
        # Etapa 5: Análise de estabilidade integrada
        if CORE_AVAILABLE:
            print("\n🎯 ANÁLISE DE ESTABILIDADE:")
            stability = symbolic_stability_analysis(tf_core.get_expression())
            print(f"📊 Estabilidade: {stability}")
            self.current_system['stability'] = stability
        
        # Etapa 6: Conversão para espaço de estados
        print("\n🔄 CONVERSÃO PARA ESPAÇO DE ESTADOS:")
        ss_result = tf_to_ss(laplace_result['transfer_function'], form='controllable')
        
        if CORE_AVAILABLE and hasattr(ss_result, 'A'):
            ss_core = SymbolicStateSpace(
                A=ss_result.A,
                B=ss_result.B, 
                C=ss_result.C,
                D=ss_result.D
            )
            print(f"✅ SymbolicStateSpace criado")
            print(f"📐 Ordem do sistema: {ss_core.A.shape[0]}")
            self.current_system['state_space'] = ss_core
        
        self.learning_path.append({
            'step': 'physical_modeling',
            'system_type': system_type,
            'parameters': parameters,
            'timestamp': 'now'
        })
        
        return self.current_system
    
    def analyze_system_properties(self, detailed: bool = True):
        """
        Análise completa das propriedades do sistema usando todos os módulos
        
        Args:
            detailed: Análise detalhada com explicações pedagógicas
        
        Returns:
            Relatório completo de análise
        """
        if not self.current_system:
            raise RuntimeError("Nenhum sistema foi modelado ainda")
        
        print("🔍 ANÁLISE COMPLETA DO SISTEMA")
        print("=" * 40)
        
        analysis_report = {
            'system_info': {},
            'mathematical_properties': {},
            'physical_interpretation': {},
            'educational_insights': []
        }
        
        # Análise básica do sistema
        tf = self.current_system['transfer_function']
        print(f"📋 Função de Transferência: {tf}")
        analysis_report['system_info']['transfer_function'] = str(tf)
        
        # Validações pedagógicas usando o módulo de modelagem
        if MODELING_AVAILABLE:
            print("\n🔍 VALIDAÇÕES PEDAGÓGICAS:")
            validations = validate_system_properties(tf.get_expression())
            
            for validation_name, result in validations.items():
                print(f"\n📊 {validation_name.replace('_', ' ').title()}:")
                print(result.get_summary())
                analysis_report['mathematical_properties'][validation_name] = {
                    'valid': result.is_valid,
                    'warnings': result.warnings,
                    'properties': result.properties
                }
        
        # Análise de formas canônicas
        if self.current_system.get('state_space'):
            print("\n📐 FORMAS CANÔNICAS:")
            ss = self.current_system['state_space']
            
            try:
                controllable_form = controllable_canonical(tf.get_expression())
                print("✅ Forma canônica controlável calculada")
                analysis_report['mathematical_properties']['controllable_canonical'] = "computed"
            except Exception as e:
                print(f"⚠️ Erro na forma controlável: {e}")
        
        # Interpretação física
        physical_system = self.current_system['physical_system']
        print(f"\n🏗️ INTERPRETAÇÃO FÍSICA:")
        print(f"Sistema: {physical_system.system_name}")
        print(f"Parâmetros: {physical_system.parameters}")
        
        analysis_report['physical_interpretation'] = {
            'system_name': physical_system.system_name,
            'parameters': physical_system.parameters,
            'physical_meaning': self._get_physical_meaning(physical_system)
        }
        
        # Insights educacionais
        insights = self._generate_educational_insights()
        analysis_report['educational_insights'] = insights
        
        print(f"\n💡 INSIGHTS EDUCACIONAIS:")
        for insight in insights:
            print(f"  • {insight}")
        
        self.learning_path.append({
            'step': 'system_analysis',
            'report': analysis_report,
            'timestamp': 'now'
        })
        
        return analysis_report
    
    def create_educational_visualization(self, plot_type: str = "comprehensive"):
        """
        Cria visualizações educativas usando integração dos módulos
        
        Args:
            plot_type: Tipo de gráfico ("bode", "root_locus", "step", "comprehensive")
        
        Returns:
            Caminhos dos arquivos de visualização criados
        """
        if not self.current_system:
            raise RuntimeError("Nenhum sistema foi modelado ainda")
        
        print("📊 CRIANDO VISUALIZAÇÕES EDUCATIVAS")
        print("=" * 40)
        
        visualization_files = []
        
        # Verificar se temos módulo numerical para gráficos
        if NUMERICAL_AVAILABLE:
            tf = self.current_system['transfer_function']
            
            print("📈 Criando diagrama de Bode assintótico...")
            try:
                bode_result = create_asymptotic_bode(tf.get_expression())
                if bode_result:
                    print("✅ Diagrama de Bode criado")
                    visualization_files.append("bode_diagram.png")
            except Exception as e:
                print(f"⚠️ Erro no Bode: {e}")
        
        # Visualização dos passos de modelagem
        if MODELING_AVAILABLE and self.current_system.get('modeling_history'):
            print("📋 Criando visualização dos passos de modelagem...")
            try:
                show_laplace_steps(self.current_system['modeling_history'])
                print("✅ Passos de modelagem visualizados")
                visualization_files.append("modeling_steps.html")
            except Exception as e:
                print(f"⚠️ Erro na visualização: {e}")
        
        self.learning_path.append({
            'step': 'visualization',
            'files': visualization_files,
            'timestamp': 'now'
        })
        
        return visualization_files
    
    def generate_learning_report(self):
        """
        Gera relatório completo do percurso de aprendizado
        
        Returns:
            Relatório educacional estruturado
        """
        print("📄 RELATÓRIO DE APRENDIZADO")
        print("=" * 30)
        
        report = {
            'session_info': {
                'topic': self.educational_context,
                'steps_completed': len(self.learning_path),
                'modules_used': []
            },
            'learning_path': self.learning_path,
            'system_summary': {},
            'educational_achievements': [],
            'next_steps': []
        }
        
        # Identificar módulos utilizados
        if CORE_AVAILABLE:
            report['session_info']['modules_used'].append("Core (Simbólico)")
        if MODELING_AVAILABLE:
            report['session_info']['modules_used'].append("Modeling (Modelagem)")
        if NUMERICAL_AVAILABLE:
            report['session_info']['modules_used'].append("Numerical (Numérico)")
        
        # Resumo do sistema atual
        if self.current_system:
            report['system_summary'] = {
                'type': self.current_system['physical_system'].system_name,
                'parameters': self.current_system['physical_system'].parameters,
                'transfer_function': str(self.current_system['transfer_function']),
                'stability': self.current_system.get('stability', 'not_analyzed')
            }
        
        # Conquistas educacionais
        achievements = []
        for step in self.learning_path:
            if step['step'] == 'physical_modeling':
                achievements.append(f"✅ Modelou sistema {step['system_type']}")
            elif step['step'] == 'system_analysis':
                achievements.append("✅ Realizou análise completa")
            elif step['step'] == 'visualization':
                achievements.append("✅ Criou visualizações educativas")
        
        report['educational_achievements'] = achievements
        
        # Próximos passos sugeridos
        next_steps = [
            "🎯 Experimente diferentes parâmetros do sistema",
            "📊 Analise resposta temporal e frequencial",
            "🎛️ Projete controladores para o sistema",
            "🔍 Compare com outros tipos de sistemas"
        ]
        report['next_steps'] = next_steps
        
        # Exibir relatório
        print(f"📚 Tópico estudado: {report['session_info']['topic']}")
        print(f"🔢 Passos completados: {report['session_info']['steps_completed']}")
        print(f"🧩 Módulos utilizados: {', '.join(report['session_info']['modules_used'])}")
        
        print(f"\n🏆 CONQUISTAS:")
        for achievement in achievements:
            print(f"  {achievement}")
        
        print(f"\n🎯 PRÓXIMOS PASSOS:")
        for step in next_steps:
            print(f"  {step}")
        
        return report
    
    def _get_physical_meaning(self, system):
        """Obtém significado físico dos parâmetros"""
        if hasattr(system, 'system_name'):
            if "Mecânico" in system.system_name:
                return {
                    'mass': 'Inércia do sistema',
                    'damping': 'Amortecimento/atrito',
                    'stiffness': 'Rigidez da mola'
                }
            elif "Elétrico" in system.system_name:
                return {
                    'resistance': 'Resistência elétrica',
                    'inductance': 'Indutância',
                    'capacitance': 'Capacitância'
                }
            elif "Térmico" in system.system_name:
                return {
                    'thermal_resistance': 'Resistência térmica',
                    'thermal_capacitance': 'Capacitância térmica'
                }
        return {}
    
    def _generate_educational_insights(self):
        """Gera insights educacionais baseados no sistema atual"""
        insights = []
        
        if self.current_system:
            # Insights baseados no tipo de sistema
            system_name = self.current_system['physical_system'].system_name
            
            if "Mecânico" in system_name:
                insights.extend([
                    "Sistemas mecânicos são fundamentais em robótica e automação",
                    "O amortecimento determina o overshoot da resposta temporal",
                    "A frequência natural depende da massa e rigidez"
                ])
            
            elif "Elétrico" in system_name:
                insights.extend([
                    "Circuitos RLC são análogos a sistemas mecânicos",
                    "A resistência introduz amortecimento no sistema",
                    "Capacitores e indutores armazenam energia"
                ])
            
            # Insights baseados na estabilidade
            if self.current_system.get('stability'):
                stability = self.current_system['stability']
                if 'stable' in str(stability).lower():
                    insights.append("Sistema estável retorna ao equilíbrio após perturbação")
                else:
                    insights.append("Sistemas instáveis requerem controle em malha fechada")
        
        return insights


# Classe de fallback para quando módulos não estão disponíveis
class FallbackEducationalPipeline:
    """Pipeline educacional simplificado quando módulos não estão disponíveis"""
    
    def __init__(self):
        self.warning_shown = False
    
    def _show_warning(self):
        if not self.warning_shown:
            print("⚠️ Funcionalidades limitadas - instale todas as dependências para experiência completa")
            self.warning_shown = True
    
    def start_learning_session(self, topic: str, level: str = "beginner"):
        self._show_warning()
        print(f"📚 Sessão de aprendizado iniciada: {topic} (nível {level})")
        print("🔧 Módulos limitados disponíveis")
    
    def model_physical_system(self, system_type: str, parameters: Dict, show_derivation: bool = True):
        self._show_warning()
        raise NotImplementedError("Módulos completos necessários para modelagem")


# Instanciar pipeline apropriado
if CORE_AVAILABLE and MODELING_AVAILABLE:
    # Pipeline completo disponível
    def create_educational_pipeline():
        return EducationalPipeline()
else:
    # Pipeline limitado
    def create_educational_pipeline():
        return FallbackEducationalPipeline()


# Funções de conveniência para integração
def integrate_with_core(modeling_result):
    """
    Integra resultado de modelagem com núcleo simbólico
    
    Args:
        modeling_result: Resultado do módulo de modelagem
    
    Returns:
        Objeto do núcleo simbólico integrado
    """
    if not CORE_AVAILABLE:
        raise RuntimeError("Núcleo simbólico não disponível")
    
    if hasattr(modeling_result, 'transfer_function'):
        tf_expr = modeling_result.transfer_function
    else:
        tf_expr = modeling_result
    
    num, den = sp.fraction(tf_expr)
    s = create_laplace_variable()
    
    return SymbolicTransferFunction(
        numerator=num,
        denominator=den,
        variable=s
    )


def create_educational_workflow(system_type: str, parameters: Dict):
    """
    Cria fluxo educacional completo para um sistema
    
    Args:
        system_type: Tipo do sistema físico
        parameters: Parâmetros do sistema
    
    Returns:
        Pipeline educacional executado
    """
    pipeline = create_educational_pipeline()
    
    # Iniciar sessão
    pipeline.start_learning_session("modeling", "intermediate")
    
    # Modelar sistema
    system = pipeline.model_physical_system(system_type, parameters)
    
    # Analisar propriedades
    analysis = pipeline.analyze_system_properties()
    
    # Criar visualizações
    visualizations = pipeline.create_educational_visualization()
    
    # Gerar relatório
    report = pipeline.generate_learning_report()
    
    return {
        'pipeline': pipeline,
        'system': system,
        'analysis': analysis,
        'visualizations': visualizations,
        'report': report
    }
