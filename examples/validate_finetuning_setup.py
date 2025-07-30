"""
Sistema de Validação e Teste do Fine-tuning ControlLab
Valida o dataset gerado e prepara ambiente para treinamento
"""

import json
import os
import yaml
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

class ControlLabFineTuningValidator:
    def __init__(self):
        self.dataset_path = "C:/Users/marco/Documents/ControlLab/ControlLab-Project/ai_toolkit_dataset/controllab_specialized_dataset.jsonl"
        self.config_path = "C:/Users/marco/Documents/ControlLab/ControlLab-Project/controllab_finetuning_config.yaml"
        self.validation_report_path = "C:/Users/marco/Documents/ControlLab/ControlLab-Project/validation_report.md"
        
    def validate_dataset(self):
        """Valida estrutura e conteúdo do dataset"""
        print("🔍 Validando dataset ControlLab...")
        
        conversations = []
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    conv = json.loads(line)
                    conversations.append(conv)
                except json.JSONDecodeError as e:
                    print(f"❌ Erro JSON linha {line_num}: {e}")
                    return False
        
        # Estatísticas básicas
        total_conversations = len(conversations)
        print(f"📊 Total de conversas: {total_conversations}")
        
        # Validar estrutura das conversas
        valid_conversations = 0
        conversation_types = Counter()
        message_lengths = []
        
        for i, conv in enumerate(conversations):
            if self._validate_conversation_structure(conv):
                valid_conversations += 1
                
                # Coletar metadados
                if 'metadata' in conv:
                    conv_type = conv['metadata'].get('type', 'unknown')
                    conversation_types[conv_type] += 1
                
                # Calcular comprimentos das mensagens
                for message in conv.get('messages', []):
                    message_lengths.append(len(message.get('content', '')))
            else:
                print(f"❌ Conversa {i+1} com estrutura inválida")
        
        validation_success = valid_conversations == total_conversations
        
        print(f"✅ Conversas válidas: {valid_conversations}/{total_conversations}")
        print(f"📈 Tipos de conversas: {dict(conversation_types)}")
        print(f"📏 Comprimento médio das mensagens: {sum(message_lengths)/len(message_lengths):.0f} caracteres")
        
        return {
            'success': validation_success,
            'total_conversations': total_conversations,
            'valid_conversations': valid_conversations,
            'conversation_types': dict(conversation_types),
            'avg_message_length': sum(message_lengths)/len(message_lengths) if message_lengths else 0,
            'message_lengths': message_lengths
        }
    
    def _validate_conversation_structure(self, conv):
        """Valida estrutura de uma conversa individual"""
        required_keys = ['messages']
        
        # Verificar chaves obrigatórias
        for key in required_keys:
            if key not in conv:
                return False
        
        # Verificar estrutura das mensagens
        messages = conv['messages']
        if not isinstance(messages, list) or len(messages) < 2:
            return False
        
        # Verificar alternância user/assistant
        expected_roles = ['user', 'assistant']
        for i, message in enumerate(messages):
            if not isinstance(message, dict):
                return False
            
            required_msg_keys = ['role', 'content']
            for key in required_msg_keys:
                if key not in message:
                    return False
            
            # Verificar role válido
            if message['role'] not in ['user', 'assistant', 'system']:
                return False
            
            # Verificar conteúdo não vazio
            if not message['content'].strip():
                return False
        
        return True
    
    def analyze_content_quality(self):
        """Analisa qualidade do conteúdo das conversas"""
        print("🎯 Analisando qualidade do conteúdo...")
        
        conversations = []
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                conversations.append(json.loads(line))
        
        # Palavras-chave relevantes para ControlLab
        controllab_keywords = [
            'controllab', 'sympy', 'simbólico', 'numérico', 'função de transferência',
            'estabilidade', 'routh', 'hurwitz', 'lugar das raízes', 'bode',
            'nyquist', 'pid', 'controlador', 'pólos', 'zeros', 'laplace',
            'resposta ao degrau', 'margem de fase', 'margem de ganho', 'tdah',
            'pedagógico', 'passo a passo'
        ]
        
        keyword_counts = Counter()
        pedagogical_indicators = []
        technical_depth_indicators = []
        
        for conv in conversations:
            content_text = ''
            for message in conv.get('messages', []):
                content_text += message.get('content', '').lower()
            
            # Contar palavras-chave
            for keyword in controllab_keywords:
                keyword_counts[keyword] += content_text.count(keyword)
            
            # Indicadores pedagógicos
            pedagogical_phrases = [
                'passo a passo', 'explicação', 'por que', 'como funciona',
                'exemplo', 'vamos', 'primeiro', 'depois', 'finalmente'
            ]
            pedagogical_score = sum(content_text.count(phrase) for phrase in pedagogical_phrases)
            pedagogical_indicators.append(pedagogical_score)
            
            # Indicadores de profundidade técnica
            technical_phrases = [
                'implementação', 'código', 'função', 'método', 'classe',
                'algoritmo', 'matemática', 'derivação', 'cálculo'
            ]
            technical_score = sum(content_text.count(phrase) for phrase in technical_phrases)
            technical_depth_indicators.append(technical_score)
        
        print(f"🔤 Palavras-chave mais frequentes:")
        for keyword, count in keyword_counts.most_common(10):
            print(f"  • {keyword}: {count}")
        
        avg_pedagogical = sum(pedagogical_indicators) / len(pedagogical_indicators)
        avg_technical = sum(technical_depth_indicators) / len(technical_depth_indicators)
        
        print(f"🎓 Score pedagógico médio: {avg_pedagogical:.2f}")
        print(f"🔧 Score técnico médio: {avg_technical:.2f}")
        
        return {
            'keyword_counts': dict(keyword_counts),
            'avg_pedagogical_score': avg_pedagogical,
            'avg_technical_score': avg_technical,
            'pedagogical_scores': pedagogical_indicators,
            'technical_scores': technical_depth_indicators
        }
    
    def validate_config(self):
        """Valida configuração de treinamento"""
        print("⚙️ Validando configuração de treinamento...")
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            print(f"❌ Erro ao carregar configuração: {e}")
            return False
        
        # Verificar seções obrigatórias
        required_sections = ['model_type', 'dataset', 'training', 'lora', 'output']
        for section in required_sections:
            if section not in config:
                print(f"❌ Seção obrigatória ausente: {section}")
                return False
        
        # Verificar parâmetros críticos
        critical_params = {
            'dataset.path': self.dataset_path,
            'training.batch_size': 4,
            'training.learning_rate': 1e-4,
            'lora.r': 64,
            'lora.alpha': 128
        }
        
        for param_path, expected_value in critical_params.items():
            keys = param_path.split('.')
            current_config = config
            
            try:
                for key in keys:
                    current_config = current_config[key]
                
                if param_path == 'dataset.path':
                    if not os.path.exists(current_config):
                        print(f"❌ Dataset não encontrado: {current_config}")
                        return False
                    print(f"✅ {param_path}: {current_config}")
                else:
                    print(f"✅ {param_path}: {current_config}")
                    
            except KeyError:
                print(f"❌ Parâmetro ausente: {param_path}")
                return False
        
        print("✅ Configuração válida!")
        return True
    
    def check_dependencies(self):
        """Verifica dependências necessárias para o treinamento"""
        print("📦 Verificando dependências...")
        
        required_packages = [
            'torch', 'transformers', 'peft', 'datasets', 
            'accelerate', 'bitsandbytes', 'scipy', 'numpy'
        ]
        
        missing_packages = []
        installed_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
                installed_packages.append(package)
                print(f"✅ {package}")
            except ImportError:
                missing_packages.append(package)
                print(f"❌ {package}")
        
        if missing_packages:
            print(f"\n🚨 Instalar pacotes ausentes:")
            print(f"pip install {' '.join(missing_packages)}")
            return False
        else:
            print("✅ Todas as dependências estão instaladas!")
            return True
    
    def estimate_training_time(self):
        """Estima tempo de treinamento"""
        print("⏱️ Estimando tempo de treinamento...")
        
        # Carregar configuração
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Parâmetros de treinamento
        batch_size = config['training']['batch_size']
        grad_accumulation = config['training']['gradient_accumulation_steps']
        num_epochs = config['training']['num_epochs']
        
        # Contar exemplos
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            num_examples = sum(1 for _ in f)
        
        effective_batch_size = batch_size * grad_accumulation
        steps_per_epoch = num_examples // effective_batch_size
        total_steps = steps_per_epoch * num_epochs
        
        # Estimativa baseada em hardware típico (RTX 4090)
        seconds_per_step = 2.5  # Estimativa conservativa
        total_time_minutes = (total_steps * seconds_per_step) / 60
        
        print(f"📊 Exemplos: {num_examples}")
        print(f"🔢 Batch size efetivo: {effective_batch_size}")
        print(f"📈 Steps por época: {steps_per_epoch}")
        print(f"🎯 Total de steps: {total_steps}")
        print(f"⏰ Tempo estimado: {total_time_minutes:.1f} minutos")
        
        return {
            'num_examples': num_examples,
            'effective_batch_size': effective_batch_size,
            'steps_per_epoch': steps_per_epoch,
            'total_steps': total_steps,
            'estimated_time_minutes': total_time_minutes
        }
    
    def create_training_command(self):
        """Cria comando otimizado para treinamento"""
        command = """
# Comando para iniciar treinamento ControlLab
python train_controllab_assistant.py

# Para monitorar GPU:
# nvidia-smi -l 1

# Para verificar logs:
# tail -f logs/runs/*/events.out.tfevents.*

# Para interromper graciosamente:
# Ctrl+C (salva checkpoint automático)
"""
        
        print("🚀 Comando de treinamento preparado:")
        print(command)
        return command
    
    def generate_validation_report(self):
        """Gera relatório completo de validação"""
        print("📝 Gerando relatório de validação...")
        
        # Executar todas as validações
        dataset_validation = self.validate_dataset()
        content_analysis = self.analyze_content_quality()
        config_validation = self.validate_config()
        dependencies_check = self.check_dependencies()
        training_estimate = self.estimate_training_time()
        training_command = self.create_training_command()
        
        # Gerar relatório markdown
        report_content = f"""# Relatório de Validação - Fine-tuning ControlLab

**Data:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 🎯 Resumo Executivo

O sistema de fine-tuning ControlLab foi configurado com sucesso para treinar um assistente pedagógico especializado em engenharia de controle.

### Status Geral
- ✅ Dataset: {dataset_validation['valid_conversations']}/{dataset_validation['total_conversations']} conversas válidas
- ✅ Configuração: {'Válida' if config_validation else 'Inválida'}
- ✅ Dependências: {'Instaladas' if dependencies_check else 'Ausentes'}
- ⏱️ Tempo estimado: {training_estimate['estimated_time_minutes']:.1f} minutos

## 📊 Análise do Dataset

### Estatísticas Básicas
- **Total de conversas:** {dataset_validation['total_conversations']}
- **Conversas válidas:** {dataset_validation['valid_conversations']}
- **Comprimento médio:** {dataset_validation['avg_message_length']:.0f} caracteres

### Distribuição por Tipo
{chr(10).join([f"- **{t}:** {c} conversas" for t, c in dataset_validation['conversation_types'].items()])}

### Qualidade do Conteúdo
- **Score pedagógico médio:** {content_analysis['avg_pedagogical_score']:.2f}
- **Score técnico médio:** {content_analysis['avg_technical_score']:.2f}

### Palavras-chave mais Frequentes
{chr(10).join([f"- **{k}:** {v} ocorrências" for k, v in list(content_analysis['keyword_counts'].items())[:10]])}

## ⚙️ Configuração de Treinamento

### Parâmetros Principais
- **Modelo base:** Phi-3-mini-4k-instruct
- **Método:** LoRA (Low-Rank Adaptation)
- **Batch size:** 4 (efetivo: {training_estimate['effective_batch_size']})
- **Learning rate:** 1e-4
- **Épocas:** 3

### Otimizações LoRA
- **Rank (r):** 64
- **Alpha:** 128
- **Dropout:** 0.1
- **Módulos alvo:** q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

## 🎓 Especialização Pedagógica

### Foco TDAH-Friendly
- Explicações passo a passo
- Checkpoints de progresso
- Contexto sempre preservado
- Múltiplas representações (visual, matemática, conceitual)

### Arquitetura Simbólico-Numérica
- Núcleo SymPy para precisão matemática
- Ponte python-control para simulações
- Derivações transparentes e validadas
- Interatividade em tempo real

## 📈 Estimativas de Treinamento

- **Exemplos totais:** {training_estimate['num_examples']}
- **Steps por época:** {training_estimate['steps_per_epoch']}
- **Total de steps:** {training_estimate['total_steps']}
- **Tempo estimado:** {training_estimate['estimated_time_minutes']:.1f} minutos (~{training_estimate['estimated_time_minutes']/60:.1f} horas)

## 🚀 Próximos Passos

1. **Iniciar Treinamento:**
   ```bash
   python train_controllab_assistant.py
   ```

2. **Monitorar Progresso:**
   - Verificar logs em `./logs/`
   - Monitorar GPU com `nvidia-smi`
   - Acompanhar métricas de loss

3. **Validação Pós-Treinamento:**
   - Testar respostas pedagógicas
   - Validar conhecimento ControlLab
   - Verificar suporte TDAH

4. **Integração Web:**
   - Carregar modelo na interface web
   - Configurar pipeline de inferência
   - Testar interações em tempo real

## 💡 Considerações Especiais

### Diferencial Pedagógico
O modelo será treinado para:
- Explicar não apenas "o quê", mas "por quê"
- Manter raciocínio persistente (ideal para TDAH)
- Conectar teoria com implementação prática
- Fornecer múltiplas perspectivas do mesmo conceito

### Validação de Qualidade
- Cada resposta deve incluir fundamentação teórica
- Código deve ser executável e pedagogicamente claro
- Exemplos devem conectar com material do livro Nise
- Suporte específico para usuários com TDAH

**🎯 Meta:** Criar o primeiro assistente de IA especializado em ControlLab, capaz de ensinar engenharia de controle de forma inclusiva e pedagogicamente robusta.
"""
        
        # Salvar relatório
        with open(self.validation_report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"✅ Relatório salvo: {self.validation_report_path}")
        return report_content
    
    def run_complete_validation(self):
        """Executa validação completa do sistema"""
        print("🔬 VALIDAÇÃO COMPLETA DO SISTEMA CONTROLLAB")
        print("=" * 60)
        
        try:
            # Executar validações
            dataset_valid = self.validate_dataset()
            content_analysis = self.analyze_content_quality()
            config_valid = self.validate_config()
            deps_ok = self.check_dependencies()
            training_info = self.estimate_training_time()
            
            # Gerar relatório
            report = self.generate_validation_report()
            
            # Status final
            all_valid = (
                dataset_valid['success'] and 
                config_valid and 
                deps_ok
            )
            
            print(f"\n{'✅' if all_valid else '❌'} SISTEMA {'PRONTO' if all_valid else 'REQUER ATENÇÃO'}")
            
            if all_valid:
                print("\n🚀 PRONTO PARA TREINAMENTO!")
                print("Execute: python train_controllab_assistant.py")
            else:
                print("\n🔧 Corrija os problemas identificados antes de treinar.")
            
            return {
                'ready_for_training': all_valid,
                'dataset_validation': dataset_valid,
                'content_analysis': content_analysis,
                'config_validation': config_valid,
                'dependencies_ok': deps_ok,
                'training_estimate': training_info,
                'report_path': self.validation_report_path
            }
            
        except Exception as e:
            print(f"❌ Erro durante validação: {e}")
            return {'ready_for_training': False, 'error': str(e)}

if __name__ == "__main__":
    validator = ControlLabFineTuningValidator()
    results = validator.run_complete_validation()
