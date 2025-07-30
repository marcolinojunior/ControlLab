"""
Script de Fine-tuning Super Otimizado para Windows - ControlLab
Versão robusta que resolve problemas de meta tensors
"""

import torch
import yaml
import os
import sys
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import json
import warnings
warnings.filterwarnings("ignore")

def load_config(config_path):
    """Carrega configuração do YAML"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def prepare_dataset(config):
    """Prepara dataset para treinamento"""
    # Usar dataset melhorado
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, "ai_toolkit_dataset", "controllab_specialized_dataset_improved.jsonl")
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset não encontrado: {dataset_path}")
        sys.exit(1)
    
    print(f"📂 Carregando dataset: {dataset_path}")
    dataset = load_dataset('json', data_files=dataset_path, split='train')
    print(f"📊 Dataset carregado: {len(dataset)} exemplos")
    
    def format_conversation(example):
        """Formata conversa para treinamento"""
        messages = example['messages']
        formatted = ""
        
        for message in messages:
            role = message['role']
            content = message['content']
            
            if role == "user":
                formatted += f"<|user|>\n{content}<|end|>\n"
            elif role == "assistant":
                formatted += f"<|assistant|>\n{content}<|end|>\n"
        
        return {"text": formatted}
    
    # Mapear dataset
    dataset = dataset.map(format_conversation, remove_columns=dataset.column_names)
    
    # Split para validação
    split_ratio = config.get('dataset', {}).get('validation_split', 0.1)
    splits = dataset.train_test_split(test_size=split_ratio, seed=42)
    
    print(f"🔄 Split criado: {len(splits['train'])} treino, {len(splits['test'])} validação")
    return splits

def setup_model_and_tokenizer_robust(config):
    """Configuração robusta que evita problemas de meta tensors"""
    model_name = config['base_model']
    print(f"🤖 Carregando modelo: {model_name}")
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir=".cache"
    )
    
    # Configurar tokens especiais
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("⚙️ Carregando modelo base...")
    
    # Modelo com carregamento robusto - evitar device_map
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map=None,  # CRÍTICO: não usar device_map 
        low_cpu_mem_usage=False,  # Usar mais CPU para evitar meta tensors
        cache_dir=".cache",
        attn_implementation="eager"
    )
    
    print("🔧 Preparando modelo para LoRA...")
    
    # Preparar modelo para treinamento quantizado (mesmo sem quantização)
    model = prepare_model_for_kbit_training(model)
    
    # Mover para GPU DEPOIS de preparar
    if torch.cuda.is_available():
        print("🚀 Movendo modelo para GPU...")
        model = model.to("cuda")
    
    # Configuração LoRA mais conservadora
    print("🎯 Aplicando LoRA...")
    lora_config = LoraConfig(
        r=16,  # Ainda mais conservador
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer

def create_safe_training_args(config):
    """Cria argumentos de treinamento ultra-seguros"""
    
    # Criar diretório de output se não existir
    output_dir = config.get('output', {}).get('output_dir', './controllab-phi3-finetuned')
    os.makedirs(output_dir, exist_ok=True)
    
    return TrainingArguments(
        output_dir=output_dir,
        
        # Configurações ultra-conservadoras
        num_train_epochs=1,  # Apenas 1 época para teste
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,  # Reduzido
        
        # Learning rate muito baixo
        learning_rate=1e-5,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        warmup_steps=50,
        
        # Logging mais frequente
        logging_steps=10,
        save_steps=50,
        eval_steps=50,
        eval_strategy="steps",
        
        # Otimizações críticas para Windows
        fp16=True,
        gradient_checkpointing=True,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        
        # Evitar problemas de memória
        remove_unused_columns=False,
        group_by_length=False,  # Desabilitar para simplicidade
        prediction_loss_only=True,
        
        # Validação
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        # Logging básico
        logging_dir=f"{output_dir}/logs",
        report_to=[],
        
        # Configurações de segurança
        max_grad_norm=0.5,  # Muito conservador
        ddp_find_unused_parameters=False,
        save_strategy="steps",
        
        # Força CPU para algumas operações
        dataloader_prefetch_factor=None,
        dataloader_persistent_workers=False,
    )

def main():
    """Função principal super robusta"""
    print("🚀 ControlLab Fine-tuning - Versão Ultra-Robusta Windows")
    print("=" * 70)
    
    # Verificar CUDA
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"🎮 GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        
        # Limpar cache CUDA
        torch.cuda.empty_cache()
    else:
        print("❌ CUDA não disponível!")
        sys.exit(1)
    
    # Carregar configuração
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "controllab_finetuning_config.yaml")
    if not os.path.exists(config_path):
        print(f"❌ Arquivo de configuração não encontrado: {config_path}")
        sys.exit(1)
    
    config = load_config(config_path)
    print("📋 Configuração carregada")
    
    # Preparar dados
    print("\n📚 Preparando dataset...")
    try:
        dataset_splits = prepare_dataset(config)
    except Exception as e:
        print(f"❌ Erro ao carregar dataset: {e}")
        sys.exit(1)
    
    # Configurar modelo
    print("\n🤖 Configurando modelo...")
    try:
        model, tokenizer = setup_model_and_tokenizer_robust(config)
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Tokenização
    print("\n🔤 Tokenizando dados...")
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            padding=False,
            max_length=1024,  # Ainda mais conservador
            return_tensors=None
        )
    
    tokenized_datasets = dataset_splits.map(
        tokenize_function, 
        batched=True,
        remove_columns=dataset_splits['train'].column_names,
        desc="Tokenizando"
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,
        return_tensors="pt"
    )
    
    # Argumentos de treinamento
    print("\n⚙️ Configurando treinamento...")
    training_args = create_safe_training_args(config)
    
    # Informações de memória
    print(f"\n📊 Informações de memória:")
    print(f"🔢 Exemplos de treino: {len(tokenized_datasets['train'])}")
    print(f"🔢 Exemplos de validação: {len(tokenized_datasets['test'])}")
    if torch.cuda.is_available():
        print(f"💾 VRAM livre: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / (1024**3):.2f}GB")
    
    # Trainer
    print("\n👨‍🏫 Criando trainer...")
    try:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets['train'],
            eval_dataset=tokenized_datasets['test'],
            data_collator=data_collator,
            tokenizer=tokenizer
        )
        print("✅ Trainer criado com sucesso!")
    except Exception as e:
        print(f"❌ Erro ao criar trainer: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Treinamento
    print("\n🎯 Iniciando treinamento...")
    print("=" * 70)
    
    try:
        trainer.train()
        
        # Salvar modelo final
        print("\n💾 Salvando modelo...")
        trainer.save_model()
        tokenizer.save_pretrained(training_args.output_dir)
        
        print("\n✅ Fine-tuning concluído com sucesso!")
        print(f"📁 Modelo salvo em: {training_args.output_dir}")
        
        # Informações finais
        print(f"\n📈 Estatísticas finais:")
        print(f"🎯 Parâmetros treináveis: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        print(f"💾 Tamanho final do modelo: {os.path.getsize(os.path.join(training_args.output_dir, 'adapter_model.safetensors')) / (1024**2):.1f}MB")
        
    except Exception as e:
        print(f"\n❌ Erro durante treinamento: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    # Configurações para Windows
    if sys.platform == "win32":
        import multiprocessing
        multiprocessing.set_start_method('spawn', force=True)
    
    # Configurar PyTorch para usar menos memória
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    main()
