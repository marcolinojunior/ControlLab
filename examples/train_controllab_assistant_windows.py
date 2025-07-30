"""
Script de Fine-tuning Otimizado para Windows - ControlLab
Versão otimizada sem dependências problemáticas
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
from peft import LoraConfig, get_peft_model
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

def setup_model_and_tokenizer(config):
    """Configura modelo e tokenizer com otimizações para Windows"""
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
    
    # Modelo com configurações otimizadas para Windows
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=None,  # Evitar problemas com meta tensors
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        cache_dir=".cache",
        attn_implementation="eager"  # Evitar problemas com flash-attention
    )
    
    # Mover modelo para GPU manualmente
    if torch.cuda.is_available():
        model = model.to("cuda")
    
    # Configuração LoRA mais conservadora
    lora_config = LoraConfig(
        r=32,  # Reduzido para estabilidade
        lora_alpha=64,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer

def create_optimized_training_args(config):
    """Cria argumentos de treinamento otimizados para Windows"""
    
    # Criar diretório de output se não existir
    output_dir = config.get('output', {}).get('output_dir', './controllab-phi3-finetuned')
    os.makedirs(output_dir, exist_ok=True)
    
    return TrainingArguments(
        output_dir=output_dir,
        
        # Configurações de treinamento conservadoras
        num_train_epochs=2,  # Reduzido para teste
        per_device_train_batch_size=1,  # Batch size pequeno para RTX 4070
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,  # Compensa batch size pequeno
        
        # Learning rate e otimização
        learning_rate=2e-5,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        
        # Logging e salvamento
        logging_steps=50,
        save_steps=200,
        eval_steps=200,
        eval_strategy="steps",
        
        # Otimizações para Windows
        fp16=True,  # Mixed precision
        gradient_checkpointing=True,  # Economia de memória
        dataloader_num_workers=0,  # Evita problemas Windows
        dataloader_pin_memory=False,  # Evita problemas Windows
        
        # Configurações de validação
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        # Logging
        logging_dir=f"{output_dir}/logs",
        report_to=[],  # Sem wandb
        
        # Otimizações adicionais
        remove_unused_columns=False,
        group_by_length=True,
        prediction_loss_only=True,
        
        # Evitar problemas de memória
        max_grad_norm=1.0,
        ddp_find_unused_parameters=False,
    )

def main():
    """Função principal otimizada para Windows"""
    print("🚀 ControlLab Fine-tuning - Versão Windows Otimizada")
    print("=" * 60)
    
    # Verificar CUDA
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"🎮 GPU: {gpu_name} ({gpu_memory:.1f}GB)")
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
        model, tokenizer = setup_model_and_tokenizer(config)
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")
        sys.exit(1)
    
    # Tokenização
    print("\n🔤 Tokenizando dados...")
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            padding=False,  # Padding dinâmico no collator
            max_length=2048,  # Reduzido para economizar memória
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
    training_args = create_optimized_training_args(config)
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test'],
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    # Informações de memória
    print(f"\n📊 Informações de memória:")
    print(f"🔢 Exemplos de treino: {len(tokenized_datasets['train'])}")
    print(f"🔢 Exemplos de validação: {len(tokenized_datasets['test'])}")
    print(f"💾 VRAM livre: {torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated():.0f} bytes")
    
    # Treinamento
    print("\n🎯 Iniciando treinamento...")
    print("=" * 60)
    
    try:
        trainer.train()
        
        # Salvar modelo final
        print("\n💾 Salvando modelo...")
        trainer.save_model()
        tokenizer.save_pretrained(training_args.output_dir)
        
        print("\n✅ Fine-tuning concluído com sucesso!")
        print(f"📁 Modelo salvo em: {training_args.output_dir}")
        
    except Exception as e:
        print(f"\n❌ Erro durante treinamento: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    # Configurações para Windows
    if sys.platform == "win32":
        # Evitar problemas de multiprocessing no Windows
        import multiprocessing
        multiprocessing.set_start_method('spawn', force=True)
    
    main()
