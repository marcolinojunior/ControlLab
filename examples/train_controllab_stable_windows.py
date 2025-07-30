#!/usr/bin/env python3
"""
ControlLab Fine-tuning Assistant - Versão Super Estável Windows
Sistema otimizado para máxima compatibilidade com GPUs RTX laptop no Windows
Resolve erros CUDA/cuBLAS com configurações conservadoras
"""

import os
import sys
import gc
import torch
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import warnings

# Configurações críticas do PyTorch para Windows + RTX laptop
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Força execução síncrona para debug
os.environ["TORCH_USE_CUDA_DSA"] = "1"    # Device-side assertions
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # Força uso apenas da GPU principal
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Evita conflitos de paralelização

# Configurações específicas para cuBLAS estável
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # Configuração determinística

# Suprimir warnings para clareza
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Configuração de logging
logging.basicConfig(
    level=logging.WARNING,  # Reduzido para menos verbosidade
    format='%(asctime)s - %(levelname)s - %(message)s'
)

try:
    import transformers
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, 
        TrainingArguments, Trainer, 
        DataCollatorForLanguageModeling,
        set_seed
    )
    from datasets import Dataset, load_dataset
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
    import numpy as np
    from tqdm import tqdm
    
    print("✅ Todas as bibliotecas carregadas com sucesso")
    
except ImportError as e:
    print(f"❌ Erro ao importar bibliotecas: {e}")
    sys.exit(1)

def print_header():
    """Imprime cabeçalho do sistema"""
    print("\n" + "="*60)
    print("🚀 ControlLab Fine-tuning - Versão SUPER ESTÁVEL Windows")
    print("="*60)
    
    # Informações da GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"🎮 GPU: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        print("❌ CUDA não disponível!")
        sys.exit(1)

def clean_gpu_memory():
    """Limpa memória GPU de forma agressiva"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()

def get_gpu_memory_info():
    """Retorna informações de memória GPU"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        cached = torch.cuda.memory_reserved() / 1e9
        return f"{allocated:.2f}GB alocada, {cached:.2f}GB cache"
    return "GPU não disponível"

@dataclass
class ControlLabTrainingConfig:
    """Configuração ultra-conservadora para máxima estabilidade"""
    
    # Modelo
    model_name: str = "microsoft/Phi-3-mini-4k-instruct"
    
    # Dataset
    dataset_path: str = r"C:\Users\marco\Documents\ControlLab\ControlLab-Project\ai_toolkit_dataset\controllab_specialized_dataset_improved.jsonl"
    max_length: int = 1024  # Reduzido para menor uso de memória
    
    # Training - Configurações ULTRA conservadoras
    batch_size: int = 1
    gradient_accumulation_steps: int = 4  # Reduzido
    learning_rate: float = 1e-5  # Learning rate menor
    num_epochs: int = 1  # Apenas 1 época para teste
    warmup_ratio: float = 0.1
    
    # LoRA - Configurações mínimas
    lora_r: int = 4  # Rank muito baixo
    lora_alpha: int = 8  # Alpha reduzido
    lora_dropout: float = 0.05
    
    # Output
    output_dir: str = "controllab_model_stable"
    save_steps: int = 50
    eval_steps: int = 50
    logging_steps: int = 10
    
    # Estabilidade
    fp16: bool = False  # Desabilita FP16 para máxima estabilidade
    bf16: bool = False  # Desabilita BF16
    dataloader_pin_memory: bool = False  # Desabilita pin memory
    dataloader_num_workers: int = 0  # Sem workers paralelos

def setup_model_and_tokenizer(config: ControlLabTrainingConfig):
    """Configura modelo e tokenizer com máxima estabilidade"""
    
    print("🤖 Configurando modelo...")
    print(f"🤖 Carregando modelo: {config.model_name}")
    
    # Limpa memória antes
    clean_gpu_memory()
    
    # Tokenizer
    print("📝 Carregando tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        padding_side="right"
    )
    
    # Adiciona pad token se necessário
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Modelo - carregamento ULTRA conservador
    print("⚙️ Carregando modelo base...")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.float32,  # Força FP32 para máxima estabilidade
        device_map=None,  # Sem device_map automático
        trust_remote_code=True,
        attn_implementation="eager",  # Força implementação eager (mais estável)
        use_cache=False  # Desabilita cache para economizar memória
    )
    
    print("🚀 Movendo modelo para GPU...")
    model = model.to("cuda")
    
    # Configuração LoRA ultra-conservadora
    print("🎯 Aplicando LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=["o_proj"],  # Apenas o_proj para simplicidade máxima
        bias="none",
        use_rslora=False,  # Desabilita RSLoRA
        use_dora=False     # Desabilita DoRA
    )
    
    model = get_peft_model(model, lora_config)
    
    # Força todos os parâmetros LoRA para FP32
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.float()
    
    # Informações
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    percentage = (trainable_params / total_params) * 100
    
    print(f"Parâmetros treináveis: {trainable_params:,} ({percentage:.2f}%)")
    
    clean_gpu_memory()
    print(f"💾 Memória GPU: {get_gpu_memory_info()}")
    
    return model, tokenizer

def load_and_prepare_dataset(config: ControlLabTrainingConfig, tokenizer):
    """Carrega e prepara dataset com validação rigorosa"""
    
    print("📚 Preparando dataset...")
    dataset_file = Path(config.dataset_path)
    
    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {dataset_file}")
    
    print(f"📂 Carregando dataset: {dataset_file.absolute()}")
    
    # Carrega dataset
    dataset = load_dataset('json', data_files=str(dataset_file))['train']
    print(f"📊 Dataset carregado: {len(dataset)} exemplos")
    
    # Split
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset['train']
    eval_dataset = dataset['test']
    
    print(f"🔄 Split criado: {len(train_dataset)} treino, {len(eval_dataset)} validação")
    
    def tokenize_function(examples):
        """Tokenização ultra-conservadora"""
        conversations = []
        
        for messages in examples['messages']:
            # Formata conversa
            text = ""
            for turn in messages:
                role = turn['role']
                content = turn['content']
                
                if role == 'user':
                    text += f"<|user|>\n{content}<|end|>\n"
                elif role == 'assistant':
                    text += f"<|assistant|>\n{content}<|end|>\n"
            
            conversations.append(text)
        
        # Tokeniza com padding limitado
        tokenized = tokenizer(
            conversations,
            truncation=True,
            padding=False,  # Sem padding aqui
            max_length=config.max_length,
            return_tensors=None
        )
        
        # Labels = input_ids (para language modeling)
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    print("🔤 Tokenizando dados...")
    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizando"
    )
    
    eval_dataset = eval_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=eval_dataset.column_names,
        desc="Tokenizando"
    )
    
    return train_dataset, eval_dataset

def create_trainer(model, tokenizer, train_dataset, eval_dataset, config: ControlLabTrainingConfig):
    """Cria trainer com configurações ultra-estáveis"""
    
    print("⚙️ Configurando treinamento...")
    
    # Argumentos de treinamento ULTRA conservadores
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        
        # Épocas e batch
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        
        # Learning rate
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type="linear",
        
        # Precisão - FP32 para máxima estabilidade
        fp16=config.fp16,
        bf16=config.bf16,
        
        # Logging e save
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        eval_strategy="steps",  # Nome atualizado do parâmetro
        save_strategy="steps",
        
        # Estabilidade máxima
        dataloader_pin_memory=config.dataloader_pin_memory,
        dataloader_num_workers=config.dataloader_num_workers,
        remove_unused_columns=False,
        
        # Otimizador estável
        optim="adamw_torch",  # Usa implementação PyTorch
        weight_decay=0.01,
        max_grad_norm=1.0,
        
        # Avaliação
        load_best_model_at_end=False,  # Desabilita para simplificar
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        # Seeds para reprodutibilidade
        seed=42,
        data_seed=42,
        
        # Desabilita recursos avançados
        prediction_loss_only=True,
        include_inputs_for_metrics=False,
        
        # Relatórios mínimos
        report_to=None,  # Sem wandb/tensorboard
        
        # Gradient checkpointing desabilitado para estabilidade
        gradient_checkpointing=False
    )
    
    # Data collator simples
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=None  # Sem padding especial
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    return trainer

def validate_model_setup(model, tokenizer):
    """Valida se modelo está funcionando antes do treinamento"""
    
    print("🔍 Verificando configuração do modelo...")
    
    try:
        # Teste simples
        test_input = "Como calcular a função de transferência"
        inputs = tokenizer(test_input, return_tensors="pt").to("cuda")
        
        # Forward pass de teste
        with torch.no_grad():
            outputs = model(**inputs)
            
        print("✅ Modelo respondendo corretamente")
        
        # Verifica gradientes
        has_gradients = any(p.requires_grad for p in model.parameters())
        if has_gradients:
            print("✅ Parâmetros com gradientes encontrados")
        else:
            print("❌ Nenhum parâmetro com gradientes!")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Erro na validação: {e}")
        return False

def main():
    """Função principal ultra-estável"""
    
    try:
        print_header()
        
        # Configuração
        config = ControlLabTrainingConfig()
        
        # Seed para reprodutibilidade
        set_seed(42)
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Setup modelo
        model, tokenizer = setup_model_and_tokenizer(config)
        
        # Dataset
        train_dataset, eval_dataset = load_and_prepare_dataset(config, tokenizer)
        
        # Trainer
        trainer = create_trainer(model, tokenizer, train_dataset, eval_dataset, config)
        
        # Configuração final
        print("\n📊 Configuração final:")
        print(f"🔢 Exemplos de treino: {len(train_dataset)}")
        print(f"🔢 Exemplos de validação: {len(eval_dataset)}")
        print(f"💾 Batch size: {config.batch_size}")
        print(f"🎯 Learning rate: {config.learning_rate}")
        print(f"🔄 Épocas: {config.num_epochs}")
        print(f"⚡ Precisão: FP32 (máxima estabilidade)")
        
        # Validação
        if not validate_model_setup(model, tokenizer):
            print("❌ Falha na validação do modelo")
            return
        
        # Limpa memória antes do treinamento
        clean_gpu_memory()
        
        print("\n🎯 Iniciando treinamento...")
        print("="*60)
        print(f"💾 Memória GPU em uso: {get_gpu_memory_info()}")
        
        # TREINAMENTO
        trainer.train()
        
        print("\n✅ Treinamento concluído!")
        
        # Salva modelo final
        trainer.save_model()
        tokenizer.save_pretrained(config.output_dir)
        
        print(f"💾 Modelo salvo em: {config.output_dir}")
        
    except Exception as e:
        print(f"\n❌ Erro durante treinamento: {e}")
        
        # Debug adicional
        print("\n🔍 Informações de debug:")
        print(f"💾 Memória GPU: {get_gpu_memory_info()}")
        
        # Lista parâmetros com gradientes
        try:
            print("🎯 Parâmetros com gradientes:")
            for name, param in model.named_parameters():
                if param.requires_grad:
                    print(f"  {name}: {param.shape}")
        except:
            pass
            
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
