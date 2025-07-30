"""
ControlLab Fine-tuning - Versão Final Windows (Sem Flask)
Solução completa e robusta para Windows sem dependências problemáticas
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
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import warnings
warnings.filterwarnings("ignore")

def load_config(config_path):
    """Carrega configuração do YAML"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def prepare_dataset():
    """Prepara dataset para treinamento"""
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
    splits = dataset.train_test_split(test_size=0.1, seed=42)
    
    print(f"🔄 Split criado: {len(splits['train'])} treino, {len(splits['test'])} validação")
    return splits

def setup_model_and_tokenizer():
    """Configuração limpa e robusta do modelo"""
    model_name = "microsoft/Phi-3-mini-4k-instruct"
    print(f"🤖 Carregando modelo: {model_name}")
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=True
    )
    
    # Configurar pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print("⚙️ Carregando modelo base...")
    
    # Modelo - configuração limpa
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map=None,  # Carregamento manual
        use_cache=False,  # CRÍTICO para treinamento
        attn_implementation="eager"
    )
    
    # Mover para GPU
    if torch.cuda.is_available():
        print("🚀 Movendo modelo para GPU...")
        model = model.cuda()
    
    # Habilitar gradientes explicitamente
    for param in model.parameters():
        param.requires_grad = False  # Primeiro desabilitar todos
    
    print("🎯 Aplicando LoRA...")
    
    # LoRA Config - configuração simples e robusta
    lora_config = LoraConfig(
        r=8,  # Muito conservador
        lora_alpha=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj"
        ],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False  # CRÍTICO para treinamento
    )
    
    # Aplicar LoRA
    model = get_peft_model(model, lora_config)
    
    # Verificar gradientes
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"Parâmetros treináveis: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    # Habilitar modo de treinamento
    model.train()
    
    return model, tokenizer

def main():
    """Função principal otimizada para Windows"""
    print("🚀 ControlLab Fine-tuning - Versão Final Windows")
    print("=" * 60)
    
    # Verificar GPU
    if not torch.cuda.is_available():
        print("❌ CUDA não disponível!")
        sys.exit(1)
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"🎮 GPU: {gpu_name} ({gpu_memory:.1f}GB)")
    
    # Limpar cache
    torch.cuda.empty_cache()
    
    # Preparar dados
    print("\n📚 Preparando dataset...")
    dataset_splits = prepare_dataset()
    
    # Configurar modelo
    print("\n🤖 Configurando modelo...")
    model, tokenizer = setup_model_and_tokenizer()
    
    # Tokenização
    print("\n🔤 Tokenizando dados...")
    def tokenize_function(examples):
        # Tokenização simples e robusta
        result = tokenizer(
            examples['text'],
            truncation=True,
            padding=False,
            max_length=1024,
            return_tensors=None
        )
        
        # Adicionar labels para language modeling
        result["labels"] = result["input_ids"].copy()
        return result
    
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
        return_tensors="pt"
    )
    
    # Argumentos de treinamento super conservadores
    print("\n⚙️ Configurando treinamento...")
    
    output_dir = "./controllab-phi3-final"
    os.makedirs(output_dir, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        
        # Configurações mínimas
        num_train_epochs=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=2,
        
        # Learning rate muito baixo
        learning_rate=5e-6,
        weight_decay=0.01,
        warmup_steps=10,
        
        # Logging
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=25,
        save_steps=50,
        
        # Otimizações Windows
        fp16=True,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        
        # Segurança
        save_total_limit=1,
        prediction_loss_only=True,
        
        # Essencial para Windows
        report_to=[],
        logging_dir=None,
        load_best_model_at_end=False,
        metric_for_best_model=None,
        
        # Evitar problemas
        gradient_checkpointing=False,  # Desabilitar para simplicidade
        group_by_length=False,
        ddp_find_unused_parameters=False,
        dataloader_persistent_workers=False,
    )
    
    # Informações
    print(f"\n📊 Configuração final:")
    print(f"🔢 Exemplos de treino: {len(tokenized_datasets['train'])}")
    print(f"🔢 Exemplos de validação: {len(tokenized_datasets['test'])}")
    print(f"💾 Batch size: {training_args.per_device_train_batch_size}")
    print(f"🎯 Learning rate: {training_args.learning_rate}")
    
    # Trainer
    print("\n👨‍🏫 Criando trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test'],
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    # Verificar se modelo está pronto
    print("🔍 Verificando configuração do modelo...")
    sample_input = tokenized_datasets['train'][0]
    
    # Teste rápido
    with torch.no_grad():
        test_input = {k: torch.tensor([v]).cuda() if torch.cuda.is_available() else torch.tensor([v]) 
                     for k, v in sample_input.items() if k in ['input_ids', 'attention_mask']}
        try:
            output = model(**test_input)
            print("✅ Modelo respondendo corretamente")
        except Exception as e:
            print(f"❌ Erro no teste do modelo: {e}")
            sys.exit(1)
    
    # Treinamento
    print("\n🎯 Iniciando treinamento...")
    print("=" * 60)
    
    try:
        # Verificar memória GPU antes
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            memory_used = torch.cuda.memory_allocated() / (1024**3)
            print(f"💾 Memória GPU em uso: {memory_used:.2f}GB")
        
        trainer.train()
        
        # Salvar modelo
        print("\n💾 Salvando modelo...")
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)
        
        print("\n✅ Fine-tuning concluído com sucesso!")
        print(f"📁 Modelo salvo em: {output_dir}")
        
    except Exception as e:
        print(f"\n❌ Erro durante treinamento: {e}")
        import traceback
        traceback.print_exc()
        
        # Informações de debug
        print("\n🔍 Informações de debug:")
        if torch.cuda.is_available():
            print(f"💾 Memória GPU: {torch.cuda.memory_allocated() / (1024**3):.2f}GB")
        
        print("🎯 Parâmetros com gradientes:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"  {name}: {param.shape}")
        
        sys.exit(1)

if __name__ == "__main__":
    # Configuração Windows
    if sys.platform == "win32":
        import multiprocessing
        multiprocessing.set_start_method('spawn', force=True)
    
    # Configurar PyTorch
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    # Desabilitar warnings
    import logging
    logging.getLogger("transformers").setLevel(logging.ERROR)
    
    main()
