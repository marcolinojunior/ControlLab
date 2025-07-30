"""
Script de Fine-tuning Especializado para ControlLab
Configura modelo Phi-3-mini para assistente pedagógico de controle
"""

import torch
import yaml
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import json

def load_config(config_path):
    """Carrega configuração do YAML"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def prepare_dataset(config):
    """Prepara dataset para treinamento"""
    # Usar dataset melhorado
    dataset_path = r"C:\Users\marco\Documents\ControlLab\ControlLab-Project\ai_toolkit_dataset\controllab_specialized_dataset_improved.jsonl"
    dataset = load_dataset('json', data_files=dataset_path, split='train')
    
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
    
    # Remover colunas desnecessárias e mapear
    dataset = dataset.map(format_conversation, remove_columns=dataset.column_names)
    return dataset.train_test_split(test_size=config['dataset']['validation_split'])

def setup_model_and_tokenizer(config):
    """Configura modelo e tokenizer"""
    model_name = config['base_model']
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = config['special_tokens']['pad_token']
    tokenizer.eos_token = config['special_tokens']['eos_token']
    
    # Modelo
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if config['optimization']['use_fp16'] else torch.float32,
        device_map=None,  # Don't use device_map with meta tensors
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    # Configuração LoRA
    lora_config = LoraConfig(
        r=config['lora']['r'],
        lora_alpha=config['lora']['alpha'],
        target_modules=config['lora']['target_modules'],
        lora_dropout=config['lora']['dropout'],
        bias=config['lora']['bias'],
        task_type=config['lora']['task_type']
    )
    
    model = get_peft_model(model, lora_config)
    
    return model, tokenizer

def main():
    """Função principal de treinamento"""
    print("🚀 Iniciando fine-tuning especializado ControlLab...")
    
    # Carregar configuração
    config_path = r"C:\Users\marco\Documents\ControlLab\ControlLab-Project\controllab_finetuning_config.yaml"
    config = load_config(config_path)
    
    # Preparar dados
    print("📚 Preparando dataset...")
    dataset_splits = prepare_dataset(config)
    
    # Configurar modelo
    print("🤖 Configurando modelo...")
    model, tokenizer = setup_model_and_tokenizer(config)
    
    # Tokenização
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            padding=True,
            max_length=config['dataset']['max_length'],
            return_tensors=None  # Não retornar tensors aqui
        )
    
    tokenized_datasets = dataset_splits.map(
        tokenize_function, 
        batched=True,
        remove_columns=dataset_splits['train'].column_names
    )
    
    # Data collator com padding dinâmico
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,  # Otimização para hardware
        return_tensors="pt"
    )
    
    # Configurar argumentos de treinamento
    training_args = TrainingArguments(
        output_dir=config['output']['output_dir'],
        run_name=config['output']['run_name'],
        num_train_epochs=config['training']['num_epochs'],
        per_device_train_batch_size=config['training']['batch_size'],
        per_device_eval_batch_size=config['training']['batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        lr_scheduler_type=config['training']['lr_scheduler'],
        warmup_steps=config['training']['warmup_steps'],
        logging_steps=config['training']['logging_steps'],
        save_steps=config['training']['save_steps'],
        eval_steps=config['training']['eval_steps'],
        eval_strategy=config['evaluation']['strategy'],
        save_total_limit=config['output']['save_total_limit'],
        load_best_model_at_end=config['evaluation']['load_best_model_at_end'],
        metric_for_best_model=config['evaluation']['metric_for_best_model'],
        greater_is_better=config['evaluation']['greater_is_better'],
        fp16=config['optimization']['use_fp16'],
        gradient_checkpointing=config['training']['gradient_checkpointing'],
        dataloader_num_workers=0,  # Evitar problemas de multiprocessing 
        group_by_length=config['training']['group_by_length'],
        logging_dir=config['output']['logging_dir'],
        report_to=[],  # Desabilita wandb se não configurado
        remove_unused_columns=config['hardware']['remove_unused_columns']
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test'],
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    # Treinamento
    print("🎯 Iniciando treinamento...")
    trainer.train()
    
    # Salvar modelo final
    print("💾 Salvando modelo...")
    trainer.save_model()
    tokenizer.save_pretrained(config['output']['output_dir'])
    
    print("✅ Fine-tuning concluído!")
    print(f"📁 Modelo salvo em: {config['output']['output_dir']}")

if __name__ == "__main__":
    main()
