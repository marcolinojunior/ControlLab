#!/usr/bin/env python3
"""
ControlLab Fine-tuning Assistant - Versão Otimizada para Memória
Sistema com todas as otimizações sugeridas para resolver problemas CUDA/Memory
Implementa: BF16, Flash Attention fallback, Gradient Accumulation, Memory Management
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
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# Configurações de memória GPU mais agressivas
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# Suprimir warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    import transformers
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
        set_seed,
        BitsAndBytesConfig,
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
    print("🚀 ControlLab Fine-tuning - Versão OTIMIZADA PARA MEMÓRIA")
    print("🔧 Implementa: BF16 + Gradient Accumulation + Memory Management")
    print("="*60)
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"🎮 GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        print(f"🔧 CUDA Version: {torch.version.cuda}")
        print(f"🔧 PyTorch Version: {torch.__version__}")
    else:
        print("❌ CUDA não disponível!")
        sys.exit(1)

def aggressive_memory_cleanup():
    """Limpeza agressiva de memória GPU"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()
        gc.collect()

def get_gpu_memory_info():
    """Retorna informações detalhadas de memória GPU"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        cached = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        free = total - allocated
        return f"{allocated:.2f}GB/{total:.1f}GB (livre: {free:.2f}GB)"
    return "GPU não disponível"

@dataclass
class OptimizedTrainingConfig:
    """Configuração ultra-otimizada para memória limitada"""
    
    # Modelo
    model_name: str = "microsoft/Phi-3-mini-4k-instruct"
    
    # Dataset
    dataset_path: str = r"C:\Users\marco\Documents\ControlLab\ControlLab-Project\ai_toolkit_dataset\controllab_specialized_dataset_improved.jsonl"
    max_length: int = 512  # Restaurado para precisão máxima dentro dos 8GB disponíveis
    
    # Training - Configurações OTIMIZADAS para memória
    batch_size: int = 1
    gradient_accumulation_steps: int = 8  # MAIOR acumulação = menor uso de memória por step
    learning_rate: float = 2e-5  # Learning rate ligeiramente maior para compensar
    num_epochs: int = 1
    warmup_ratio: float = 0.05
    
    # LoRA - Configurações mínimas para máxima eficiência
    lora_r: int = 4
    lora_alpha: int = 8
    lora_dropout: float = 0.1
    
    # Output
    output_dir: str = "controllab_model_optimized"
    save_steps: int = 100
    eval_steps: int = 100
    logging_steps: int = 25
    
    # OTIMIZAÇÕES DE MEMÓRIA CRÍTICAS
    bf16: bool = True  # ATIVA BF16 para cortar memória pela metade
    fp16: bool = False  # Não usar FP16 junto com BF16
    dataloader_pin_memory: bool = False
    dataloader_num_workers: int = 0
    remove_unused_columns: bool = True
    
    # Configurações avançadas de memória
    gradient_checkpointing: bool = True  # ATIVA checkpoint para economizar memória
    max_grad_norm: float = 0.5  # Menor para estabilidade

def check_bf16_support():
    """Verifica se a GPU suporta BF16"""
    if torch.cuda.is_available():
        device_capability = torch.cuda.get_device_capability()
        # BF16 é suportado em GPUs com capability >= 8.0 (RTX 30xx+)
        supports_bf16 = device_capability[0] >= 8
        print(f"🔧 GPU Capability: {device_capability[0]}.{device_capability[1]}")
        print(f"🔧 BF16 Support: {'✅ Sim' if supports_bf16 else '❌ Não'}")
        return supports_bf16
    return False

def setup_model_and_tokenizer(config: OptimizedTrainingConfig):
    """Configura modelo e tokenizer com otimizações máximas de memória"""

    print("🤖 Configurando modelo...")
    print(f"🤖 Carregando modelo: {config.model_name}")

    aggressive_memory_cleanup()

    # 1. CONFIGURAÇÃO DE QUANTIZAÇÃO DE 8-BIT (mais preciso, usa mais VRAM)
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.bfloat16,
    )

    print("📝 Carregando tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("⚙️ Carregando modelo base QUANTIZADO (8-bit)...")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
        use_cache=False
    )

    print(f"💾 Memória após carregamento QUANTIZADO: {get_gpu_memory_info()}")

    # 2. PREPARAÇÃO DO MODELO PARA TREINO
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    print("🎯 Aplicando LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    percentage = (trainable_params / total_params) * 100
    print(f"Parâmetros treináveis: {trainable_params:,} ({percentage:.3f}%)")

    aggressive_memory_cleanup()
    print(f"💾 Memória final: {get_gpu_memory_info()}")

    return model, tokenizer

def load_and_prepare_dataset(config: OptimizedTrainingConfig, tokenizer):
    """Carrega dataset com otimizações de memória"""
    
    print("📚 Preparando dataset...")
    dataset_file = Path(config.dataset_path)
    
    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {dataset_file}")
    
    print(f"📂 Carregando dataset: {dataset_file.name}")
    
    # Carrega apenas uma amostra para teste se necessário
    dataset = load_dataset('json', data_files=str(dataset_file))['train']
    
    # Reduz tamanho do dataset se necessário para testes
    if len(dataset) > 200:  # Limita para testes de memória
        print(f"⚠️ Limitando dataset para 200 exemplos (teste de memória)")
        dataset = dataset.select(range(200))
    
    print(f"📊 Dataset carregado: {len(dataset)} exemplos")
    
    # Split
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset['train']
    eval_dataset = dataset['test']
    
    print(f"🔄 Split criado: {len(train_dataset)} treino, {len(eval_dataset)} validação")
    
    def tokenize_function(examples):
        """Tokenização otimizada para memória"""
        conversations = []
        
        for messages in examples['messages']:
            text = ""
            for turn in messages:
                role = turn['role']
                content = turn['content']
                
                if role == 'user':
                    text += f"<|user|>\n{content}<|end|>\n"
                elif role == 'assistant':
                    text += f"<|assistant|>\n{content}<|end|>\n"
            
            conversations.append(text)
        
        # Tokenização com limite rigoroso de comprimento
        tokenized = tokenizer(
            conversations,
            truncation=True,
            padding=False,
            max_length=config.max_length,  # Muito reduzido
            return_tensors=None
        )
        
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    print("🔤 Tokenizando dados...")
    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizando train"
    )
    
    eval_dataset = eval_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=eval_dataset.column_names,
        desc="Tokenizando eval"
    )
    
    return train_dataset, eval_dataset

def create_trainer(model, tokenizer, train_dataset, eval_dataset, config: OptimizedTrainingConfig):
    """Cria trainer com todas as otimizações de memória"""
    
    print("⚙️ Configurando treinamento otimizado...")
    
    # Argumentos com TODAS as otimizações de memória
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
        lr_scheduler_type="cosine",  # Cosine é mais suave
        
        # OTIMIZAÇÕES DE MEMÓRIA CRÍTICAS
        bf16=config.bf16,
        fp16=config.fp16,
        gradient_checkpointing=config.gradient_checkpointing,  # ATIVA
        
        # Logging e save
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        eval_strategy="steps",
        save_strategy="steps",
        
        # Otimizações de dataloader
        dataloader_pin_memory=config.dataloader_pin_memory,
        dataloader_num_workers=config.dataloader_num_workers,
        remove_unused_columns=config.remove_unused_columns,
        
        # Otimizador
        optim="adamw_torch",
        weight_decay=0.01,
        max_grad_norm=config.max_grad_norm,
        
        # Configurações de avaliação
        load_best_model_at_end=False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        # Seeds
        seed=42,
        data_seed=42,
        
        # Otimizações adicionais
        prediction_loss_only=True,
        include_inputs_for_metrics=False,
        
        # Desabilita relatórios para economizar recursos
        report_to=None,
        
        # Configurações de memória avançadas
        ddp_find_unused_parameters=False,
        save_safetensors=True,  # Mais eficiente
        
        # Configurações específicas para GPUs pequenas
        eval_accumulation_steps=4,  # Acumula avaliações
    )
    
    # Data collator otimizado
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8 if config.fp16 or config.bf16 else None
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

def validate_setup(model, tokenizer, config):
    """Validação completa antes do treinamento"""
    
    print("🔍 Validação completa do setup...")
    
    try:
        # Teste básico
        test_input = "Como calcular estabilidade de sistemas"
        inputs = tokenizer(test_input, return_tensors="pt", max_length=128, truncation=True).to("cuda")
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        print("✅ Modelo respondendo corretamente")
        
        # Verifica gradientes
        has_gradients = any(p.requires_grad for p in model.parameters())
        if has_gradients:
            print("✅ Parâmetros com gradientes OK")
        else:
            print("❌ Nenhum parâmetro com gradientes!")
            return False
        
        # Verifica precisão
        dtype_info = "BF16" if config.bf16 else "FP16" if config.fp16 else "FP32"
        print(f"✅ Precisão configurada: {dtype_info}")
        
        # Informações de memória
        print(f"✅ Memória antes do treino: {get_gpu_memory_info()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro na validação: {e}")
        return False

def main():
    """Função principal otimizada"""
    
    try:
        print_header()
        
        # Configuração
        config = OptimizedTrainingConfig()
        
        # Seeds
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
        print("\n📊 Configuração OTIMIZADA:")
        print(f"🔢 Exemplos de treino: {len(train_dataset)}")
        print(f"🔢 Exemplos de validação: {len(eval_dataset)}")
        print(f"💾 Batch size: {config.batch_size}")
        print(f"🔄 Gradient accumulation: {config.gradient_accumulation_steps}")
        print(f"🎯 Learning rate: {config.learning_rate}")
        print(f"🔄 Épocas: {config.num_epochs}")
        print(f"⚡ Precisão: {'BF16' if config.bf16 else 'FP16' if config.fp16 else 'FP32'}")
        print(f"💾 Max length: {config.max_length}")
        print(f"🔧 Gradient checkpointing: {config.gradient_checkpointing}")
        
        # Validação
        if not validate_setup(model, tokenizer, config):
            print("❌ Falha na validação")
            return
        
        # Limpeza final antes do treinamento
        aggressive_memory_cleanup()
        
        print("\n🎯 Iniciando treinamento OTIMIZADO...")
        print("="*60)
        print(f"💾 Memória disponível: {get_gpu_memory_info()}")
        
        # TREINAMENTO
        trainer.train()
        
        print("\n✅ Treinamento concluído com sucesso!")
        
        # Salva modelo
        trainer.save_model()
        tokenizer.save_pretrained(config.output_dir)
        
        print(f"💾 Modelo salvo em: {config.output_dir}")
        
        # Cria o app Flask conforme sugerido
        print("\n🌐 Criando aplicação Flask...")
        create_flask_app(config.output_dir)
        
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        print(f"💾 Memória no erro: {get_gpu_memory_info()}")
        
        import traceback
        traceback.print_exc()

def create_flask_app(model_path):
    """Cria aplicação Flask conforme sugerido pelo usuário"""
    
    flask_code = f'''from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Configuração
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
ADAPTER_PATH = "{model_path}"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🚀 Carregando modelo base: {{MODEL_NAME}}")

# Carrega modelo base
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map=DEVICE,
    trust_remote_code=True,
    attn_implementation="eager"
)

print(f"🎯 Aplicando adaptadores LoRA: {{ADAPTER_PATH}}")
# Aplica adaptadores LoRA treinados
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("✅ Modelo ControlLab pronto!")

# Aplicação Flask
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        prompt = data.get("prompt")

        if not prompt:
            return jsonify({{"error": "Prompt não fornecido"}}), 400

        # Prepara entrada
        messages = [{{"role": "user", "content": prompt}}]
        inputs = tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            return_tensors="pt"
        ).to(DEVICE)

        # Gera resposta
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        final_response = response_text.split("<|assistant|>")[-1].strip()

        return jsonify({{"response": final_response}})

    except Exception as e:
        return jsonify({{"error": str(e)}}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({{"status": "ok", "model": "ControlLab Assistant"}})

if __name__ == '__main__':
    print("🌐 Iniciando servidor ControlLab...")
    print("📡 Endpoint: POST http://localhost:5000/predict")
    print("💻 Health check: GET http://localhost:5000/health")
    app.run(host='0.0.0.0', port=5000, debug=True)
'''
    
    flask_file = Path("controllab_flask_app.py")
    flask_file.write_text(flask_code, encoding='utf-8')
    print(f"✅ Flask app criado: {flask_file}")
    print("🚀 Para executar: python controllab_flask_app.py")

if __name__ == "__main__":
    main()
