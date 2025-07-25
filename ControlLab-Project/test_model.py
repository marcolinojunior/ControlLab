#!/usr/bin/env python3
"""
Teste rápido do modelo ControlLab treinado
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def test_controllab_model():
    print("🔍 Testando modelo ControlLab treinado...")
    
    # Carrega modelo base
    base_model_name = "microsoft/Phi-3-mini-4k-instruct"
    model_path = "controllab_model_optimized"
    
    print("📝 Carregando tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("🤖 Carregando modelo base...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        use_cache=False,  # Desabilita cache para evitar erros
        attn_implementation="eager"  # Implementação mais estável
    )
    
    print("🎯 Carregando adaptação LoRA...")
    model = PeftModel.from_pretrained(base_model, model_path)
    
    # Teste com pergunta de controle
    test_questions = [
        "Explique o que é um sistema de controle em malha fechada.",
        "Como calcular a função de transferência de um sistema?",
        "O que é estabilidade em sistemas de controle?"
    ]
    
    print("\n" + "="*50)
    print("🧪 TESTE DO MODELO TREINADO")
    print("="*50)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n📌 Pergunta {i}: {question}")
        
        # Formata entrada
        prompt = f"<|user|>\n{question}<|end|>\n<|assistant|>\n"
        
        # Tokeniza
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Gera resposta
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=False,  # Evita problemas de cache
                repetition_penalty=1.1
            )
        
        # Decodifica resposta
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        assistant_response = response.split("<|assistant|>")[-1].strip()
        
        print(f"🤖 Resposta: {assistant_response}")
        print("-" * 50)

if __name__ == "__main__":
    test_controllab_model()
