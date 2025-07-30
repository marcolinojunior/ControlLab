#!/usr/bin/env python3
"""
Teste específico do conhecimento ControlLab
Verifica se o modelo compreende o contexto e funcionalidades do projeto
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def test_controllab_specific_knowledge():
    print("🔍 Testando conhecimento específico do ControlLab...")
    
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
        use_cache=False,
        attn_implementation="eager"
    )
    
    print("🎯 Carregando adaptação LoRA ControlLab...")
    model = PeftModel.from_pretrained(base_model, model_path)
    
    # Perguntas específicas do ControlLab baseadas na documentação
    controllab_questions = [
        "Como o ControlLab implementa a dualidade simbólico-numérica?",
        "Explique o papel do sympy.lambdify na arquitetura do ControlLab.",
        "Como o ControlLab trata casos especiais na tabela de Routh-Hurwitz?",
        "Qual é a vantagem da classe SymbolicTransferFunction sobre representações puramente numéricas?",
        "Como o ControlLab implementa a comunicação WebSocket para interatividade em tempo real?",
        "Explique como o ControlLab aplica a transformação de Tustin simbolicamente.",
        "Como funciona a fórmula de Ackermann no módulo de alocação de polos do ControlLab?",
        "Qual é o papel do NumericalInterface na arquitetura do ControlLab?",
        "Como o ControlLab constrói o contorno de Nyquist com indentações?",
        "Explique a arquitetura de plugin via Entry Points no ControlLab."
    ]
    
    print("\n" + "="*80)
    print("🧪 TESTE DE CONHECIMENTO ESPECÍFICO DO CONTROLLAB")
    print("="*80)
    
    for i, question in enumerate(controllab_questions, 1):
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
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=False,
                repetition_penalty=1.1
            )
        
        # Decodifica resposta
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        assistant_response = response.split("<|assistant|>")[-1].strip()
        
        print(f"🤖 Resposta: {assistant_response}")
        print("-" * 80)
        
        # Análise básica da qualidade da resposta
        keywords_controllab = [
            'SymPy', 'lambdify', 'simbólico', 'numérico', 'ControlLab', 
            'python-control', 'Routh', 'Tustin', 'Ackermann', 'WebSocket',
            'Entry Points', 'NumericalInterface', 'SymbolicTransferFunction'
        ]
        
        found_keywords = [kw for kw in keywords_controllab if kw.lower() in assistant_response.lower()]
        relevance_score = len(found_keywords) / len(keywords_controllab) * 100
        
        print(f"📊 Score de Relevância: {relevance_score:.1f}% (palavras-chave encontradas: {len(found_keywords)})")
        if found_keywords:
            print(f"🎯 Termos específicos encontrados: {', '.join(found_keywords)}")

if __name__ == "__main__":
    test_controllab_specific_knowledge()
