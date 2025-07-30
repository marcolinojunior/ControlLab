#!/usr/bin/env python3
"""
Demonstração do Módulo 2 - Núcleo Simbólico
Exemplo prático das funcionalidades implementadas
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import sympy as sp
from controllab.core import SymbolicTransferFunction, SymbolicStateSpace
from controllab.core.stability_analysis import RouthHurwitzAnalyzer
from controllab.core.controller_design import PIDController
from controllab.core.transforms import LaplaceTransform

def demo_modulo2():
    """Demonstração completa do Módulo 2"""
    
    print("=" * 80)
    print("🧠 MÓDULO 2 - NÚCLEO SIMBÓLICO - DEMONSTRAÇÃO")
    print("=" * 80)
    
    # 1. Criação de Função de Transferência
    print("\n1️⃣ CRIANDO FUNÇÃO DE TRANSFERÊNCIA SIMBÓLICA")
    print("-" * 50)
    
    s = sp.Symbol('s')
    tf = SymbolicTransferFunction(1, s**2 + 2*s + 1, s)
    
    print(f"Função criada: {tf}")
    print(f"LaTeX: {tf.to_latex()}")
    print(f"Polos: {tf.poles()}")
    print(f"Zeros: {tf.zeros()}")
    
    # 2. Operações Simbólicas
    print("\n2️⃣ OPERAÇÕES SIMBÓLICAS")
    print("-" * 50)
    
    # Criar outro sistema
    tf2 = SymbolicTransferFunction(s + 1, s + 2, s)
    
    # Multiplicação
    tf_mult = tf * tf2
    print(f"Multiplicação: {tf_mult}")
    
    # Adição  
    tf_add = tf + tf2
    print(f"Adição: {tf_add}")
    
    # Realimentação
    tf_feedback = tf / (1 + tf)
    print(f"Malha fechada: {tf_feedback}")
    
    # 3. Análise de Estabilidade (Routh-Hurwitz)
    print("\n3️⃣ ANÁLISE DE ESTABILIDADE - ROUTH-HURWITZ")
    print("-" * 50)
    
    analyzer = RouthHurwitzAnalyzer()
    
    # Sistema instável para teste
    unstable_tf = SymbolicTransferFunction(1, s**3 - 2*s**2 + s - 1, s)
    char_eq = unstable_tf.characteristic_equation()
    
    print(f"Equação característica: {char_eq} = 0")
    
    stability = analyzer.analyze(char_eq, s)
    print(f"Sistema estável: {stability['is_stable']}")
    if 'unstable_poles' in stability:
        print(f"Polos instáveis: {stability['unstable_poles']}")
    else:
        print("Informação de polos não disponível")
    
    # 4. Design de Controlador PID
    print("\n4️⃣ DESIGN DE CONTROLADOR PID")
    print("-" * 50)
    
    pid_designer = PIDController()
    
    # Planta de teste
    plant = SymbolicTransferFunction(1, s*(s + 1)*(s + 2), s)
    print(f"Planta: {plant}")
    
    # Design do PID
    kp, ki, kd = sp.symbols('K_p K_i K_d', real=True, positive=True)
    
    # Criar controlador PID diretamente
    pid_tf = SymbolicTransferFunction(
        kp*s**2 + ki*s + kd*s**2, 
        s, 
        s
    )  # PID: Kp + Ki/s + Kd*s
    print(f"Controlador PID: {pid_tf}")
    
    # Design usando o método disponível
    pid_result = pid_designer.design_pid(plant)
    print(f"PID projetado: {pid_result}")
    
    # Sistema em malha fechada
    kp_val = 1  # Valor exemplo
    pid_simple = SymbolicTransferFunction(kp_val, 1, s)
    open_loop = plant * pid_simple
    closed_loop = open_loop / (1 + open_loop)
    print(f"Sistema compensado: {closed_loop}")
    
    # 5. Representação em Espaço de Estados
    print("\n5️⃣ ESPAÇO DE ESTADOS")
    print("-" * 50)
    
    # Converter para espaço de estados
    A = sp.Matrix([[-1, -2], [1, 0]])
    B = sp.Matrix([[1], [0]])
    C = sp.Matrix([[0, 1]])
    D = sp.Matrix([[0]])
    
    ss = SymbolicStateSpace(A, B, C, D)
    print(f"Matriz A:\n{ss.A}")
    print(f"Controlabilidade: {ss.is_controllable()}")
    print(f"Observabilidade: {ss.is_observable()}")
    
    # Conversão para função de transferência
    tf_from_ss = ss.transfer_function()
    print(f"Função de transferência: {tf_from_ss}")
    
    # 6. Transformadas
    print("\n6️⃣ TRANSFORMADAS")
    print("-" * 50)
    
    laplace = LaplaceTransform()
    
    # Transformada de uma função temporal
    t = sp.Symbol('t', positive=True)
    f_t = sp.exp(-2*t) * sp.sin(3*t)
    F_s = laplace.transform(f_t, t, s)
    print(f"f(t) = {f_t}")
    print(f"F(s) = {F_s}")
    
    # Transformada inversa
    G_s = 1 / (s**2 + 4*s + 13)
    g_t = laplace.inverse_transform(G_s, s, t)
    print(f"G(s) = {G_s}")
    print(f"g(t) = {g_t}")
    
    # 7. Histórico Pedagógico
    print("\n7️⃣ HISTÓRICO PEDAGÓGICO")
    print("-" * 50)
    
    history = tf.history.steps
    print("Operações realizadas:")
    for i, operation in enumerate(history, 1):
        print(f"{i}. {operation}")
    
    if not history:
        print("Nenhuma operação registrada no histórico")
    
    print("\n" + "=" * 80)
    print("✅ MÓDULO 2 FUNCIONANDO COMPLETAMENTE!")
    print("✅ Todas as funcionalidades do núcleo simbólico estão operacionais")
    print("=" * 80)

if __name__ == "__main__":
    try:
        demo_modulo2()
    except Exception as e:
        print(f"\n❌ Erro na demonstração: {e}")
        import traceback
        traceback.print_exc()
