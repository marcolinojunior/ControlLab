"""
Módulo de Conversões entre Representações
=========================================

Este módulo implementa conversões entre diferentes representações de sistemas
(função de transferência, espaço de estados) e operações entre sistemas.

Funções:
    tf_to_ss: Converte função de transferência para espaço de estados
    ss_to_tf: Converte espaço de estados para função de transferência
    parallel_to_series: Conecta sistemas em paralelo
    series_to_parallel: Converte conexão série para paralela
    feedback_connection: Implementa realimentação
"""

import sympy as sp
import numpy as np
from sympy import symbols, Matrix, eye, zeros, solve, simplify
from typing import Dict, List, Tuple, Any, Optional, Union
import warnings

# Importar classes do core se disponíveis
try:
    from ..core.symbolic_tf import SymbolicTransferFunction
    from ..core.symbolic_ss import SymbolicStateSpace
except ImportError:
    # Fallback para desenvolvimento
    SymbolicTransferFunction = None
    SymbolicStateSpace = None


class ConversionHistory:
    """Histórico de conversões para fins pedagógicos"""
    
    def __init__(self):
        self.steps = []
        self.method = ""
        self.original_form = ""
        self.target_form = ""
        
    def add_step(self, description: str, result: Any, explanation: str = ""):
        step = {
            'step': len(self.steps) + 1,
            'description': description,
            'result': result,
            'explanation': explanation
        }
        self.steps.append(step)
        
    def get_formatted_report(self) -> str:
        if not self.steps:
            return "Nenhuma conversão registrada."
            
        report = f"🔄 CONVERSÃO {self.original_form} → {self.target_form}\n"
        report += "=" * 60 + "\n"
        report += f"Método utilizado: {self.method}\n\n"
        
        for step in self.steps:
            report += f"📋 Passo {step['step']}: {step['description']}\n"
            report += f"Resultado: {step['result']}\n"
            if step['explanation']:
                report += f"Explicação: {step['explanation']}\n"
            report += "-" * 40 + "\n"
            
        return report


def tf_to_ss(tf_obj, form='controllable'):
    """
    Converte função de transferência para representação em espaço de estados
    
    Args:
        tf_obj: Objeto SymbolicTransferFunction ou expressão simbólica
        form: Forma canônica ('controllable', 'observable')
        
    Returns:
        Objeto SymbolicStateSpace ou tupla (A, B, C, D)
    """
    history = ConversionHistory()
    history.original_form = "Função de Transferência"
    history.target_form = "Espaço de Estados"
    history.method = f"Forma canônica {form}"
    
    try:
        # Extrair numerador e denominador
        if hasattr(tf_obj, 'numerator') and hasattr(tf_obj, 'denominator'):
            num = tf_obj.numerator
            den = tf_obj.denominator
            s = tf_obj.variable
        else:
            # Assumir que é uma expressão simbólica
            num = sp.numer(tf_obj)
            den = sp.denom(tf_obj)
            s = list(tf_obj.free_symbols)[0]
        
        history.add_step(
            "Extração de numerador e denominador",
            f"N(s) = {num}, D(s) = {den}",
            "Identificando polinômios da função de transferência"
        )
        
        # Obter coeficientes
        num_coeffs = sp.Poly(num, s).all_coeffs()
        den_coeffs = sp.Poly(den, s).all_coeffs()
        
        # Normalizar pelo coeficiente principal do denominador
        leading_coeff = den_coeffs[0]
        den_coeffs = [c/leading_coeff for c in den_coeffs]
        num_coeffs = [c/leading_coeff for c in num_coeffs]
        
        history.add_step(
            "Normalização de coeficientes",
            f"Denominador normalizado: {den_coeffs}",
            "Dividindo por coeficiente principal"
        )
        
        n = len(den_coeffs) - 1  # Ordem do sistema
        m = len(num_coeffs) - 1  # Ordem do numerador
        
        # Garantir que numerador tenha mesma ordem (padding com zeros)
        if m < n:
            num_coeffs = [0] * (n - m) + num_coeffs
        
        if form.lower() == 'controllable':
            A, B, C, D = _controllable_canonical_form(den_coeffs, num_coeffs, n)
            form_name = "controlável"
        elif form.lower() == 'observable':
            A, B, C, D = _observable_canonical_form(den_coeffs, num_coeffs, n)
            form_name = "observável"
        else:
            raise ValueError(f"Forma '{form}' não reconhecida. Use 'controllable' ou 'observable'")
        
        history.add_step(
            f"Construção da forma canônica {form_name}",
            f"Matrizes A({n}x{n}), B({n}x1), C(1x{n}), D(1x1)",
            f"Representação em espaço de estados na forma {form_name}"
        )
        
        history.add_step(
            "Resultado final",
            f"A = {A}\nB = {B}\nC = {C}\nD = {D}",
            "Sistema convertido com sucesso"
        )
        
        # Retornar dicionário com matrizes e histórico
        result = {
            'A': A,
            'B': B, 
            'C': C,
            'D': D,
            'history': history,
            'form': form
        }
        
        # Adicionar objeto SymbolicStateSpace se disponível
        if SymbolicStateSpace:
            ss_obj = SymbolicStateSpace(A, B, C, D)
            ss_obj.conversion_history = history
            result['ss_object'] = ss_obj
            
        return result
            
    except Exception as e:
        history.add_step(
            "Erro na conversão",
            str(e),
            "Falha durante o processo de conversão"
        )
        raise ValueError(f"Erro na conversão TF→SS: {e}")


def ss_to_tf(ss_obj):
    """
    Converte representação em espaço de estados para função de transferência
    
    Args:
        ss_obj: Objeto SymbolicStateSpace ou tupla (A, B, C, D)
        
    Returns:
        Função de transferência simbólica
    """
    history = ConversionHistory()
    history.original_form = "Espaço de Estados"
    history.target_form = "Função de Transferência"
    history.method = "Fórmula G(s) = C(sI - A)⁻¹B + D"
    
    try:
        # Extrair matrizes
        if hasattr(ss_obj, 'A'):
            A, B, C, D = ss_obj.A, ss_obj.B, ss_obj.C, ss_obj.D
        else:
            A, B, C, D = ss_obj
        
        history.add_step(
            "Extração de matrizes",
            f"A = {A}, B = {B}, C = {C}, D = {D}",
            "Identificando matrizes do sistema em espaço de estados"
        )
        
        s = symbols('s')
        n = A.shape[0]
        
        # Calcular (sI - A)
        sI_minus_A = s * eye(n) - A
        
        history.add_step(
            "Cálculo de (sI - A)",
            sI_minus_A,
            "Matriz característica do sistema"
        )
        
        # Calcular inversa (sI - A)⁻¹
        try:
            inv_sI_minus_A = sI_minus_A.inv()
            
            history.add_step(
                "Cálculo da inversa",
                inv_sI_minus_A,
                "Inversa da matriz característica"
            )
        except Exception:
            # Método alternativo usando determinante
            det_sI_minus_A = sI_minus_A.det()
            adj_sI_minus_A = sI_minus_A.adjugate()
            inv_sI_minus_A = adj_sI_minus_A / det_sI_minus_A
            
            history.add_step(
                "Cálculo da inversa (método adjugado)",
                inv_sI_minus_A,
                "Usando matriz adjugada e determinante"
            )
        
        # Calcular G(s) = C(sI - A)⁻¹B + D
        tf_matrix = C * inv_sI_minus_A * B + D
        
        # Para sistemas SISO, extrair elemento escalar
        if tf_matrix.shape == (1, 1):
            tf_expr = tf_matrix[0, 0]
        else:
            tf_expr = tf_matrix
        
        # Simplificar a expressão
        tf_simplified = simplify(tf_expr)
        
        history.add_step(
            "Cálculo da função de transferência",
            tf_simplified,
            "G(s) = C(sI - A)⁻¹B + D simplificada"
        )
        
        # Retornar objeto SymbolicTransferFunction se disponível
        if SymbolicTransferFunction and tf_matrix.shape == (1, 1):
            tf_obj = SymbolicTransferFunction(sp.numer(tf_simplified), sp.denom(tf_simplified), s)
            tf_obj.conversion_history = history
            return tf_obj
        else:
            return tf_simplified, history
            
    except Exception as e:
        history.add_step(
            "Erro na conversão",
            str(e),
            "Falha durante o processo de conversão"
        )
        raise ValueError(f"Erro na conversão SS→TF: {e}")


def parallel_to_series(tf_list):
    """
    Converte conexão em paralelo para equivalente em série
    
    Args:
        tf_list: Lista de funções de transferência em paralelo
        
    Returns:
        Função de transferência equivalente
    """
    if not tf_list:
        raise ValueError("Lista de funções de transferência não pode estar vazia")
    
    # Soma das funções de transferência
    result = tf_list[0]
    for tf in tf_list[1:]:
        result = result + tf
    
    return simplify(result)


def series_to_parallel(tf_cascaded):
    """
    Analisa conexão em série para identificar componentes
    
    Args:
        tf_cascaded: Função de transferência resultante da cascata
        
    Returns:
        Análise da decomposição (informativo)
    """
    # Esta função é mais conceitual - uma TF em série é o produto
    # Decompor o produto em fatores seria o equivalente
    
    s = list(tf_cascaded.free_symbols)[0]
    
    # Tentar fatorar numerador e denominador
    num = sp.numer(tf_cascaded)
    den = sp.denom(tf_cascaded)
    
    num_factors = sp.factor(num)
    den_factors = sp.factor(den)
    
    analysis = {
        'original': tf_cascaded,
        'numerator_factors': num_factors,
        'denominator_factors': den_factors,
        'note': 'Fatoração pode revelar componentes individuais da cascata'
    }
    
    return analysis


def feedback_connection(forward_tf, feedback_tf=1, sign=-1):
    """
    Implementa conexão com realimentação
    
    Args:
        forward_tf: Função de transferência direta G(s)
        feedback_tf: Função de transferência de realimentação H(s)
        sign: Sinal da realimentação (+1 positiva, -1 negativa)
        
    Returns:
        Função de transferência de malha fechada
    """
    history = ConversionHistory()
    history.original_form = "Sistema em Malha Aberta"
    history.target_form = "Sistema em Malha Fechada"
    history.method = f"Realimentação {'negativa' if sign == -1 else 'positiva'}"
    
    try:
        history.add_step(
            "Configuração inicial",
            f"G(s) = {forward_tf}, H(s) = {feedback_tf}",
            f"Realimentação com sinal {'+' if sign == 1 else '-'}"
        )
        
        # Fórmula de malha fechada: T(s) = G(s) / (1 ± G(s)H(s))
        denominator = 1 + sign * forward_tf * feedback_tf
        closed_loop_tf = forward_tf / denominator
        
        history.add_step(
            "Aplicação da fórmula",
            f"T(s) = G(s) / (1 {'+' if sign == 1 else '-'} G(s)H(s))",
            "Fórmula padrão para sistemas realimentados"
        )
        
        # Simplificar resultado
        simplified_tf = simplify(closed_loop_tf)
        
        history.add_step(
            "Resultado simplificado",
            simplified_tf,
            "Função de transferência de malha fechada"
        )
        
        # Adicionar histórico ao resultado se possível
        if hasattr(simplified_tf, '__dict__'):
            simplified_tf.feedback_history = history
        
        return simplified_tf, history
        
    except Exception as e:
        history.add_step(
            "Erro na conexão",
            str(e),
            "Falha durante o cálculo da realimentação"
        )
        raise ValueError(f"Erro na conexão com realimentação: {e}")


def _controllable_canonical_form(den_coeffs, num_coeffs, n):
    """Constrói forma canônica controlável"""
    # Matriz A (forma companion)
    A = zeros(n, n)
    
    # Última linha: -coeficientes do denominador (exceto o principal)
    for i in range(n):
        A[n-1, i] = -den_coeffs[n-i]
    
    # Superdiagonal com 1s
    for i in range(n-1):
        A[i, i+1] = 1
    
    # Matriz B
    B = zeros(n, 1)
    B[n-1, 0] = 1
    
    # Matriz C
    C = zeros(1, n)
    for i in range(n):
        if i < len(num_coeffs):
            C[0, i] = num_coeffs[n-1-i] - num_coeffs[0] * den_coeffs[n-i]
    
    # Matriz D
    D = Matrix([[num_coeffs[0]]])
    
    return A, B, C, D


def _observable_canonical_form(den_coeffs, num_coeffs, n):
    """Constrói forma canônica observável"""
    # Transposta da forma controlável
    A_c, B_c, C_c, D_c = _controllable_canonical_form(den_coeffs, num_coeffs, n)
    
    # A^T, C^T, B^T para obter forma observável
    A = A_c.T
    B = C_c.T
    C = B_c.T
    D = D_c
    
    return A, B, C, D


# Funções auxiliares para análise de sistemas
def analyze_system_connection(tf1, tf2, connection_type='series'):
    """
    Analisa a conexão entre dois sistemas
    
    Args:
        tf1, tf2: Funções de transferência
        connection_type: 'series', 'parallel', 'feedback'
        
    Returns:
        Análise da conexão
    """
    analysis = {
        'system1': tf1,
        'system2': tf2,
        'connection': connection_type
    }
    
    if connection_type == 'series':
        result = tf1 * tf2
        analysis['result'] = simplify(result)
        analysis['description'] = "Multiplicação das funções de transferência"
        
    elif connection_type == 'parallel':
        result = tf1 + tf2
        analysis['result'] = simplify(result)
        analysis['description'] = "Soma das funções de transferência"
        
    elif connection_type == 'feedback':
        result = tf1 / (1 + tf1 * tf2)
        analysis['result'] = simplify(result)
        analysis['description'] = "Realimentação negativa unitária"
        
    else:
        raise ValueError(f"Tipo de conexão '{connection_type}' não reconhecido")
    
    return analysis


def validate_conversion(original, converted, conversion_type):
    """
    Valida uma conversão entre representações
    
    Args:
        original: Sistema original
        converted: Sistema convertido
        conversion_type: Tipo de conversão realizada
        
    Returns:
        Resultado da validação
    """
    validation = {
        'conversion_type': conversion_type,
        'success': False,
        'errors': [],
        'warnings': []
    }
    
    try:
        if conversion_type == 'tf_to_ss_to_tf':
            # Converter de volta e comparar
            recovered_tf = ss_to_tf(converted)
            
            # Comparar simbolicamente
            diff = simplify(original - recovered_tf)
            if diff == 0:
                validation['success'] = True
            else:
                validation['errors'].append(f"Diferença na conversão: {diff}")
                
        validation['success'] = len(validation['errors']) == 0
        
    except Exception as e:
        validation['errors'].append(f"Erro na validação: {e}")
    
    return validation
