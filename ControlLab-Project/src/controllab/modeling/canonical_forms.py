"""
Módulo de Formas Canônicas
==========================

Este módulo implementa diferentes formas canônicas para representação
de sistemas em espaço de estados com explicações pedagógicas.

Funções:
    controllable_canonical: Forma canônica controlável
    observable_canonical: Forma canônica observável  
    modal_canonical: Forma canônica modal (diagonal)
    jordan_canonical: Forma canônica de Jordan
"""

import sympy as sp
from sympy import symbols, Matrix, eye, zeros, diag, solve, simplify
try:
    from sympy import jordan_form
except ImportError:
    # Para versões mais antigas do SymPy - definir fallback
    def jordan_form(matrix):
        """Fallback para jordan_form se não disponível"""
        try:
            # Usar método da matriz para diagonalização
            P, J = matrix.diagonalize()
            return P, J
        except:
            return Matrix.eye(matrix.rows), matrix
from typing import Dict, List, Tuple, Any, Optional
import warnings

# Importar classes do core se disponíveis
try:
    from ..core.symbolic_tf import SymbolicTransferFunction
    from ..core.symbolic_ss import SymbolicStateSpace
except ImportError:
    # Fallback para desenvolvimento
    SymbolicTransferFunction = None
    SymbolicStateSpace = None


class CanonicalFormHistory:
    """Histórico da transformação para forma canônica"""
    
    def __init__(self):
        self.steps = []
        self.transformation_matrix = None
        self.inverse_transformation = None
        self.canonical_type = ""
        
    def add_step(self, description: str, result: Any, explanation: str = ""):
        step = {
            'step': len(self.steps) + 1,
            'description': description,
            'result': result,
            'explanation': explanation
        }
        self.steps.append(step)
        
    def get_formatted_explanation(self) -> str:
        if not self.steps:
            return "Nenhuma transformação canônica registrada."
            
        explanation = f"🏗️ TRANSFORMAÇÃO PARA FORMA CANÔNICA {self.canonical_type.upper()}\n"
        explanation += "=" * 70 + "\n"
        
        for step in self.steps:
            explanation += f"\n📋 Passo {step['step']}: {step['description']}\n"
            explanation += f"Resultado: {step['result']}\n"
            if step['explanation']:
                explanation += f"Explicação: {step['explanation']}\n"
            explanation += "-" * 50 + "\n"
        
        if self.transformation_matrix is not None:
            explanation += f"\n🔧 MATRIZ DE TRANSFORMAÇÃO:\n"
            explanation += f"T = {self.transformation_matrix}\n"
            explanation += f"T⁻¹ = {self.inverse_transformation}\n"
            
        return explanation


def controllable_canonical(tf_obj):
    """
    Converte para forma canônica controlável
    
    Args:
        tf_obj: Função de transferência ou sistema em espaço de estados
        
    Returns:
        Sistema na forma canônica controlável
    """
    history = CanonicalFormHistory()
    history.canonical_type = "controlável"
    
    try:
        # Se já é um sistema SS, extrair TF primeiro
        if hasattr(tf_obj, 'A'):
            # Converter SS para TF primeiro
            from .conversions import ss_to_tf
            tf_expr, _ = ss_to_tf(tf_obj)
            
            history.add_step(
                "Conversão SS→TF",
                tf_expr,
                "Obtendo função de transferência do sistema"
            )
        else:
            tf_expr = tf_obj
        
        # Extrair numerador e denominador
        if hasattr(tf_expr, 'numerator'):
            num = tf_expr.numerator
            den = tf_expr.denominator
            s = tf_expr.variable  # Corrigido: usar 'variable' em vez de 's'
        else:
            num = sp.numer(tf_expr)
            den = sp.denom(tf_expr)
            s = list(tf_expr.free_symbols)[0]
        
        history.add_step(
            "Extração de polinômios",
            f"N(s) = {num}, D(s) = {den}",
            "Identificando numerador e denominador"
        )
        
        # Obter coeficientes
        den_poly = sp.Poly(den, s)
        num_poly = sp.Poly(num, s)
        
        den_coeffs = den_poly.all_coeffs()
        num_coeffs = num_poly.all_coeffs()
        
        # Normalizar
        a0 = den_coeffs[0]  # Coeficiente principal
        den_coeffs = [c/a0 for c in den_coeffs]
        num_coeffs = [c/a0 for c in num_coeffs]
        
        n = len(den_coeffs) - 1  # Ordem do sistema
        
        history.add_step(
            "Normalização de coeficientes",
            f"Ordem n = {n}, coeficientes normalizados",
            "Dividindo por coeficiente principal do denominador"
        )
        
        # Ajustar comprimento do numerador
        if len(num_coeffs) < len(den_coeffs):
            num_coeffs = [0] * (len(den_coeffs) - len(num_coeffs)) + num_coeffs
        elif len(num_coeffs) > len(den_coeffs):
            raise ValueError("Função de transferência imprópria - grau num > grau den")
        
        # Construir matriz A (forma companion)
        A = zeros(n, n)
        
        # Última linha: -a₁, -a₂, ..., -aₙ
        for i in range(n):
            A[n-1, i] = -den_coeffs[i+1]  # Pular a₀ = 1
        
        # Superdiagonal de 1s
        for i in range(n-1):
            A[i, i+1] = 1
        
        history.add_step(
            "Construção da matriz A",
            A,
            "Matriz A na forma companion (controlável)"
        )
        
        # Matriz B
        B = zeros(n, 1)
        B[n-1, 0] = 1
        
        history.add_step(
            "Construção da matriz B",
            B,
            "Vetor B com 1 na última posição"
        )
        
        # Matriz C
        C = zeros(1, n)
        
        # Coeficientes da matriz C
        for i in range(n):
            if i < len(num_coeffs):
                # C[i] = bᵢ - b₀*aᵢ onde b₀ é coef. principal do num
                C[0, i] = num_coeffs[n-i] - num_coeffs[0] * den_coeffs[n-i]
        
        history.add_step(
            "Construção da matriz C",
            C,
            "Matriz C baseada nos coeficientes do numerador"
        )
        
        # Matriz D
        D = Matrix([[num_coeffs[0]]])
        
        history.add_step(
            "Construção da matriz D",
            D,
            "Matriz D = coeficiente principal do numerador"
        )
        
        # Criar resultado
        if SymbolicStateSpace:
            result = SymbolicStateSpace(A, B, C, D)
            result.canonical_form = "controllable"
            result.transformation_history = history
            return result
        else:
            return (A, B, C, D), history
            
    except Exception as e:
        history.add_step(
            "Erro na transformação",
            str(e),
            "Falha durante a conversão para forma controlável"
        )
        raise ValueError(f"Erro na forma canônica controlável: {e}")


def observable_canonical(tf_obj):
    """
    Converte para forma canônica observável
    
    Args:
        tf_obj: Função de transferência ou sistema em espaço de estados
        
    Returns:
        Sistema na forma canônica observável
    """
    history = CanonicalFormHistory()
    history.canonical_type = "observável"
    
    try:
        # Obter forma controlável primeiro
        controllable_result = controllable_canonical(tf_obj)
        
        if isinstance(controllable_result, tuple):
            A_c, B_c, C_c, D_c = controllable_result[0]
        else:
            A_c, B_c, C_c, D_c = controllable_result.A, controllable_result.B, controllable_result.C, controllable_result.D
        
        history.add_step(
            "Obtenção da forma controlável",
            "Forma controlável calculada",
            "Base para transformação observável"
        )
        
        # Forma observável é a transposta da controlável
        A = A_c.T
        B = C_c.T  
        C = B_c.T
        D = D_c
        
        history.add_step(
            "Transposição das matrizes",
            f"A_obs = A_ctrl^T, B_obs = C_ctrl^T, C_obs = B_ctrl^T",
            "Relação entre formas controlável e observável"
        )
        
        history.add_step(
            "Resultado final",
            f"A = {A}\nB = {B}\nC = {C}\nD = {D}",
            "Sistema na forma canônica observável"
        )
        
        # Criar resultado
        if SymbolicStateSpace:
            result = SymbolicStateSpace(A, B, C, D)
            result.canonical_form = "observable"
            result.transformation_history = history
            return result
        else:
            return (A, B, C, D), history
            
    except Exception as e:
        history.add_step(
            "Erro na transformação",
            str(e),
            "Falha durante a conversão para forma observável"
        )
        raise ValueError(f"Erro na forma canônica observável: {e}")


def modal_canonical(ss_obj):
    """
    Converte para forma canônica modal (diagonal)
    
    Args:
        ss_obj: Sistema em espaço de estados
        
    Returns:
        Sistema na forma canônica modal
    """
    history = CanonicalFormHistory()
    history.canonical_type = "modal"
    
    try:
        # Extrair matrizes
        if hasattr(ss_obj, 'A'):
            A, B, C, D = ss_obj.A, ss_obj.B, ss_obj.C, ss_obj.D
        else:
            A, B, C, D = ss_obj
        
        history.add_step(
            "Sistema original",
            f"A = {A}",
            "Matriz A do sistema original"
        )
        
        # Calcular autovalores e autovetores
        try:
            P, J = A.diagonalize()
            
            history.add_step(
                "Diagonalização",
                f"Autovalores: {J.diagonal()}\nAutovetores: {P}",
                "Decomposição A = P*J*P⁻¹"
            )
            
            # Transformação modal: z = P⁻¹*x
            P_inv = P.inv()
            
            # Novas matrizes
            A_modal = J  # Matriz diagonal com autovalores
            B_modal = P_inv * B
            C_modal = C * P
            D_modal = D
            
            history.transformation_matrix = P_inv
            history.inverse_transformation = P
            
            history.add_step(
                "Transformação das matrizes",
                f"A_modal = J (diagonal)\nB_modal = P⁻¹*B\nC_modal = C*P",
                "Aplicando transformação z = P⁻¹*x"
            )
            
        except Exception:
            # Se não é diagonalizável, usar forma de Jordan
            warnings.warn("Matriz não diagonalizável, usando forma de Jordan")
            return jordan_canonical(ss_obj)
        
        history.add_step(
            "Resultado final",
            f"A = {A_modal}\nB = {B_modal}\nC = {C_modal}\nD = {D_modal}",
            "Sistema na forma canônica modal"
        )
        
        # Criar resultado
        if SymbolicStateSpace:
            result = SymbolicStateSpace(A_modal, B_modal, C_modal, D_modal)
            result.canonical_form = "modal"
            result.transformation_history = history
            result.eigenvalues = J.diagonal()
            return result
        else:
            return (A_modal, B_modal, C_modal, D_modal), history
            
    except Exception as e:
        history.add_step(
            "Erro na transformação",
            str(e),
            "Falha durante a conversão para forma modal"
        )
        raise ValueError(f"Erro na forma canônica modal: {e}")


def jordan_canonical(ss_obj):
    """
    Converte para forma canônica de Jordan
    
    Args:
        ss_obj: Sistema em espaço de estados
        
    Returns:
        Sistema na forma canônica de Jordan
    """
    history = CanonicalFormHistory()
    history.canonical_type = "Jordan"
    
    try:
        # Extrair matrizes
        if hasattr(ss_obj, 'A'):
            A, B, C, D = ss_obj.A, ss_obj.B, ss_obj.C, ss_obj.D
        else:
            A, B, C, D = ss_obj
        
        history.add_step(
            "Sistema original",
            f"A = {A}",
            "Matriz A do sistema original"
        )
        
        # Calcular forma de Jordan
        P, J = jordan_form(A)
        
        history.add_step(
            "Forma de Jordan",
            f"J = {J}\nP = {P}",
            "Decomposição A = P*J*P⁻¹ com blocos de Jordan"
        )
        
        # Transformação: z = P⁻¹*x
        P_inv = P.inv()
        
        # Novas matrizes
        A_jordan = J
        B_jordan = P_inv * B
        C_jordan = C * P
        D_jordan = D
        
        history.transformation_matrix = P_inv
        history.inverse_transformation = P
        
        history.add_step(
            "Transformação das matrizes",
            f"A_jordan = J\nB_jordan = P⁻¹*B\nC_jordan = C*P",
            "Aplicando transformação z = P⁻¹*x"
        )
        
        # Analisar estrutura dos blocos de Jordan
        jordan_blocks = _analyze_jordan_blocks(J)
        
        history.add_step(
            "Análise dos blocos de Jordan",
            jordan_blocks,
            "Identificando autovalores e suas multiplicidades"
        )
        
        history.add_step(
            "Resultado final",
            f"A = {A_jordan}\nB = {B_jordan}\nC = {C_jordan}\nD = {D_jordan}",
            "Sistema na forma canônica de Jordan"
        )
        
        # Criar resultado
        if SymbolicStateSpace:
            result = SymbolicStateSpace(A_jordan, B_jordan, C_jordan, D_jordan)
            result.canonical_form = "jordan"
            result.transformation_history = history
            result.jordan_blocks = jordan_blocks
            return result
        else:
            return (A_jordan, B_jordan, C_jordan, D_jordan), history
            
    except Exception as e:
        history.add_step(
            "Erro na transformação",
            str(e),
            "Falha durante a conversão para forma de Jordan"
        )
        raise ValueError(f"Erro na forma canônica de Jordan: {e}")


def _analyze_jordan_blocks(jordan_matrix):
    """Analisa a estrutura dos blocos de Jordan"""
    n = jordan_matrix.shape[0]
    blocks = []
    
    i = 0
    while i < n:
        # Autovalor na diagonal
        eigenvalue = jordan_matrix[i, i]
        
        # Determinar tamanho do bloco
        block_size = 1
        j = i + 1
        
        while j < n and jordan_matrix[j, j] == eigenvalue:
            # Verificar se há 1 na superdiagonal
            if j > i and jordan_matrix[i + block_size - 1, i + block_size] == 1:
                block_size += 1
            else:
                break
            j += 1
        
        blocks.append({
            'eigenvalue': eigenvalue,
            'size': block_size,
            'algebraic_multiplicity': block_size,
            'start_index': i
        })
        
        i += block_size
    
    return blocks


def compare_canonical_forms(tf_obj):
    """
    Compara diferentes formas canônicas do mesmo sistema
    
    Args:
        tf_obj: Função de transferência
        
    Returns:
        Dicionário com todas as formas canônicas
    """
    comparison = {
        'original': tf_obj,
        'forms': {}
    }
    
    try:
        # Forma controlável
        controllable_result = controllable_canonical(tf_obj)
        comparison['forms']['controllable'] = controllable_result
        
        # Forma observável
        observable_result = observable_canonical(tf_obj)
        comparison['forms']['observable'] = observable_result
        
        # Tentar forma modal (requer conversão para SS primeiro)
        try:
            if isinstance(controllable_result, tuple):
                ss_for_modal = controllable_result[0]
            else:
                ss_for_modal = controllable_result
                
            modal_result = modal_canonical(ss_for_modal)
            comparison['forms']['modal'] = modal_result
        except Exception as e:
            comparison['forms']['modal'] = f"Erro: {e}"
        
        # Análise comparativa
        comparison['analysis'] = {
            'controllable_properties': "Entrada concentrada, fácil análise de controlabilidade",
            'observable_properties': "Saída concentrada, fácil análise de observabilidade", 
            'modal_properties': "Modos desacoplados, comportamento natural visível"
        }
        
    except Exception as e:
        comparison['error'] = str(e)
    
    return comparison


def get_canonical_form_documentation():
    """
    Retorna documentação pedagógica sobre formas canônicas
    
    Returns:
        String com documentação detalhada
    """
    doc = """
📚 DOCUMENTAÇÃO: FORMAS CANÔNICAS EM ESPAÇO DE ESTADOS
======================================================

🏗️ FORMA CANÔNICA CONTROLÁVEL:
-----------------------------
• Estrutura: Matriz A em forma "companion"
• Características:
  - Última linha contém coeficientes da eq. característica
  - Superdiagonal de 1s
  - Entrada concentrada na última posição (B = [0 0 ... 1]ᵀ)
• Vantagens:
  - Fácil verificação de controlabilidade
  - Relação direta com função de transferência
  - Implementação simples de controladores

🔍 FORMA CANÔNICA OBSERVÁVEL:
----------------------------
• Estrutura: Transposta da forma controlável
• Características:
  - Primeira coluna contém coeficientes da eq. característica
  - Subdiagonal de 1s
  - Saída concentrada na primeira posição (C = [1 0 ... 0])
• Vantagens:
  - Fácil verificação de observabilidade
  - Projeto eficiente de observadores
  - Dualidade com forma controlável

⚡ FORMA CANÔNICA MODAL:
-----------------------
• Estrutura: Matriz A diagonal (autovalores)
• Características:
  - Estados são modos naturais do sistema
  - Dinâmica desacoplada
  - Matriz A = diag(λ₁, λ₂, ..., λₙ)
• Vantagens:
  - Interpretação física clara
  - Análise de estabilidade direta
  - Projeto modal de controladores

🔗 FORMA CANÔNICA DE JORDAN:
---------------------------
• Estrutura: Blocos de Jordan na diagonal
• Características:
  - Generalização da forma modal
  - Trata autovalores repetidos
  - Blocos Jᵢ para cada autovalor
• Vantagens:
  - Forma mais geral possível
  - Análise de multiplicidade geométrica
  - Base para teoria de sistemas lineares

🎯 ESCOLHA DA FORMA CANÔNICA:
----------------------------
• Controlável: Projeto de controladores
• Observável: Projeto de observadores
• Modal: Análise de estabilidade e modos
• Jordan: Análise teórica completa

💡 NOTA PEDAGÓGICA:
------------------
Todas as formas são matematicamente equivalentes e representam
o mesmo sistema físico. A escolha depende da aplicação específica
e da intuição desejada sobre o comportamento do sistema.
"""
    
    return doc
