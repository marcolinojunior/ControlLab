#!/usr/bin/env python3
"""
Conversões Avançadas TF ↔ SS - ControlLab Numerical
Conversões avançadas entre Transfer Function e State Space
"""

import sympy as sp
import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Any
import warnings

# Importações condicionais
try:
    import control
    CONTROL_AVAILABLE = True
except ImportError:
    CONTROL_AVAILABLE = False
    control = None

from ..core.symbolic_tf import SymbolicTransferFunction
from ..core.symbolic_ss import SymbolicStateSpace
from ..core.history import OperationHistory


class StateSpaceConverter:
    """
    Conversor avançado entre Transfer Function e State Space
    """
    
    def __init__(self):
        self.history = OperationHistory()
    
    def tf_to_ss_canonical_controllable(self, tf_system, substitutions=None):
        """
        Converte TF para SS na forma canônica controlável
        
        Args:
            tf_system: SymbolicTransferFunction
            substitutions: Substituições de símbolos
            
        Returns:
            SymbolicStateSpace: Sistema em forma canônica controlável
        """
        if not isinstance(tf_system, SymbolicTransferFunction):
            raise TypeError("tf_system deve ser SymbolicTransferFunction")
        
        self.history.add_step(
            "CONVERSÃO_TF_SS_CONTROLÁVEL",
            "Convertendo TF para forma canônica controlável",
            str(tf_system),
            "Iniciando conversão..."
        )
        
        try:
            s = tf_system.variable
            num = tf_system.numerator
            den = tf_system.denominator
            
            if substitutions:
                num = num.subs(substitutions)
                den = den.subs(substitutions)
            
            # Extrair coeficientes do denominador
            den_poly = sp.Poly(den, s)
            den_coeffs = den_poly.all_coeffs()
            n = len(den_coeffs) - 1  # Ordem do sistema
            
            if n == 0:
                raise ValueError("Sistema de ordem 0 não pode ser convertido para SS")
            
            # Normalizar coeficientes (coeficiente de s^n = 1)
            leading_coeff = den_coeffs[0]
            den_coeffs = [c / leading_coeff for c in den_coeffs]
            
            # Extrair coeficientes do numerador
            num_poly = sp.Poly(num, s)
            num_coeffs = num_poly.all_coeffs()
            
            # Pad numerador se necessário
            if len(num_coeffs) < n + 1:
                num_coeffs = [0] * (n + 1 - len(num_coeffs)) + num_coeffs
            elif len(num_coeffs) > n + 1:
                # Sistema impróprio - não implementado
                raise ValueError("Sistema impróprio não suportado")
            
            # Normalizar numerador
            num_coeffs = [c / leading_coeff for c in num_coeffs]
            
            # Construir matriz A (forma canônica controlável)
            A = sp.zeros(n, n)
            
            # Primeira linha: coeficientes negativos do denominador
            for j in range(n):
                A[0, j] = -den_coeffs[n - j]
            
            # Restante: matriz identidade deslocada
            for i in range(1, n):
                A[i, i-1] = 1
            
            # Construir matriz B
            B = sp.zeros(n, 1)
            B[0, 0] = 1
            
            # Construir matriz C
            C = sp.zeros(1, n)
            for j in range(n):
                if j < len(num_coeffs) - 1:
                    C[0, j] = num_coeffs[n - j] - num_coeffs[0] * den_coeffs[n - j]
            
            # Construir matriz D
            D = sp.Matrix([[num_coeffs[0]]])
            
            # Criar sistema no espaço de estados
            ss_system = SymbolicStateSpace(A, B, C, D)
            # Armazenar variável separadamente
            ss_system.variable = tf_system.variable
            
            result = {
                'original_tf': tf_system,
                'state_space': ss_system,
                'canonical_form': 'controllable',
                'order': n,
                'coefficients': {
                    'numerator': num_coeffs,
                    'denominator': den_coeffs
                }
            }
            
            self.history.add_step(
                "CONVERSÃO_CONCLUÍDA",
                f"TF convertida para SS controlável de ordem {n}",
                f"A: {A.shape}, B: {B.shape}, C: {C.shape}, D: {D.shape}",
                result
            )
            
            return ss_system
            
        except Exception as e:
            error_msg = f"Erro na conversão TF→SS controlável: {e}"
            self.history.add_step("ERRO_CONVERSÃO", error_msg, "", str(e))
            raise ValueError(error_msg) from e
    
    def tf_to_ss_canonical_observable(self, tf_system, substitutions=None):
        """
        Converte TF para SS na forma canônica observável
        
        Args:
            tf_system: SymbolicTransferFunction
            substitutions: Substituições de símbolos
            
        Returns:
            SymbolicStateSpace: Sistema em forma canônica observável
        """
        if not isinstance(tf_system, SymbolicTransferFunction):
            raise TypeError("tf_system deve ser SymbolicTransferFunction")
        
        self.history.add_step(
            "CONVERSÃO_TF_SS_OBSERVÁVEL",
            "Convertendo TF para forma canônica observável",
            str(tf_system),
            "Iniciando conversão..."
        )
        
        try:
            s = tf_system.variable
            num = tf_system.numerator
            den = tf_system.denominator
            
            if substitutions:
                num = num.subs(substitutions)
                den = den.subs(substitutions)
            
            # Extrair coeficientes
            den_poly = sp.Poly(den, s)
            den_coeffs = den_poly.all_coeffs()
            n = len(den_coeffs) - 1
            
            if n == 0:
                raise ValueError("Sistema de ordem 0 não pode ser convertido para SS")
            
            # Normalizar
            leading_coeff = den_coeffs[0]
            den_coeffs = [c / leading_coeff for c in den_coeffs]
            
            num_poly = sp.Poly(num, s)
            num_coeffs = num_poly.all_coeffs()
            
            if len(num_coeffs) < n + 1:
                num_coeffs = [0] * (n + 1 - len(num_coeffs)) + num_coeffs
            
            num_coeffs = [c / leading_coeff for c in num_coeffs]
            
            # Construir matriz A (forma canônica observável - transposta da controlável)
            A = sp.zeros(n, n)
            
            # Última coluna: coeficientes negativos do denominador
            for i in range(n):
                A[i, n-1] = -den_coeffs[n - i]
            
            # Restante: matriz identidade deslocada
            for i in range(n-1):
                A[i+1, i] = 1
            
            # Construir matriz B
            B = sp.zeros(n, 1)
            for i in range(n):
                if i < len(num_coeffs) - 1:
                    B[i, 0] = num_coeffs[n - i] - num_coeffs[0] * den_coeffs[n - i]
            
            # Construir matriz C
            C = sp.zeros(1, n)
            C[0, n-1] = 1
            
            # Construir matriz D
            D = sp.Matrix([[num_coeffs[0]]])
            
            # Criar sistema no espaço de estados
            ss_system = SymbolicStateSpace(A, B, C, D)
            # Armazenar variável separadamente  
            ss_system.variable = tf_system.variable
            
            result = {
                'original_tf': tf_system,
                'state_space': ss_system,
                'canonical_form': 'observable',
                'order': n,
                'coefficients': {
                    'numerator': num_coeffs,
                    'denominator': den_coeffs
                }
            }
            
            self.history.add_step(
                "CONVERSÃO_CONCLUÍDA",
                f"TF convertida para SS observável de ordem {n}",
                f"A: {A.shape}, B: {B.shape}, C: {C.shape}, D: {D.shape}",
                result
            )
            
            return ss_system
            
        except Exception as e:
            error_msg = f"Erro na conversão TF→SS observável: {e}"
            self.history.add_step("ERRO_CONVERSÃO", error_msg, "", str(e))
            raise ValueError(error_msg) from e
    
    def ss_to_tf_via_characteristic(self, ss_system, substitutions=None):
        """
        Converte SS para TF usando polinômio característico
        
        Args:
            ss_system: SymbolicStateSpace
            substitutions: Substituições de símbolos
            
        Returns:
            SymbolicTransferFunction: Função de transferência resultante
        """
        if not isinstance(ss_system, SymbolicStateSpace):
            raise TypeError("ss_system deve ser SymbolicStateSpace")
        
        self.history.add_step(
            "CONVERSÃO_SS_TF",
            "Convertendo SS para TF via polinômio característico",
            f"Sistema {ss_system.A.shape}",
            "Calculando TF..."
        )
        
        try:
            s = ss_system.variable
            A = ss_system.A
            B = ss_system.B
            C = ss_system.C
            D = ss_system.D
            
            if substitutions:
                A = A.subs(substitutions)
                B = B.subs(substitutions)
                C = C.subs(substitutions)
                D = D.subs(substitutions)
            
            # Calcular (sI - A)
            I = sp.eye(A.rows)
            sI_minus_A = s * I - A
            
            # Calcular determinante det(sI - A)
            denominator = sI_minus_A.det()
            
            # Calcular C * adj(sI - A) * B + D * det(sI - A)
            adj_sI_minus_A = sI_minus_A.adjugate()
            
            # Para sistemas SISO
            if C.rows == 1 and B.cols == 1:
                numerator_matrix = C * adj_sI_minus_A * B
                numerator = numerator_matrix[0, 0] + D[0, 0] * denominator
            else:
                # Sistemas MIMO não implementados ainda
                raise ValueError("Conversão SS→TF implementada apenas para sistemas SISO")
            
            # Simplificar
            numerator = sp.simplify(numerator)
            denominator = sp.simplify(denominator)
            
            # Criar função de transferência
            tf_system = SymbolicTransferFunction(numerator, denominator, s)
            
            result = {
                'original_ss': ss_system,
                'transfer_function': tf_system,
                'method': 'characteristic_polynomial'
            }
            
            self.history.add_step(
                "CONVERSÃO_CONCLUÍDA",
                "SS convertido para TF com sucesso",
                f"TF: {tf_system}",
                result
            )
            
            return tf_system
            
        except Exception as e:
            error_msg = f"Erro na conversão SS→TF: {e}"
            self.history.add_step("ERRO_CONVERSÃO", error_msg, "", str(e))
            raise ValueError(error_msg) from e
    
    def check_controllability(self, ss_system, substitutions=None):
        """
        Verifica controlabilidade do sistema SS
        
        Args:
            ss_system: SymbolicStateSpace
            substitutions: Substituições de símbolos
            
        Returns:
            dict: Análise de controlabilidade
        """
        if not isinstance(ss_system, SymbolicStateSpace):
            raise TypeError("ss_system deve ser SymbolicStateSpace")
        
        try:
            A = ss_system.A
            B = ss_system.B
            
            if substitutions:
                A = A.subs(substitutions)
                B = B.subs(substitutions)
            
            n = A.rows
            
            # Construir matriz de controlabilidade [B AB A²B ... A^(n-1)B]
            controllability_matrix = B
            A_power = sp.eye(n)
            
            for i in range(1, n):
                A_power = A_power * A
                controllability_matrix = controllability_matrix.row_join(A_power * B)
            
            # Calcular rank
            rank = controllability_matrix.rank()
            is_controllable = (rank == n)
            
            result = {
                'controllability_matrix': controllability_matrix,
                'rank': rank,
                'expected_rank': n,
                'is_controllable': is_controllable,
                'matrix_shape': controllability_matrix.shape
            }
            
            self.history.add_step(
                "ANÁLISE_CONTROLABILIDADE",
                f"Sistema {'controlável' if is_controllable else 'não controlável'}",
                f"Rank: {rank}/{n}",
                result
            )
            
            return result
            
        except Exception as e:
            error_msg = f"Erro na análise de controlabilidade: {e}"
            self.history.add_step("ERRO_CONTROLABILIDADE", error_msg, "", str(e))
            raise ValueError(error_msg) from e
    
    def check_observability(self, ss_system, substitutions=None):
        """
        Verifica observabilidade do sistema SS
        
        Args:
            ss_system: SymbolicStateSpace
            substitutions: Substituições de símbolos
            
        Returns:
            dict: Análise de observabilidade
        """
        if not isinstance(ss_system, SymbolicStateSpace):
            raise TypeError("ss_system deve ser SymbolicStateSpace")
        
        try:
            A = ss_system.A
            C = ss_system.C
            
            if substitutions:
                A = A.subs(substitutions)
                C = C.subs(substitutions)
            
            n = A.rows
            
            # Construir matriz de observabilidade [C; CA; CA²; ...; CA^(n-1)]
            observability_matrix = C
            A_power = sp.eye(n)
            
            for i in range(1, n):
                A_power = A_power * A
                observability_matrix = observability_matrix.col_join(C * A_power)
            
            # Calcular rank
            rank = observability_matrix.rank()
            is_observable = (rank == n)
            
            result = {
                'observability_matrix': observability_matrix,
                'rank': rank,
                'expected_rank': n,
                'is_observable': is_observable,
                'matrix_shape': observability_matrix.shape
            }
            
            self.history.add_step(
                "ANÁLISE_OBSERVABILIDADE",
                f"Sistema {'observável' if is_observable else 'não observável'}",
                f"Rank: {rank}/{n}",
                result
            )
            
            return result
            
        except Exception as e:
            error_msg = f"Erro na análise de observabilidade: {e}"
            self.history.add_step("ERRO_OBSERVABILIDADE", error_msg, "", str(e))
            raise ValueError(error_msg) from e
    
    def get_minimal_realization(self, ss_system, substitutions=None):
        """
        Obtém realização mínima do sistema SS
        
        Args:
            ss_system: SymbolicStateSpace
            substitutions: Substituições de símbolos
            
        Returns:
            dict: Análise da realização mínima
        """
        try:
            # Verificar controlabilidade e observabilidade
            controllability = self.check_controllability(ss_system, substitutions)
            observability = self.check_observability(ss_system, substitutions)
            
            is_minimal = controllability['is_controllable'] and observability['is_observable']
            
            result = {
                'original_system': ss_system,
                'controllability_analysis': controllability,
                'observability_analysis': observability,
                'is_minimal_realization': is_minimal,
                'system_order': ss_system.A.rows,
                'controllable_subspace_dimension': controllability['rank'],
                'observable_subspace_dimension': observability['rank']
            }
            
            if not is_minimal:
                recommendations = []
                if not controllability['is_controllable']:
                    recommendations.append("Sistema não é completamente controlável")
                if not observability['is_observable']:
                    recommendations.append("Sistema não é completamente observável")
                
                result['recommendations'] = recommendations
            
            self.history.add_step(
                "REALIZAÇÃO_MÍNIMA",
                f"Sistema {'é' if is_minimal else 'não é'} realização mínima",
                f"Ordem: {ss_system.A.rows}, Controlável: {controllability['is_controllable']}, Observável: {observability['is_observable']}",
                result
            )
            
            return result
            
        except Exception as e:
            error_msg = f"Erro na análise de realização mínima: {e}"
            self.history.add_step("ERRO_REALIZAÇÃO", error_msg, "", str(e))
            raise ValueError(error_msg) from e
    
    def get_conversion_summary(self):
        """Retorna resumo das operações de conversão realizadas"""
        return {
            'operations_history': self.history.get_summary(),
            'available_methods': [
                'tf_to_ss_canonical_controllable',
                'tf_to_ss_canonical_observable', 
                'ss_to_tf_via_characteristic',
                'check_controllability',
                'check_observability',
                'get_minimal_realization'
            ],
            'supported_forms': [
                'Canonical Controllable',
                'Canonical Observable',
                'Characteristic Polynomial Method'
            ]
        }
