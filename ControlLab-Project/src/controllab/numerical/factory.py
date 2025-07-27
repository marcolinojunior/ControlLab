#!/usr/bin/env python3
"""
Factory de Sistemas Numéricos - ControlLab Numerical
Factory pattern para criação e cache de sistemas numéricos
"""

import sympy as sp
import warnings
from typing import Dict, Any, Union, Optional
from functools import lru_cache
import hashlib

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
from .interface import NumericalInterface
from .validation import NumericalValidator


class NumericalSystemFactory:
    """
    Factory para criação eficiente de sistemas numéricos com cache
    """
    
    def __init__(self, cache_size: int = 128):
        self.interface = NumericalInterface()
        self.validator = NumericalValidator()
        self.history = OperationHistory()
        self._cache = {}
        self._cache_size = cache_size
    
    def create_from_symbolic(self, symbolic_tf: SymbolicTransferFunction):
        num_symbols = symbolic_tf.numerator.free_symbols
        den_symbols = symbolic_tf.denominator.free_symbols
        free_symbols = num_symbols.union(den_symbols)
        free_symbols.discard(symbolic_tf.variable) # Remove a variável principal, 's'

        if free_symbols:
            history_report = symbolic_tf.history.get_formatted_report()
            error_message = (
                f"FALHA NA CONVERSÃO SIMBÓLICO->NUMÉRICO: A expressão ainda contém parâmetros simbólicos.\n\n"
                f"--> SÍMBOLOS NÃO RESOLVIDOS ENCONTRADOS: {free_symbols}\n\n"
                f"--> HISTÓRICO DO OBJETO:\n{history_report}\n\n"
                f"--> AÇÃO RECOMENDADA:\n"
                f"    Use o método `.subs({{simbolo: valor_numerico}})` na sua função de transferência para substituir todos os parâmetros por valores numéricos antes de tentar a conversão."
            )
            raise ValueError(error_message)

        # ... (continua com a extração de coeficientes e conversão)
        return self.create_tf_system(symbolic_tf)

    def create_tf_system(self, symbolic_tf: SymbolicTransferFunction,
                        substitutions: Optional[Dict] = None) -> Any:
        """
        Cria sistema de função de transferência numérico com cache
        
        Args:
            symbolic_tf: Função de transferência simbólica
            substitutions: Substituições numéricas
            
        Returns:
            Sistema numérico (control.TransferFunction)
        """
        if not CONTROL_AVAILABLE:
            raise ImportError("python-control não disponível")
        
        # Gerar chave do cache
        cache_key = self._generate_cache_key('tf', symbolic_tf, substitutions)
        
        # Verificar cache
        if cache_key in self._cache:
            self.history.add_step(
                "SISTEMA_TF_CACHE",
                "Sistema recuperado do cache",
                str(symbolic_tf),
                f"Cache hit: {cache_key[:16]}..."
            )
            return self._cache[cache_key]
        
        self.history.add_step(
            "CRIAÇÃO_SISTEMA_TF",
            "Criando sistema de função de transferência numérico",
            str(symbolic_tf),
            f"Substituições: {substitutions}"
        )
        
        try:
            # Validar substituições se fornecidas
            if substitutions:
                validation = self.validator.validate_numerical_substitutions(
                    symbolic_tf.numerator + symbolic_tf.denominator, 
                    substitutions
                )
                if not validation['valid']:
                    warnings.warn(f"Substituições problemáticas: {validation}")
            
            # Criar sistema numérico
            tf_system = self.interface.symbolic_to_control_tf(symbolic_tf, substitutions)
            
            # Armazenar no cache
            self._add_to_cache(cache_key, tf_system)
            
            self.history.add_step(
                "SISTEMA_TF_CRIADO",
                "Sistema de função de transferência criado com sucesso",
                f"Ordem: {tf_system.nstates if hasattr(tf_system, 'nstates') else 'N/A'}",
                f"Cache: {cache_key[:16]}..."
            )
            
            return tf_system
            
        except Exception as e:
            self.history.add_step(
                "ERRO_CRIAÇÃO_TF",
                f"Erro na criação: {str(e)}",
                str(symbolic_tf),
                None
            )
            raise
    
    def create_ss_system(self, symbolic_ss: SymbolicStateSpace,
                        substitutions: Optional[Dict] = None) -> Any:
        """
        Cria sistema em espaço de estados numérico com cache
        
        Args:
            symbolic_ss: Sistema simbólico em espaço de estados
            substitutions: Substituições numéricas
            
        Returns:
            Sistema numérico (control.StateSpace)
        """
        if not CONTROL_AVAILABLE:
            raise ImportError("python-control não disponível")
        
        # Gerar chave do cache
        cache_key = self._generate_cache_key('ss', symbolic_ss, substitutions)
        
        # Verificar cache
        if cache_key in self._cache:
            self.history.add_step(
                "SISTEMA_SS_CACHE",
                "Sistema recuperado do cache",
                str(symbolic_ss),
                f"Cache hit: {cache_key[:16]}..."
            )
            return self._cache[cache_key]
        
        self.history.add_step(
            "CRIAÇÃO_SISTEMA_SS",
            "Criando sistema em espaço de estados numérico",
            str(symbolic_ss),
            f"Substituições: {substitutions}"
        )
        
        try:
            # Validar substituições para todas as matrizes
            if substitutions:
                all_elements = []
                for matrix in [symbolic_ss.A, symbolic_ss.B, symbolic_ss.C, symbolic_ss.D]:
                    all_elements.extend([matrix[i,j] for i in range(matrix.rows) for j in range(matrix.cols)])
                
                combined_expr = sum(all_elements, sp.S.Zero)
                validation = self.validator.validate_numerical_substitutions(combined_expr, substitutions)
                
                if not validation['valid']:
                    warnings.warn(f"Substituições problemáticas: {validation}")
            
            # Criar sistema numérico
            ss_system = self.interface.symbolic_to_control_ss(symbolic_ss, substitutions)
            
            # Armazenar no cache
            self._add_to_cache(cache_key, ss_system)
            
            self.history.add_step(
                "SISTEMA_SS_CRIADO",
                "Sistema em espaço de estados criado com sucesso",
                f"Estados: {symbolic_ss.n_states}, Entradas: {symbolic_ss.n_inputs}, Saídas: {symbolic_ss.n_outputs}",
                f"Cache: {cache_key[:16]}..."
            )
            
            return ss_system
            
        except Exception as e:
            self.history.add_step(
                "ERRO_CRIAÇÃO_SS",
                f"Erro na criação: {str(e)}",
                str(symbolic_ss),
                None
            )
            raise
    
    def create_closed_loop_system(self, plant: Union[SymbolicTransferFunction, SymbolicStateSpace],
                                 controller: Union[SymbolicTransferFunction, SymbolicStateSpace],
                                 substitutions: Optional[Dict] = None,
                                 feedback: float = -1) -> Any:
        """
        Cria sistema em malha fechada
        
        Args:
            plant: Planta (simbólica)
            controller: Controlador (simbólico)
            substitutions: Substituições numéricas
            feedback: Tipo de realimentação (+1 positiva, -1 negativa)
            
        Returns:
            Sistema em malha fechada numérico
        """
        if not CONTROL_AVAILABLE:
            raise ImportError("python-control não disponível")
        
        self.history.add_step(
            "MALHA_FECHADA",
            "Criando sistema em malha fechada",
            f"Planta: {plant}, Controlador: {controller}",
            f"Realimentação: {feedback}"
        )
        
        try:
            # Converter sistemas para numérico
            if isinstance(plant, SymbolicTransferFunction):
                plant_num = self.create_tf_system(plant, substitutions)
            else:
                plant_num = self.create_ss_system(plant, substitutions)
            
            if isinstance(controller, SymbolicTransferFunction):
                controller_num = self.create_tf_system(controller, substitutions)
            else:
                controller_num = self.create_ss_system(controller, substitutions)
            
            # Criar malha fechada
            forward_path = control.series(controller_num, plant_num)
            closed_loop = control.feedback(forward_path, 1, sign=feedback)
            
            self.history.add_step(
                "MALHA_FECHADA_CRIADA",
                "Sistema em malha fechada criado",
                f"Ordem: {closed_loop.nstates if hasattr(closed_loop, 'nstates') else 'N/A'}",
                str(closed_loop)
            )
            
            return closed_loop
            
        except Exception as e:
            self.history.add_step(
                "ERRO_MALHA_FECHADA",
                f"Erro na criação: {str(e)}",
                f"Planta: {plant}, Controlador: {controller}",
                None
            )
            raise
    
    def batch_create_systems(self, systems: Dict[str, tuple],
                           substitutions: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Cria múltiplos sistemas em lote
        
        Args:
            systems: Dict com nome -> (tipo, sistema_simbólico)
            substitutions: Substituições comuns
            
        Returns:
            Dict com sistemas numéricos criados
        """
        self.history.add_step(
            "CRIAÇÃO_LOTE",
            f"Criando {len(systems)} sistemas em lote",
            f"Sistemas: {list(systems.keys())}",
            f"Substituições: {substitutions}"
        )
        
        results = {}
        errors = {}
        
        for name, (system_type, symbolic_system) in systems.items():
            try:
                if system_type == 'tf':
                    results[name] = self.create_tf_system(symbolic_system, substitutions)
                elif system_type == 'ss':
                    results[name] = self.create_ss_system(symbolic_system, substitutions)
                else:
                    raise ValueError(f"Tipo de sistema desconhecido: {system_type}")
            except Exception as e:
                errors[name] = str(e)
        
        self.history.add_step(
            "LOTE_PROCESSADO",
            f"Lote processado: {len(results)} sucessos, {len(errors)} erros",
            f"Sucessos: {list(results.keys())}",
            f"Erros: {list(errors.keys())}"
        )
        
        if errors:
            warnings.warn(f"Erros na criação em lote: {errors}")
        
        return results
    
    def _generate_cache_key(self, system_type: str, 
                           symbolic_system: Union[SymbolicTransferFunction, SymbolicStateSpace],
                           substitutions: Optional[Dict] = None) -> str:
        """Gera chave única para cache"""
        # Criar string única baseada no sistema e substituições
        system_str = str(symbolic_system)
        subs_str = str(sorted(substitutions.items())) if substitutions else "None"
        
        # Gerar hash MD5
        content = f"{system_type}_{system_str}_{subs_str}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _add_to_cache(self, key: str, system: Any):
        """Adiciona sistema ao cache com limite de tamanho"""
        if len(self._cache) >= self._cache_size:
            # Remove item mais antigo (FIFO simples)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        
        self._cache[key] = system
    
    def clear_cache(self):
        """Limpa o cache de sistemas"""
        cache_size = len(self._cache)
        self._cache.clear()
        
        self.history.add_step(
            "CACHE_LIMPO",
            f"Cache de sistemas limpo",
            f"Eram {cache_size} sistemas em cache",
            "Cache vazio"
        )
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do cache"""
        return {
            'cached_systems': len(self._cache),
            'cache_size_limit': self._cache_size,
            'cache_usage': len(self._cache) / self._cache_size,
            'cache_keys': list(self._cache.keys())
        }


# Adaptadores para diferentes representações
class ControlTransferFunctionAdapter:
    """Adaptador para funções de transferência do python-control"""
    
    def __init__(self, ctrl_tf):
        self.ctrl_tf = ctrl_tf
    
    def get_poles(self):
        """Retorna polos do sistema"""
        if hasattr(self.ctrl_tf, 'pole'):
            return self.ctrl_tf.pole()
        else:
            return []
    
    def get_zeros(self):
        """Retorna zeros do sistema"""
        if hasattr(self.ctrl_tf, 'zero'):
            return self.ctrl_tf.zero()
        else:
            return []
    
    def evaluate(self, s_values):
        """Avalia a função de transferência em pontos específicos"""
        if hasattr(self.ctrl_tf, 'evalfr'):
            return [self.ctrl_tf.evalfr(s) for s in s_values]
        else:
            return []


class ControlStateSpaceAdapter:
    """Adaptador para sistemas em espaço de estados do python-control"""
    
    def __init__(self, ctrl_ss):
        self.ctrl_ss = ctrl_ss
    
    def get_eigenvalues(self):
        """Retorna autovalores (polos) do sistema"""
        if hasattr(self.ctrl_ss, 'pole'):
            return self.ctrl_ss.pole()
        else:
            return []
    
    def get_matrices(self):
        """Retorna matrizes A, B, C, D"""
        if hasattr(self.ctrl_ss, 'A'):
            return self.ctrl_ss.A, self.ctrl_ss.B, self.ctrl_ss.C, self.ctrl_ss.D
        else:
            return None, None, None, None
