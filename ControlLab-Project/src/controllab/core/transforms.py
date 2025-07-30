"""
Módulo de transformadas para sistemas de controle
"""

import sympy as sp
from typing import Dict, Union, List, Optional
from .symbolic_tf import SymbolicTransferFunction
from .history import OperationHistory


class LaplaceTransform:
    """Classe para transformadas de Laplace"""
    
    def __init__(self):
        self.history = OperationHistory()
    
    def transform(self, time_function: sp.Expr, 
                 time_var: sp.Symbol = None,
                 s_var: sp.Symbol = None) -> sp.Expr:
        """
        Aplica transformada de Laplace
        
        Args:
            time_function: Função no domínio do tempo
            time_var: Variável temporal (default: t)
            s_var: Variável de Laplace (default: s)
            
        Returns:
            sp.Expr: Transformada de Laplace
        """
        if time_var is None:
            time_var = sp.Symbol('t', real=True, positive=True)
        if s_var is None:
            s_var = sp.Symbol('s', complex=True)
        
        self.history.add_step(
            "TRANSFORMADA_LAPLACE",
            f"Aplicando L{{{time_function}}}",
            f"f(t) = {time_function}",
            "Calculando transformada..."
        )
        
        try:
            # Aplica transformada de Laplace
            result = sp.laplace_transform(time_function, time_var, s_var)
            
            # result é uma tupla (F(s), a, cond) onde:
            # F(s) é a transformada, a é o limite inferior, cond são as condições
            transformed = result[0]
            
            self.history.add_step(
                "RESULTADO_LAPLACE",
                f"L{{{time_function}}} = {transformed}",
                f"Domínio do tempo: {time_function}",
                f"Domínio de Laplace: {transformed}"
            )
            
            return transformed
            
        except Exception as e:
            # Se falhar, tenta com regras básicas
            transformed = self._apply_basic_rules(time_function, time_var, s_var)
            
            self.history.add_step(
                "LAPLACE_REGRAS_BÁSICAS",
                f"Usando regras básicas para {time_function}",
                str(e),
                f"Resultado: {transformed}"
            )
            
            return transformed
    
    def inverse_transform(self, s_function: sp.Expr,
                         s_var: sp.Symbol = None,
                         time_var: sp.Symbol = None) -> sp.Expr:
        """
        Aplica transformada inversa de Laplace
        
        Args:
            s_function: Função no domínio de Laplace
            s_var: Variável de Laplace (default: s)
            time_var: Variável temporal (default: t)
            
        Returns:
            sp.Expr: Função no domínio do tempo
        """
        if s_var is None:
            s_var = sp.Symbol('s', complex=True)
        if time_var is None:
            time_var = sp.Symbol('t', real=True, positive=True)
        
        self.history.add_step(
            "TRANSFORMADA_INVERSA_LAPLACE",
            f"Aplicando L⁻¹{{{s_function}}}",
            f"F(s) = {s_function}",
            "Calculando transformada inversa..."
        )
        
        try:
            # Aplica transformada inversa
            result = sp.inverse_laplace_transform(s_function, s_var, time_var)
            
            self.history.add_step(
                "RESULTADO_INVERSA_LAPLACE",
                f"L⁻¹{{{s_function}}} = {result}",
                f"Domínio de Laplace: {s_function}",
                f"Domínio do tempo: {result}"
            )
            
            return result
            
        except Exception as e:
            # Se falhar, tenta frações parciais
            partial_result = self._partial_fraction_inverse(s_function, s_var, time_var)
            
            self.history.add_step(
                "INVERSA_FRAÇÕES_PARCIAIS",
                f"Usando frações parciais para {s_function}",
                str(e),
                f"Resultado: {partial_result}"
            )
            
            return partial_result
    
    def _apply_basic_rules(self, time_function: sp.Expr, 
                          time_var: sp.Symbol, s_var: sp.Symbol) -> sp.Expr:
        """Aplica regras básicas da transformada de Laplace"""
        
        # Regras básicas conhecidas
        if time_function == 1:
            return 1/s_var
        elif time_function == time_var:
            return 1/s_var**2
        elif time_function == time_var**2:
            return 2/s_var**3
        elif time_function.has(sp.exp):
            # e^(at) -> 1/(s-a)
            args = time_function.args
            if len(args) == 1 and args[0].has(time_var):
                a = args[0].coeff(time_var, 1)
                return 1/(s_var - a)
        elif time_function.has(sp.sin):
            # sin(ωt) -> ω/(s²+ω²)
            args = time_function.args
            if len(args) == 1:
                omega = args[0].coeff(time_var, 1)
                return omega/(s_var**2 + omega**2)
        elif time_function.has(sp.cos):
            # cos(ωt) -> s/(s²+ω²)
            args = time_function.args
            if len(args) == 1:
                omega = args[0].coeff(time_var, 1)
                return s_var/(s_var**2 + omega**2)
        
        # Se não reconhecer, retorna forma simbólica
        return sp.Function('L')(time_function)
    
    def _partial_fraction_inverse(self, s_function: sp.Expr,
                                 s_var: sp.Symbol, time_var: sp.Symbol) -> sp.Expr:
        """Calcula inversa usando frações parciais"""
        try:
            # Decompõe em frações parciais
            partial_fractions = sp.apart(s_function, s_var)
            
            # Aplica inversa a cada termo
            if hasattr(partial_fractions, 'args') and len(partial_fractions.args) > 1:
                result = 0
                for term in partial_fractions.args:
                    result += self._inverse_simple_term(term, s_var, time_var)
                return result
            else:
                return self._inverse_simple_term(partial_fractions, s_var, time_var)
                
        except:
            return sp.Function('L_inv')(s_function)
    
    def _inverse_simple_term(self, term: sp.Expr, 
                           s_var: sp.Symbol, time_var: sp.Symbol) -> sp.Expr:
        """Calcula inversa de termos simples"""
        try:
            # 1/s -> 1
            if term == 1/s_var:
                return 1
            # 1/s² -> t
            elif term == 1/s_var**2:
                return time_var
            # 1/(s-a) -> e^(at)
            elif term.has(1/(s_var - sp.Wild('a'))):
                a = -(term.as_numer_denom()[1] - s_var)
                return sp.exp(a * time_var)
            # s/(s²+ω²) -> cos(ωt)
            elif term.as_numer_denom()[0] == s_var and term.as_numer_denom()[1].has(s_var**2):
                omega_sq = term.as_numer_denom()[1] - s_var**2
                omega = sp.sqrt(omega_sq)
                return sp.cos(omega * time_var)
            # ω/(s²+ω²) -> sin(ωt)
            elif not term.as_numer_denom()[0].has(s_var) and term.as_numer_denom()[1].has(s_var**2):
                omega = term.as_numer_denom()[0]
                return sp.sin(omega * time_var)
            else:
                return sp.Function('L_inv')(term)
        except:
            return sp.Function('L_inv')(term)


class ZTransform:
    """Classe para transformadas Z"""
    
    def __init__(self):
        self.history = OperationHistory()
    
    def transform(self, discrete_sequence: sp.Expr,
                 n_var: sp.Symbol = None,
                 z_var: sp.Symbol = None) -> sp.Expr:
        """
        Aplica transformada Z
        
        Args:
            discrete_sequence: Sequência discreta x[n]
            n_var: Variável discreta (default: n)
            z_var: Variável Z (default: z)
            
        Returns:
            sp.Expr: Transformada Z
        """
        if n_var is None:
            n_var = sp.Symbol('n', integer=True)
        if z_var is None:
            z_var = sp.Symbol('z', complex=True)
        
        self.history.add_step(
            "TRANSFORMADA_Z",
            f"Aplicando Z{{{discrete_sequence}}}",
            f"x[n] = {discrete_sequence}",
            "Calculando transformada Z..."
        )
        
        try:
            # Para sequências simples, aplica regras conhecidas
            transformed = self._apply_z_rules(discrete_sequence, n_var, z_var)
            
            self.history.add_step(
                "RESULTADO_Z",
                f"Z{{{discrete_sequence}}} = {transformed}",
                f"Sequência: {discrete_sequence}",
                f"Transformada Z: {transformed}"
            )
            
            return transformed
            
        except Exception as e:
            self.history.add_step(
                "ERRO_Z",
                f"Erro na transformada Z: {str(e)}",
                f"x[n] = {discrete_sequence}",
                "Retornando forma simbólica"
            )
            return sp.Function('Z')(discrete_sequence)
    
    def inverse_transform(self, z_function: sp.Expr,
                         z_var: sp.Symbol = None,
                         n_var: sp.Symbol = None) -> sp.Expr:
        """
        Aplica transformada inversa Z
        
        Args:
            z_function: Função no domínio Z
            z_var: Variável Z (default: z)
            n_var: Variável discreta (default: n)
            
        Returns:
            sp.Expr: Sequência no domínio discreto
        """
        if z_var is None:
            z_var = sp.Symbol('z', complex=True)
        if n_var is None:
            n_var = sp.Symbol('n', integer=True)
        
        self.history.add_step(
            "TRANSFORMADA_INVERSA_Z",
            f"Aplicando Z⁻¹{{{z_function}}}",
            f"X(z) = {z_function}",
            "Calculando transformada inversa Z..."
        )
        
        try:
            # Aplica frações parciais e regras inversas
            result = self._z_inverse_partial_fractions(z_function, z_var, n_var)
            
            self.history.add_step(
                "RESULTADO_INVERSA_Z",
                f"Z⁻¹{{{z_function}}} = {result}",
                f"Domínio Z: {z_function}",
                f"Sequência: {result}"
            )
            
            return result
            
        except Exception as e:
            self.history.add_step(
                "ERRO_INVERSA_Z",
                f"Erro na transformada inversa Z: {str(e)}",
                f"X(z) = {z_function}",
                "Retornando forma simbólica"
            )
            return sp.Function('Z_inv')(z_function)
    
    def _apply_z_rules(self, sequence: sp.Expr, n_var: sp.Symbol, z_var: sp.Symbol) -> sp.Expr:
        """Aplica regras básicas da transformada Z"""
        
        # δ[n] -> 1
        if sequence == sp.DiracDelta(n_var):
            return 1
        
        # u[n] (degrau unitário) -> z/(z-1)
        if sequence == sp.Heaviside(n_var):
            return z_var / (z_var - 1)
        
        # a^n * u[n] -> z/(z-a)
        if sequence.has(sp.Heaviside(n_var)):
            # Extrai a base da exponencial
            base_term = sequence / sp.Heaviside(n_var)
            if base_term.has(n_var) and base_term.is_Pow:
                a = base_term.base
                return z_var / (z_var - a)
        
        # n * u[n] -> z / (z-1)²
        if sequence == n_var * sp.Heaviside(n_var):
            return z_var / (z_var - 1)**2
        
        # Forma geral para exponenciais
        if sequence.has(sp.Pow) and sequence.has(n_var):
            base = sequence.base
            return z_var / (z_var - base)
        
        # Se não reconhecer, retorna forma simbólica
        return sp.Function('Z')(sequence)
    
    def _z_inverse_partial_fractions(self, z_function: sp.Expr,
                                   z_var: sp.Symbol, n_var: sp.Symbol) -> sp.Expr:
        """Calcula inversa Z usando frações parciais"""
        try:
            # Primeiro, converte X(z) para X(z)/z para facilitar frações parciais
            x_over_z = z_function / z_var
            
            # Decompõe em frações parciais
            partial_fractions = sp.apart(x_over_z, z_var)
            
            # Multiplica novamente por z
            partial_fractions = partial_fractions * z_var
            
            # Aplica inversa a cada termo
            if hasattr(partial_fractions, 'args') and len(partial_fractions.args) > 1:
                result = 0
                for term in partial_fractions.args:
                    result += self._z_inverse_simple_term(term, z_var, n_var)
                return result
            else:
                return self._z_inverse_simple_term(partial_fractions, z_var, n_var)
                
        except:
            return sp.Function('Z_inv')(z_function)
    
    def _z_inverse_simple_term(self, term: sp.Expr,
                             z_var: sp.Symbol, n_var: sp.Symbol) -> sp.Expr:
        """Calcula inversa Z de termos simples"""
        try:
            # z/(z-1) -> u[n]
            if term == z_var / (z_var - 1):
                return sp.Heaviside(n_var)
            
            # z/(z-a) -> a^n * u[n]
            elif term.as_numer_denom()[0] == z_var:
                denominator = term.as_numer_denom()[1]
                if denominator.has(z_var - sp.Wild('a')):
                    a = -(denominator - z_var)
                    return (a**n_var) * sp.Heaviside(n_var)
            
            # 1 -> δ[n]
            elif term == 1:
                return sp.DiracDelta(n_var)
            
            else:
                return sp.Function('Z_inv')(term)
                
        except:
            return sp.Function('Z_inv')(term)


class FourierTransform:
    """Classe para transformadas de Fourier"""
    
    def __init__(self):
        self.history = OperationHistory()
    
    def transform(self, time_function: sp.Expr,
                 time_var: sp.Symbol = None,
                 freq_var: sp.Symbol = None) -> sp.Expr:
        """
        Aplica transformada de Fourier
        
        Args:
            time_function: Função no domínio do tempo
            time_var: Variável temporal (default: t)
            freq_var: Variável de frequência (default: omega)
            
        Returns:
            sp.Expr: Transformada de Fourier
        """
        if time_var is None:
            time_var = sp.Symbol('t', real=True)
        if freq_var is None:
            freq_var = sp.Symbol('omega', real=True)
        
        self.history.add_step(
            "TRANSFORMADA_FOURIER",
            f"Aplicando F{{{time_function}}}",
            f"f(t) = {time_function}",
            "Calculando transformada de Fourier..."
        )
        
        try:
            # Aplica transformada de Fourier
            result = sp.fourier_transform(time_function, time_var, freq_var)
            
            self.history.add_step(
                "RESULTADO_FOURIER",
                f"F{{{time_function}}} = {result}",
                f"Domínio do tempo: {time_function}",
                f"Domínio da frequência: {result}"
            )
            
            return result
            
        except Exception as e:
            self.history.add_step(
                "ERRO_FOURIER",
                f"Erro na transformada de Fourier: {str(e)}",
                f"f(t) = {time_function}",
                "Retornando forma simbólica"
            )
            return sp.Function('F')(time_function)
    
    def inverse_transform(self, freq_function: sp.Expr,
                         freq_var: sp.Symbol = None,
                         time_var: sp.Symbol = None) -> sp.Expr:
        """
        Aplica transformada inversa de Fourier
        
        Args:
            freq_function: Função no domínio da frequência
            freq_var: Variável de frequência (default: omega)
            time_var: Variável temporal (default: t)
            
        Returns:
            sp.Expr: Função no domínio do tempo
        """
        if freq_var is None:
            freq_var = sp.Symbol('omega', real=True)
        if time_var is None:
            time_var = sp.Symbol('t', real=True)
        
        self.history.add_step(
            "TRANSFORMADA_INVERSA_FOURIER",
            f"Aplicando F⁻¹{{{freq_function}}}",
            f"F(ω) = {freq_function}",
            "Calculando transformada inversa de Fourier..."
        )
        
        try:
            # Aplica transformada inversa de Fourier
            result = sp.inverse_fourier_transform(freq_function, freq_var, time_var)
            
            self.history.add_step(
                "RESULTADO_INVERSA_FOURIER",
                f"F⁻¹{{{freq_function}}} = {result}",
                f"Domínio da frequência: {freq_function}",
                f"Domínio do tempo: {result}"
            )
            
            return result
            
        except Exception as e:
            self.history.add_step(
                "ERRO_INVERSA_FOURIER",
                f"Erro na transformada inversa de Fourier: {str(e)}",
                f"F(ω) = {freq_function}",
                "Retornando forma simbólica"
            )
            return sp.Function('F_inv')(freq_function)
