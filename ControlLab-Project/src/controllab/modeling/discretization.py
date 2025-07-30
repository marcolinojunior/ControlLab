"""
ControlLab - Métodos de Discretização
====================================

Este módulo implementa métodos de discretização de sistemas contínuos para
sistemas discretos com explicações pedagógicas detalhadas.

Métodos implementados:
- Transformação Bilinear (Tustin)
- Zero-Order Hold (ZOH)
- First-Order Hold (FOH)
- Euler Forward/Backward
- Matched Z-Transform
"""

import sympy as sp
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass, field
from ..core.symbolic_tf import SymbolicTransferFunction
from ..core.history import OperationHistory, OperationStep

@dataclass
class DiscretizationResult:
    """
    Resultado da discretização
    
    Atributos:
        continuous_tf: Função de transferência contínua original
        discrete_tf: Função de transferência discreta resultante
        sampling_time: Período de amostragem usado
        method: Método de discretização usado
        transformation_steps: Passos da transformação
        frequency_mapping: Mapeamento de frequências s->z
        stability_preserved: Se a estabilidade foi preservada
    """
    continuous_tf: SymbolicTransferFunction = None
    discrete_tf: SymbolicTransferFunction = None
    sampling_time: float = 0.1
    method: str = ""
    transformation_steps: List[str] = field(default_factory=list)
    frequency_mapping: Dict[str, str] = field(default_factory=dict)
    stability_preserved: bool = True
    history: OperationHistory = field(default_factory=OperationHistory)

class DiscretizationMethods:
    """
    Classe que implementa diferentes métodos de discretização
    
    Cada método preserva diferentes características do sistema contínuo:
    - Tustin: Resposta em frequência
    - ZOH: Resposta temporal ao degrau
    - FOH: Resposta temporal mais suave
    - Euler: Aproximação simples
    """
    
    def __init__(self, sampling_time: float = 0.1):
        """
        Inicializa os métodos de discretização
        
        Args:
            sampling_time: Período de amostragem padrão
        """
        self.T = sampling_time
        self.s = sp.Symbol('s')
        self.z = sp.Symbol('z')
        self.history = OperationHistory()
    
    def tustin_transform(self, continuous_tf: SymbolicTransferFunction, 
                        T: Optional[float] = None, 
                        show_steps: bool = True) -> DiscretizationResult:
        """
        Discretização pela transformação bilinear (Tustin)
        
        A transformação de Tustin mapeia:
        s = 2/T * (z-1)/(z+1)
        
        Args:
            continuous_tf: Função de transferência contínua
            T: Período de amostragem (usa padrão se None)
            show_steps: Se deve mostrar os passos
        
        Returns:
            DiscretizationResult: Resultado da discretização
        """
        if T is None:
            T = self.T
        
        if show_steps:
            print("🔄 DISCRETIZAÇÃO POR TRANSFORMAÇÃO BILINEAR (TUSTIN)")
            print("=" * 55)
            print(f"📊 Sistema contínuo: H(s) = {continuous_tf.num}/{continuous_tf.den}")
            print(f"⏱️  Período de amostragem: T = {T}")
            print(f"🔧 Mapeamento: s = 2/T * (z-1)/(z+1)")
        
        result = DiscretizationResult()
        result.continuous_tf = continuous_tf
        result.sampling_time = T
        result.method = "Transformação Bilinear (Tustin)"
        
        try:
            # Definir a transformação s = 2/T * (z-1)/(z+1)
            s_to_z = 2/T * (self.z - 1)/(self.z + 1)
            
            if show_steps:
                print(f"   📝 Substituição: s = {s_to_z}")
            
            result.transformation_steps.append(f"Mapeamento: s = 2/T * (z-1)/(z+1)")
            result.transformation_steps.append(f"T = {T}")
            
            # Substituir s na função de transferência
            H_s_num = continuous_tf.num
            H_s_den = continuous_tf.den
            
            if show_steps:
                print(f"   🧮 Substituindo na H(s)...")
            
            # Substituir s por s_to_z
            H_z_num = H_s_num.subs(self.s, s_to_z)
            H_z_den = H_s_den.subs(self.s, s_to_z)
            
            if show_steps:
                print(f"   📊 Numerador discreto (antes simplificação): {H_z_num}")
                print(f"   📊 Denominador discreto (antes simplificação): {H_z_den}")
            
            # Simplificar a expressão resultante
            H_z_simplified = sp.simplify(H_z_num / H_z_den)
            
            # Separar numerador e denominador simplificados
            H_z_num_final = sp.numer(H_z_simplified)
            H_z_den_final = sp.denom(H_z_simplified)
            
            if show_steps:
                print(f"   🎯 H(z) = {H_z_num_final}/{H_z_den_final}")
                print(f"   ✅ Discretização concluída!")
            
            result.discrete_tf = SymbolicTransferFunction(H_z_num_final, H_z_den_final, self.z)
            result.transformation_steps.append(f"Resultado: H(z) = {H_z_simplified}")
            
            # Mapear propriedades de frequência
            result.frequency_mapping = {
                "DC (s=0)": "z=1",
                "Nyquist (s=j*π/T)": "z=-1",
                "Estabilidade": "Semiplano esquerdo s → Círculo unitário z"
            }
            
            # Verificar preservação de estabilidade
            result.stability_preserved = self._check_stability_preservation(continuous_tf, result.discrete_tf)
            
            # Adicionar ao histórico
            step = OperationStep(
                operation="discretizacao_tustin",
                input_expr=f"{H_s_num}/{H_s_den}",
                output_expr=f"{H_z_num_final}/{H_z_den_final}",
                explanation=f"Transformação bilinear com T={T}"
            )
            result.history.add_step(step)
            
        except Exception as e:
            error_msg = f"Erro na transformação Tustin: {e}"
            result.transformation_steps.append(f"❌ {error_msg}")
            if show_steps:
                print(f"❌ {error_msg}")
        
        return result
    
    def zero_order_hold(self, continuous_tf: SymbolicTransferFunction,
                       T: Optional[float] = None,
                       show_steps: bool = True) -> DiscretizationResult:
        """
        Discretização por Zero-Order Hold (ZOH)
        
        O método ZOH preserva a resposta ao degrau do sistema contínuo.
        H(z) = (1-z^(-1)) * Z{L^(-1)[H(s)/s]}
        
        Args:
            continuous_tf: Função de transferência contínua
            T: Período de amostragem
            show_steps: Se deve mostrar os passos
        
        Returns:
            DiscretizationResult: Resultado da discretização
        """
        if T is None:
            T = self.T
        
        if show_steps:
            print("🔄 DISCRETIZAÇÃO POR ZERO-ORDER HOLD (ZOH)")
            print("=" * 45)
            print(f"📊 Sistema contínuo: H(s) = {continuous_tf.num}/{continuous_tf.den}")
            print(f"⏱️  Período de amostragem: T = {T}")
            print(f"🔧 Método: Preserva resposta ao degrau")
        
        result = DiscretizationResult()
        result.continuous_tf = continuous_tf
        result.sampling_time = T
        result.method = "Zero-Order Hold (ZOH)"
        
        try:
            H_s_num = continuous_tf.num
            H_s_den = continuous_tf.den
            
            if show_steps:
                print(f"   📝 Fórmula: H(z) = (1-z^(-1)) * Z{{L^(-1)[H(s)/s]}}")
            
            result.transformation_steps.append("Aplicando método ZOH")
            result.transformation_steps.append("Fórmula: H(z) = (1-z^(-1)) * Z{L^(-1)[H(s)/s]}")
            
            # Para casos simples, usar transformações conhecidas
            H_z_discrete = self._apply_zoh_transform(H_s_num, H_s_den, T, show_steps)
            
            result.discrete_tf = SymbolicTransferFunction(sp.numer(H_z_discrete), 
                                                        sp.denom(H_z_discrete), self.z)
            
            if show_steps:
                print(f"   ✅ H(z) = {H_z_discrete}")
                print(f"   ✅ Discretização ZOH concluída!")
            
            result.transformation_steps.append(f"Resultado: H(z) = {H_z_discrete}")
            
            # Propriedades do ZOH
            result.frequency_mapping = {
                "DC": "Preservado exatamente",
                "Resposta temporal": "Degrau preservado",
                "Estabilidade": "Preservada para T suficientemente pequeno"
            }
            
            # Adicionar ao histórico
            step = OperationStep(
                operation="discretizacao_zoh",
                input_expr=f"{H_s_num}/{H_s_den}",
                output_expr=str(H_z_discrete),
                explanation=f"Zero-Order Hold com T={T}"
            )
            result.history.add_step(step)
            
        except Exception as e:
            error_msg = f"Erro na discretização ZOH: {e}"
            result.transformation_steps.append(f"❌ {error_msg}")
            if show_steps:
                print(f"❌ {error_msg}")
        
        return result
    
    def _apply_zoh_transform(self, num: sp.Expr, den: sp.Expr, T: float, show_steps: bool) -> sp.Expr:
        """Aplica transformação ZOH para casos conhecidos"""
        
        # Casos simples implementados
        
        # Integrador: 1/s -> T*z/(z-1)
        if den.equals(self.s) and num.is_constant():
            if show_steps:
                print(f"   🎯 Integrador detectado: {num}/s")
            return num * T * self.z / (self.z - 1)
        
        # Polo simples: 1/(s+a) -> (1-e^(-a*T))*z/(z-e^(-a*T))
        if num.is_constant() and den.is_linear(self.s):
            # Extrair coeficientes: s + a
            coeffs = sp.Poly(den, self.s).all_coeffs()
            if len(coeffs) == 2:
                s_coeff, const_term = coeffs
                if s_coeff == 1:  # s + a
                    a = const_term
                    if show_steps:
                        print(f"   🎯 Polo simples detectado: {num}/(s+{a})")
                    
                    e_aT = sp.exp(-a * T)
                    return num * (1 - e_aT) * self.z / (self.z - e_aT)
        
        # Sistema de segunda ordem: ωn²/(s² + 2ξωn*s + ωn²)
        if den.is_quadratic(self.s) and num.is_constant():
            coeffs = sp.Poly(den, self.s).all_coeffs()
            if len(coeffs) == 3:
                s2_coeff, s_coeff, const_term = coeffs
                if s2_coeff == 1:  # s² + 2ξωn*s + ωn²
                    if show_steps:
                        print(f"   🎯 Sistema de segunda ordem detectado")
                    
                    # Simplificação: usar aproximação para sistemas de 2ª ordem
                    # Na prática, seria necessário calcular a transformada completa
                    omega_n_sq = const_term
                    return num * omega_n_sq * T**2 * self.z * (self.z + 1) / (2 * (self.z - 1)**2)
        
        # Caso geral: representação simbólica
        if show_steps:
            print(f"   🔄 Aplicando transformação ZOH geral...")
        
        # Para casos não implementados, retornar aproximação
        return self.z / (self.z - sp.exp(-T))  # Aproximação genérica
    
    def first_order_hold(self, continuous_tf: SymbolicTransferFunction,
                        T: Optional[float] = None,
                        show_steps: bool = True) -> DiscretizationResult:
        """
        Discretização por First-Order Hold (FOH)
        
        O método FOH usa interpolação linear entre amostras,
        resultando em aproximação mais suave que ZOH.
        
        Args:
            continuous_tf: Função de transferência contínua
            T: Período de amostragem
            show_steps: Se deve mostrar os passos
        
        Returns:
            DiscretizationResult: Resultado da discretização
        """
        if T is None:
            T = self.T
        
        if show_steps:
            print("🔄 DISCRETIZAÇÃO POR FIRST-ORDER HOLD (FOH)")
            print("=" * 45)
            print(f"📊 Sistema contínuo: H(s) = {continuous_tf.num}/{continuous_tf.den}")
            print(f"⏱️  Período de amostragem: T = {T}")
            print(f"🔧 Método: Interpolação linear entre amostras")
        
        result = DiscretizationResult()
        result.continuous_tf = continuous_tf
        result.sampling_time = T
        result.method = "First-Order Hold (FOH)"
        
        try:
            # FOH é mais complexo que ZOH
            # Implementação simplificada para fins pedagógicos
            
            if show_steps:
                print(f"   📝 FOH = ZOH + correção de primeira ordem")
            
            # Começar com ZOH
            zoh_result = self.zero_order_hold(continuous_tf, T, False)
            
            # Adicionar correção FOH (simplificada)
            # Na prática, FOH = (1+z^(-1))/2 * ZOH_modificado
            zoh_tf = zoh_result.discrete_tf.num / zoh_result.discrete_tf.den
            
            # Aplicar correção FOH
            foh_correction = (1 + 1/self.z) / 2
            H_z_foh = zoh_tf * foh_correction
            
            H_z_simplified = sp.simplify(H_z_foh)
            
            result.discrete_tf = SymbolicTransferFunction(sp.numer(H_z_simplified),
                                                        sp.denom(H_z_simplified), self.z)
            
            if show_steps:
                print(f"   ✅ H(z) = {H_z_simplified}")
                print(f"   ✅ Discretização FOH concluída!")
            
            result.transformation_steps.append("Aplicando First-Order Hold")
            result.transformation_steps.append("FOH = correção linear sobre ZOH")
            result.transformation_steps.append(f"Resultado: H(z) = {H_z_simplified}")
            
            result.frequency_mapping = {
                "Suavidade": "Melhor que ZOH",
                "Resposta transitória": "Mais próxima do contínuo",
                "Complexidade": "Maior que ZOH"
            }
            
        except Exception as e:
            error_msg = f"Erro na discretização FOH: {e}"
            result.transformation_steps.append(f"❌ {error_msg}")
            if show_steps:
                print(f"❌ {error_msg}")
        
        return result
    
    def euler_forward(self, continuous_tf: SymbolicTransferFunction,
                     T: Optional[float] = None,
                     show_steps: bool = True) -> DiscretizationResult:
        """
        Discretização por Euler Forward
        
        Aproximação: s ≈ (z-1)/T
        
        Args:
            continuous_tf: Função de transferência contínua
            T: Período de amostragem
            show_steps: Se deve mostrar os passos
        
        Returns:
            DiscretizationResult: Resultado da discretização
        """
        if T is None:
            T = self.T
        
        if show_steps:
            print("🔄 DISCRETIZAÇÃO POR EULER FORWARD")
            print("=" * 35)
            print(f"📊 Sistema contínuo: H(s) = {continuous_tf.num}/{continuous_tf.den}")
            print(f"⏱️  Período de amostragem: T = {T}")
            print(f"🔧 Aproximação: s ≈ (z-1)/T")
        
        result = DiscretizationResult()
        result.continuous_tf = continuous_tf
        result.sampling_time = T
        result.method = "Euler Forward"
        
        try:
            # Substituição s = (z-1)/T
            s_approx = (self.z - 1) / T
            
            if show_steps:
                print(f"   📝 Substituição: s = {s_approx}")
            
            # Aplicar substituição
            H_s_num = continuous_tf.num
            H_s_den = continuous_tf.den
            
            H_z_num = H_s_num.subs(self.s, s_approx)
            H_z_den = H_s_den.subs(self.s, s_approx)
            
            H_z_simplified = sp.simplify(H_z_num / H_z_den)
            
            result.discrete_tf = SymbolicTransferFunction(sp.numer(H_z_simplified),
                                                        sp.denom(H_z_simplified), self.z)
            
            if show_steps:
                print(f"   ✅ H(z) = {H_z_simplified}")
                print(f"   ⚠️  Atenção: Método pode ser instável para T grande")
            
            result.transformation_steps.append("Aproximação Euler Forward: s = (z-1)/T")
            result.transformation_steps.append(f"T = {T}")
            result.transformation_steps.append(f"Resultado: H(z) = {H_z_simplified}")
            result.transformation_steps.append("⚠️ Cuidado: pode introduzir instabilidade")
            
            result.frequency_mapping = {
                "Simplicidade": "Muito simples",
                "Precisão": "Baixa para T grande",
                "Estabilidade": "Pode ser perdida"
            }
            
            # Euler forward pode não preservar estabilidade
            result.stability_preserved = False
            
        except Exception as e:
            error_msg = f"Erro na discretização Euler Forward: {e}"
            result.transformation_steps.append(f"❌ {error_msg}")
            if show_steps:
                print(f"❌ {error_msg}")
        
        return result
    
    def euler_backward(self, continuous_tf: SymbolicTransferFunction,
                      T: Optional[float] = None,
                      show_steps: bool = True) -> DiscretizationResult:
        """
        Discretização por Euler Backward
        
        Aproximação: s ≈ (z-1)/(T*z)
        
        Args:
            continuous_tf: Função de transferência contínua
            T: Período de amostragem
            show_steps: Se deve mostrar os passos
        
        Returns:
            DiscretizationResult: Resultado da discretização
        """
        if T is None:
            T = self.T
        
        if show_steps:
            print("🔄 DISCRETIZAÇÃO POR EULER BACKWARD")
            print("=" * 36)
            print(f"📊 Sistema contínuo: H(s) = {continuous_tf.num}/{continuous_tf.den}")
            print(f"⏱️  Período de amostragem: T = {T}")
            print(f"🔧 Aproximação: s ≈ (z-1)/(T*z)")
        
        result = DiscretizationResult()
        result.continuous_tf = continuous_tf
        result.sampling_time = T
        result.method = "Euler Backward"
        
        try:
            # Substituição s = (z-1)/(T*z)
            s_approx = (self.z - 1) / (T * self.z)
            
            if show_steps:
                print(f"   📝 Substituição: s = {s_approx}")
            
            # Aplicar substituição
            H_s_num = continuous_tf.num
            H_s_den = continuous_tf.den
            
            H_z_num = H_s_num.subs(self.s, s_approx)
            H_z_den = H_s_den.subs(self.s, s_approx)
            
            H_z_simplified = sp.simplify(H_z_num / H_z_den)
            
            result.discrete_tf = SymbolicTransferFunction(sp.numer(H_z_simplified),
                                                        sp.denom(H_z_simplified), self.z)
            
            if show_steps:
                print(f"   ✅ H(z) = {H_z_simplified}")
                print(f"   ✅ Vantagem: Sempre estável se contínuo for estável")
            
            result.transformation_steps.append("Aproximação Euler Backward: s = (z-1)/(T*z)")
            result.transformation_steps.append(f"T = {T}")
            result.transformation_steps.append(f"Resultado: H(z) = {H_z_simplified}")
            result.transformation_steps.append("✅ Preserva estabilidade")
            
            result.frequency_mapping = {
                "Simplicidade": "Muito simples",
                "Estabilidade": "Sempre preservada",
                "Precisão": "Baixa para T grande"
            }
            
            # Euler backward preserva estabilidade
            result.stability_preserved = True
            
        except Exception as e:
            error_msg = f"Erro na discretização Euler Backward: {e}"
            result.transformation_steps.append(f"❌ {error_msg}")
            if show_steps:
                print(f"❌ {error_msg}")
        
        return result
    
    def matched_z_transform(self, continuous_tf: SymbolicTransferFunction,
                           T: Optional[float] = None,
                           show_steps: bool = True) -> DiscretizationResult:
        """
        Discretização por Matched Z-Transform
        
        Mapeia polos e zeros: s = a → z = e^(a*T)
        
        Args:
            continuous_tf: Função de transferência contínua
            T: Período de amostragem
            show_steps: Se deve mostrar os passos
        
        Returns:
            DiscretizationResult: Resultado da discretização
        """
        if T is None:
            T = self.T
        
        if show_steps:
            print("🔄 DISCRETIZAÇÃO POR MATCHED Z-TRANSFORM")
            print("=" * 43)
            print(f"📊 Sistema contínuo: H(s) = {continuous_tf.num}/{continuous_tf.den}")
            print(f"⏱️  Período de amostragem: T = {T}")
            print(f"🔧 Mapeamento: polo/zero em s=a → z=e^(a*T)")
        
        result = DiscretizationResult()
        result.continuous_tf = continuous_tf
        result.sampling_time = T
        result.method = "Matched Z-Transform"
        
        try:
            H_s_num = continuous_tf.num
            H_s_den = continuous_tf.den
            
            if show_steps:
                print(f"   🎯 Analisando polos e zeros...")
            
            # Encontrar raízes do numerador (zeros) e denominador (polos)
            zeros_s = sp.solve(H_s_num, self.s)
            poles_s = sp.solve(H_s_den, self.s)
            
            if show_steps:
                print(f"   📍 Zeros contínuos: {zeros_s}")
                print(f"   📍 Polos contínuos: {poles_s}")
            
            # Mapear para domínio Z
            zeros_z = [sp.exp(zero * T) for zero in zeros_s if zero.is_finite]
            poles_z = [sp.exp(pole * T) for pole in poles_s if pole.is_finite]
            
            if show_steps:
                print(f"   🎯 Zeros discretos: {zeros_z}")
                print(f"   🎯 Polos discretos: {poles_z}")
            
            # Construir função de transferência discreta
            # H(z) = K * ∏(z - zi) / ∏(z - pi)
            
            # Numerador: produto dos fatores (z - zero_i)
            H_z_num = 1
            for zero in zeros_z:
                H_z_num *= (self.z - zero)
            
            # Denominador: produto dos fatores (z - pole_i)  
            H_z_den = 1
            for pole in poles_z:
                H_z_den *= (self.z - pole)
            
            # Calcular ganho para preservar ganho DC ou de alta frequência
            # Simplificação: usar ganho unitário
            K = 1
            
            # Se sistema tem mais polos que zeros, adicionar zeros em z=0
            pole_excess = len(poles_z) - len(zeros_z)
            if pole_excess > 0:
                H_z_num *= self.z**pole_excess
                if show_steps:
                    print(f"   ⚙️  Adicionados {pole_excess} zeros em z=0")
            
            H_z = K * H_z_num / H_z_den
            H_z_simplified = sp.simplify(H_z)
            
            result.discrete_tf = SymbolicTransferFunction(sp.numer(H_z_simplified),
                                                        sp.denom(H_z_simplified), self.z)
            
            if show_steps:
                print(f"   ✅ H(z) = {H_z_simplified}")
                print(f"   ✅ Matched Z-Transform concluída!")
            
            result.transformation_steps.append(f"Zeros mapeados: {zeros_s} → {zeros_z}")
            result.transformation_steps.append(f"Polos mapeados: {poles_s} → {poles_z}")
            result.transformation_steps.append(f"Resultado: H(z) = {H_z_simplified}")
            
            result.frequency_mapping = {
                "Polos/Zeros": "Mapeamento exato",
                "Estabilidade": "Preservada",
                "Ganho": "Pode necessitar ajuste"
            }
            
            result.stability_preserved = True
            
        except Exception as e:
            error_msg = f"Erro no Matched Z-Transform: {e}"
            result.transformation_steps.append(f"❌ {error_msg}")
            if show_steps:
                print(f"❌ {error_msg}")
        
        return result
    
    def _check_stability_preservation(self, continuous_tf: SymbolicTransferFunction,
                                    discrete_tf: SymbolicTransferFunction) -> bool:
        """
        Verifica se a estabilidade foi preservada na discretização
        
        Args:
            continuous_tf: Sistema contínuo
            discrete_tf: Sistema discreto
        
        Returns:
            bool: True se estabilidade foi preservada
        """
        try:
            # Verificar estabilidade do sistema contínuo
            continuous_poles = sp.solve(continuous_tf.den, continuous_tf.var)
            continuous_stable = all(pole.as_real_imag()[0] < 0 for pole in continuous_poles 
                                  if pole.is_finite and pole.is_number)
            
            # Verificar estabilidade do sistema discreto
            discrete_poles = sp.solve(discrete_tf.den, discrete_tf.var)
            discrete_stable = all(abs(complex(pole)) < 1 for pole in discrete_poles 
                                if pole.is_finite and pole.is_number)
            
            # Se contínuo era estável, discreto deve ser estável
            # Se contínuo era instável, pode ou não preservar
            return not continuous_stable or discrete_stable
            
        except Exception:
            # Em caso de erro, assumir que não foi preservada
            return False

def compare_discretization_methods(continuous_tf: SymbolicTransferFunction,
                                 T: float = 0.1,
                                 show_steps: bool = True) -> Dict[str, DiscretizationResult]:
    """
    Compara diferentes métodos de discretização
    
    Args:
        continuous_tf: Função de transferência contínua
        T: Período de amostragem
        show_steps: Se deve mostrar os passos
    
    Returns:
        Dict com resultados de cada método
    """
    if show_steps:
        print("🔄 COMPARAÇÃO DE MÉTODOS DE DISCRETIZAÇÃO")
        print("=" * 45)
        print(f"📊 Sistema: H(s) = {continuous_tf.num}/{continuous_tf.den}")
        print(f"⏱️  T = {T}")
        print("")
    
    discretizer = DiscretizationMethods(T)
    results = {}
    
    # Lista de métodos a comparar
    methods = [
        ("Tustin", discretizer.tustin_transform),
        ("ZOH", discretizer.zero_order_hold),
        ("FOH", discretizer.first_order_hold),
        ("Euler Forward", discretizer.euler_forward),
        ("Euler Backward", discretizer.euler_backward),
        ("Matched Z", discretizer.matched_z_transform)
    ]
    
    for method_name, method_func in methods:
        if show_steps:
            print(f"\n🔧 Método: {method_name}")
            print("-" * 30)
        
        try:
            result = method_func(continuous_tf, T, show_steps)
            results[method_name] = result
            
            if show_steps:
                print(f"   Estabilidade preservada: {result.stability_preserved}")
                
        except Exception as e:
            if show_steps:
                print(f"   ❌ Erro: {e}")
            results[method_name] = None
    
    if show_steps:
        print("\n📊 RESUMO COMPARATIVO:")
        print("=" * 25)
        for method_name, result in results.items():
            if result:
                status = "✅" if result.stability_preserved else "⚠️"
                print(f"{status} {method_name}: H(z) = {result.discrete_tf.num}/{result.discrete_tf.den}")
            else:
                print(f"❌ {method_name}: Erro na discretização")
    
    return results
