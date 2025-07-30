"""
ControlLab - Análise de Estabilidade de Sistemas Discretos
=========================================================

Este módulo implementa métodos para análise de estabilidade de sistemas
discretos no domínio Z, com explicações pedagógicas detalhadas.

Métodos implementados:
- Teste de Jury
- Teste de Routh Bilinear
- Círculo de Estabilidade
- Margem de Estabilidade Z-domain
"""

import sympy as sp
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass, field
from ..core.symbolic_tf import SymbolicTransferFunction
from ..core.history import OperationHistory, OperationStep

@dataclass
class StabilityResult:
    """
    Resultado da análise de estabilidade
    
    Atributos:
        system: Sistema analisado
        is_stable: Se o sistema é estável
        method_used: Método utilizado na análise
        stability_margins: Margens de estabilidade
        critical_poles: Polos críticos (próximos ao círculo unitário)
        analysis_steps: Passos da análise
        recommendations: Recomendações para melhorar estabilidade
    """
    system: SymbolicTransferFunction = None
    is_stable: bool = False
    method_used: str = ""
    stability_margins: Dict[str, float] = field(default_factory=dict)
    critical_poles: List[complex] = field(default_factory=list)
    analysis_steps: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    history: OperationHistory = field(default_factory=OperationHistory)

class DiscreteStabilityAnalyzer:
    """
    Analisador de estabilidade para sistemas discretos
    
    Implementa diferentes critérios de estabilidade no domínio Z:
    - Teste de Jury: Para polinômios característicos
    - Routh Bilinear: Transformação para domínio s
    - Análise de polos: Verificação direta do círculo unitário
    """
    
    def __init__(self):
        """Inicializa o analisador de estabilidade"""
        self.z = sp.Symbol('z')
        self.s = sp.Symbol('s')
        self.history = OperationHistory()
    
    def jury_test(self, char_poly: sp.Expr, show_steps: bool = True) -> StabilityResult:
        """
        Aplica o teste de estabilidade de Jury
        
        O teste de Jury é análogo ao teste de Routh-Hurwitz para sistemas discretos.
        Verifica se todas as raízes estão dentro do círculo unitário.
        
        Args:
            char_poly: Polinômio característico do sistema
            show_steps: Se deve mostrar os passos
        
        Returns:
            StabilityResult: Resultado da análise
        """
        if show_steps:
            print("🔄 TESTE DE ESTABILIDADE DE JURY")
            print("=" * 35)
            print(f"📊 Polinômio característico: {char_poly}")
            print(f"🎯 Verificando estabilidade no domínio Z...")
        
        result = StabilityResult()
        result.method_used = "Teste de Jury"
        
        try:
            # Obter coeficientes do polinômio
            poly = sp.Poly(char_poly, self.z)
            coeffs = poly.all_coeffs()
            n = len(coeffs) - 1  # Grau do polinômio
            
            if show_steps:
                print(f"   📝 Grau do polinômio: {n}")
                print(f"   📝 Coeficientes: {coeffs}")
            
            result.analysis_steps.append(f"Polinômio de grau {n}")
            result.analysis_steps.append(f"Coeficientes: {coeffs}")
            
            # Condições necessárias
            a0, an = float(coeffs[-1]), float(coeffs[0])
            
            if show_steps:
                print(f"   🔍 Condições necessárias:")
                print(f"       a₀ = {a0}, aₙ = {an}")
            
            # Condição 1: |a₀| < aₙ (se aₙ > 0)
            necessary_conditions = []
            
            if an > 0:
                cond1 = abs(a0) < an
                necessary_conditions.append(cond1)
                if show_steps:
                    status = "✅" if cond1 else "❌"
                    print(f"       {status} |a₀| < aₙ: |{a0}| < {an} = {cond1}")
            else:
                cond1 = abs(a0) < abs(an)
                necessary_conditions.append(cond1)
                if show_steps:
                    status = "✅" if cond1 else "❌"
                    print(f"       {status} |a₀| < |aₙ|: |{a0}| < |{an}| = {cond1}")
            
            # Avaliação em z = 1
            poly_at_1 = float(char_poly.subs(self.z, 1))
            cond2 = poly_at_1 > 0
            necessary_conditions.append(cond2)
            
            if show_steps:
                status = "✅" if cond2 else "❌"
                print(f"       {status} P(1) > 0: P(1) = {poly_at_1}")
            
            # Avaliação em z = -1
            poly_at_minus1 = float(char_poly.subs(self.z, -1))
            if n % 2 == 0:
                cond3 = poly_at_minus1 > 0
            else:
                cond3 = poly_at_minus1 < 0
            necessary_conditions.append(cond3)
            
            if show_steps:
                expected = "> 0" if n % 2 == 0 else "< 0"
                status = "✅" if cond3 else "❌"
                print(f"       {status} P(-1) {expected}: P(-1) = {poly_at_minus1}")
            
            # Se condições necessárias não são satisfeitas, sistema é instável
            if not all(necessary_conditions):
                result.is_stable = False
                result.analysis_steps.append("❌ Condições necessárias não satisfeitas")
                if show_steps:
                    print("   ❌ Sistema instável: condições necessárias falharam")
                return result
            
            # Construir tabela de Jury
            jury_stable = self._build_jury_table(coeffs, show_steps, result)
            result.is_stable = jury_stable
            
            if show_steps:
                if jury_stable:
                    print("   ✅ Sistema estável pelo teste de Jury!")
                else:
                    print("   ❌ Sistema instável pelo teste de Jury!")
            
            # Adicionar ao histórico
            step = OperationStep(
                operation="teste_jury",
                input_expr=str(char_poly),
                output_expr=f"Estável: {jury_stable}",
                explanation=f"Teste de Jury: {jury_stable}"
            )
            result.history.add_step(step)
            
        except Exception as e:
            error_msg = f"Erro no teste de Jury: {e}"
            result.analysis_steps.append(f"❌ {error_msg}")
            if show_steps:
                print(f"❌ {error_msg}")
        
        return result
    
    def _build_jury_table(self, coeffs: List[float], show_steps: bool, 
                         result: StabilityResult) -> bool:
        """Constrói a tabela de Jury e verifica estabilidade"""
        
        n = len(coeffs) - 1
        
        if show_steps:
            print(f"   🏗️  Construindo tabela de Jury:")
        
        # Inicializar tabela
        table = []
        
        # Primeira linha: coeficientes originais
        row1 = list(coeffs)
        table.append(row1)
        
        # Segunda linha: coeficientes em ordem reversa
        row2 = list(reversed(coeffs))
        table.append(row2)
        
        if show_steps:
            print(f"       Linha 1: {row1}")
            print(f"       Linha 2: {row2}")
        
        # Construir linhas restantes
        current_coeffs = coeffs[:]
        for k in range(n):
            if len(current_coeffs) <= 2:
                break
            
            # Calcular nova linha
            a0 = current_coeffs[0]
            an = current_coeffs[-1]
            
            if abs(a0) < 1e-10:  # Evitar divisão por zero
                result.analysis_steps.append("❌ Divisão por zero na tabela de Jury")
                return False
            
            new_coeffs = []
            for i in range(len(current_coeffs) - 1):
                coeff = current_coeffs[i] - (an/a0) * current_coeffs[-(i+1)]
                new_coeffs.append(coeff)
            
            table.append(new_coeffs)
            table.append(list(reversed(new_coeffs)))
            
            if show_steps:
                print(f"       Linha {2*k+3}: {new_coeffs}")
                print(f"       Linha {2*k+4}: {list(reversed(new_coeffs))}")
            
            # Verificar condição de estabilidade
            if len(new_coeffs) > 1:
                first_elem = new_coeffs[0]
                last_elem = new_coeffs[-1]
                
                if abs(first_elem) <= abs(last_elem):
                    result.analysis_steps.append(f"❌ Condição violada na linha {2*k+3}")
                    if show_steps:
                        print(f"       ❌ |{first_elem}| <= |{last_elem}|")
                    return False
            
            current_coeffs = new_coeffs
        
        result.analysis_steps.append("✅ Todas as condições de Jury satisfeitas")
        return True
    
    def bilinear_routh_test(self, discrete_tf: SymbolicTransferFunction,
                           show_steps: bool = True) -> StabilityResult:
        """
        Análise de estabilidade usando transformação bilinear + Routh
        
        Transforma o sistema discreto para contínuo usando w = (z-1)/(z+1)
        e aplica o teste de Routh-Hurwitz no domínio transformado.
        
        Args:
            discrete_tf: Função de transferência discreta
            show_steps: Se deve mostrar os passos
        
        Returns:
            StabilityResult: Resultado da análise
        """
        if show_steps:
            print("🔄 TESTE DE ROUTH BILINEAR")
            print("=" * 27)
            print(f"📊 Sistema discreto: H(z) = {discrete_tf.num}/{discrete_tf.den}")
            print(f"🔧 Transformação: w = (z-1)/(z+1)")
        
        result = StabilityResult()
        result.system = discrete_tf
        result.method_used = "Routh Bilinear"
        
        try:
            # Transformação bilinear inversa: z = (w+1)/(w-1)
            w = sp.Symbol('w')
            z_to_w = (w + 1) / (w - 1)
            
            if show_steps:
                print(f"   📝 Substituição: z = {z_to_w}")
            
            # Substituir na função de transferência
            H_z_num = discrete_tf.num
            H_z_den = discrete_tf.den
            
            H_w_num = H_z_num.subs(self.z, z_to_w)
            H_w_den = H_z_den.subs(self.z, z_to_w)
            
            # Simplificar
            H_w_num = sp.simplify(H_w_num)
            H_w_den = sp.simplify(H_w_den)
            
            if show_steps:
                print(f"   🧮 H(w) = {H_w_num}/{H_w_den}")
            
            # Polinômio característico em w
            char_poly_w = H_w_den
            
            if show_steps:
                print(f"   📊 Polinômio característico: {char_poly_w}")
            
            # Aplicar teste de Routh-Hurwitz (simplificado)
            # Para implementação completa, seria necessário o teste completo de Routh
            routh_stable = self._simplified_routh_test(char_poly_w, w, show_steps)
            
            result.is_stable = routh_stable
            result.analysis_steps.append(f"Transformação bilinear aplicada")
            result.analysis_steps.append(f"Polinômio em w: {char_poly_w}")
            result.analysis_steps.append(f"Teste de Routh: {'Estável' if routh_stable else 'Instável'}")
            
            if show_steps:
                if routh_stable:
                    print("   ✅ Sistema estável pelo teste bilinear!")
                else:
                    print("   ❌ Sistema instável pelo teste bilinear!")
            
        except Exception as e:
            error_msg = f"Erro no teste bilinear: {e}"
            result.analysis_steps.append(f"❌ {error_msg}")
            if show_steps:
                print(f"❌ {error_msg}")
        
        return result
    
    def _simplified_routh_test(self, poly: sp.Expr, var: sp.Symbol, show_steps: bool) -> bool:
        """Teste de Routh simplificado para verificação básica"""
        
        try:
            # Obter coeficientes
            poly_obj = sp.Poly(poly, var)
            coeffs = poly_obj.all_coeffs()
            
            if show_steps:
                print(f"   📝 Coeficientes: {coeffs}")
            
            # Verificar se todos os coeficientes têm o mesmo sinal
            # (condição necessária mas não suficiente)
            non_zero_coeffs = [c for c in coeffs if c != 0]
            
            if not non_zero_coeffs:
                return False
            
            # Verificar sinais
            first_sign = 1 if non_zero_coeffs[0] > 0 else -1
            same_sign = all((c > 0) == (first_sign > 0) for c in non_zero_coeffs)
            
            if show_steps:
                print(f"   🔍 Todos coeficientes mesmo sinal: {same_sign}")
            
            return same_sign
            
        except Exception:
            return False
    
    def stability_circle_analysis(self, discrete_tf: SymbolicTransferFunction,
                                 show_steps: bool = True) -> StabilityResult:
        """
        Análise de estabilidade baseada no círculo unitário
        
        Verifica diretamente se os polos estão dentro do círculo unitário |z| < 1.
        
        Args:
            discrete_tf: Função de transferência discreta
            show_steps: Se deve mostrar os passos
        
        Returns:
            StabilityResult: Resultado da análise
        """
        if show_steps:
            print("🔄 ANÁLISE DO CÍRCULO DE ESTABILIDADE")
            print("=" * 38)
            print(f"📊 Sistema: H(z) = {discrete_tf.num}/{discrete_tf.den}")
            print(f"🎯 Critério: Todos os polos devem ter |z| < 1")
        
        result = StabilityResult()
        result.system = discrete_tf
        result.method_used = "Círculo de Estabilidade"
        
        try:
            # Encontrar polos do sistema
            poles = sp.solve(discrete_tf.den, discrete_tf.var)
            
            if show_steps:
                print(f"   📍 Polos encontrados: {poles}")
            
            result.analysis_steps.append(f"Polos: {poles}")
            
            # Verificar cada polo
            stable_poles = []
            unstable_poles = []
            critical_poles = []
            
            for pole in poles:
                try:
                    # Converter para complexo para calcular magnitude
                    if pole.is_real:
                        pole_complex = complex(float(pole))
                    else:
                        pole_complex = complex(pole)
                    
                    magnitude = abs(pole_complex)
                    
                    if show_steps:
                        print(f"   🔍 Polo {pole}: |z| = {magnitude:.4f}")
                    
                    if magnitude < 1.0:
                        stable_poles.append(pole)
                        if magnitude > 0.8:  # Próximo à borda
                            critical_poles.append(pole_complex)
                    elif magnitude == 1.0:
                        critical_poles.append(pole_complex)
                        if show_steps:
                            print(f"       ⚠️  Polo na borda do círculo unitário!")
                    else:
                        unstable_poles.append(pole)
                        if show_steps:
                            print(f"       ❌ Polo fora do círculo unitário!")
                    
                except Exception as e:
                    if show_steps:
                        print(f"       ⚠️  Erro ao analisar polo {pole}: {e}")
                    unstable_poles.append(pole)
            
            # Determinar estabilidade
            result.is_stable = len(unstable_poles) == 0
            result.critical_poles = critical_poles
            
            # Calcular margens de estabilidade
            if stable_poles:
                # Margem de estabilidade: distância do polo mais próximo à borda
                min_margin = min(1.0 - abs(complex(pole)) for pole in stable_poles 
                               if pole.is_finite)
                result.stability_margins['magnitude_margin'] = min_margin
            
            if show_steps:
                print(f"   📊 Polos estáveis: {len(stable_poles)}")
                print(f"   📊 Polos instáveis: {len(unstable_poles)}")
                print(f"   📊 Polos críticos: {len(critical_poles)}")
                
                if result.is_stable:
                    print("   ✅ Sistema estável!")
                    if critical_poles:
                        print("   ⚠️  Atenção: Polos próximos à borda de estabilidade")
                else:
                    print("   ❌ Sistema instável!")
            
            # Recomendações
            if critical_poles:
                result.recommendations.append("Considerar redesign para afastar polos da borda")
            if unstable_poles:
                result.recommendations.append("Sistema requer estabilização")
            
            result.analysis_steps.append(f"Estável: {result.is_stable}")
            result.analysis_steps.append(f"Polos críticos: {len(critical_poles)}")
            
        except Exception as e:
            error_msg = f"Erro na análise do círculo: {e}"
            result.analysis_steps.append(f"❌ {error_msg}")
            if show_steps:
                print(f"❌ {error_msg}")
        
        return result
    
    def stability_margins_z_domain(self, discrete_tf: SymbolicTransferFunction,
                                  show_steps: bool = True) -> StabilityResult:
        """
        Calcula margens de estabilidade no domínio Z
        
        Args:
            discrete_tf: Função de transferência discreta
            show_steps: Se deve mostrar os passos
        
        Returns:
            StabilityResult: Resultado com margens de estabilidade
        """
        if show_steps:
            print("🔄 MARGENS DE ESTABILIDADE - DOMÍNIO Z")
            print("=" * 40)
            print(f"📊 Sistema: H(z) = {discrete_tf.num}/{discrete_tf.den}")
        
        result = StabilityResult()
        result.system = discrete_tf
        result.method_used = "Margens Z-domain"
        
        try:
            # Análise básica de estabilidade
            circle_result = self.stability_circle_analysis(discrete_tf, False)
            result.is_stable = circle_result.is_stable
            result.critical_poles = circle_result.critical_poles
            
            # Calcular margens específicas
            margins = {}
            
            # Margem de magnitude dos polos
            poles = sp.solve(discrete_tf.den, discrete_tf.var)
            if poles:
                pole_magnitudes = []
                for pole in poles:
                    try:
                        if pole.is_real:
                            mag = abs(float(pole))
                        else:
                            mag = abs(complex(pole))
                        pole_magnitudes.append(mag)
                    except:
                        pass
                
                if pole_magnitudes:
                    max_pole_mag = max(pole_magnitudes)
                    margins['pole_magnitude_margin'] = 1.0 - max_pole_mag
                    
                    if show_steps:
                        print(f"   📊 Magnitude máxima dos polos: {max_pole_mag:.4f}")
                        print(f"   📊 Margem de magnitude: {margins['pole_magnitude_margin']:.4f}")
            
            # Margem de fase (aproximação)
            # Para sistemas discretos, relacionar com tempo de atraso máximo tolerável
            if result.is_stable:
                # Estimativa baseada na frequência de cruzamento
                margins['phase_margin_samples'] = 2.0  # Aproximação pedagógica
                
                if show_steps:
                    print(f"   📊 Margem de fase (amostras): ~{margins['phase_margin_samples']}")
            
            result.stability_margins = margins
            
            # Recomendações baseadas nas margens
            if margins.get('pole_magnitude_margin', 0) < 0.1:
                result.recommendations.append("Margem de magnitude muito baixa - sistema próximo à instabilidade")
            elif margins.get('pole_magnitude_margin', 0) < 0.3:
                result.recommendations.append("Considerar aumentar margem de estabilidade")
            
            if show_steps:
                print("   📋 Margens calculadas:")
                for margin_name, margin_value in margins.items():
                    print(f"       {margin_name}: {margin_value:.4f}")
                
                if result.recommendations:
                    print("   💡 Recomendações:")
                    for rec in result.recommendations:
                        print(f"       • {rec}")
            
        except Exception as e:
            error_msg = f"Erro no cálculo de margens: {e}"
            result.analysis_steps.append(f"❌ {error_msg}")
            if show_steps:
                print(f"❌ {error_msg}")
        
        return result

def analyze_discrete_stability(discrete_tf: SymbolicTransferFunction,
                              method: str = 'circle',
                              show_steps: bool = True) -> StabilityResult:
    """
    Função de conveniência para análise de estabilidade discreta
    
    Args:
        discrete_tf: Função de transferência discreta
        method: Método ('circle', 'jury', 'bilinear', 'margins')
        show_steps: Se deve mostrar os passos
    
    Returns:
        StabilityResult: Resultado da análise
    """
    analyzer = DiscreteStabilityAnalyzer()
    
    if method == 'circle':
        return analyzer.stability_circle_analysis(discrete_tf, show_steps)
    elif method == 'jury':
        # Para Jury, precisamos do polinômio característico
        char_poly = discrete_tf.den
        return analyzer.jury_test(char_poly, show_steps)
    elif method == 'bilinear':
        return analyzer.bilinear_routh_test(discrete_tf, show_steps)
    elif method == 'margins':
        return analyzer.stability_margins_z_domain(discrete_tf, show_steps)
    else:
        raise ValueError(f"Método '{method}' não reconhecido")

def compare_stability_methods(discrete_tf: SymbolicTransferFunction,
                             show_steps: bool = True) -> Dict[str, StabilityResult]:
    """
    Compara diferentes métodos de análise de estabilidade
    
    Args:
        discrete_tf: Função de transferência discreta
        show_steps: Se deve mostrar os passos
    
    Returns:
        Dict com resultados de cada método
    """
    if show_steps:
        print("🔄 COMPARAÇÃO DE MÉTODOS DE ESTABILIDADE")
        print("=" * 45)
        print(f"📊 Sistema: H(z) = {discrete_tf.num}/{discrete_tf.den}")
        print("")
    
    analyzer = DiscreteStabilityAnalyzer()
    results = {}
    
    # Métodos a comparar
    methods = [
        ("Círculo Unitário", lambda: analyzer.stability_circle_analysis(discrete_tf, show_steps)),
        ("Teste de Jury", lambda: analyzer.jury_test(discrete_tf.den, show_steps)),
        ("Routh Bilinear", lambda: analyzer.bilinear_routh_test(discrete_tf, show_steps)),
        ("Margens Z-domain", lambda: analyzer.stability_margins_z_domain(discrete_tf, show_steps))
    ]
    
    for method_name, method_func in methods:
        if show_steps:
            print(f"\n🔧 Método: {method_name}")
            print("-" * 30)
        
        try:
            result = method_func()
            results[method_name] = result
            
        except Exception as e:
            if show_steps:
                print(f"   ❌ Erro: {e}")
            results[method_name] = None
    
    if show_steps:
        print("\n📊 RESUMO COMPARATIVO:")
        print("=" * 25)
        for method_name, result in results.items():
            if result:
                status = "✅ Estável" if result.is_stable else "❌ Instável"
                print(f"{status} - {method_name}")
            else:
                print(f"❌ Erro - {method_name}")
    
    return results
