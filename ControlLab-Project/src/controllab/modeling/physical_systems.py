"""
Módulo de Sistemas Físicos
==========================

Este módulo implementa modelagem de sistemas físicos a partir de 
primeiros princípios, incluindo sistemas mecânicos, elétricos e térmicos.

Classes:
    MechanicalSystem: Sistemas massa-mola-amortecedor
    ElectricalSystem: Circuitos RLC
    ThermalSystem: Sistemas térmicos
    PhysicalSystemBase: Classe base para todos os sistemas físicos
"""

import sympy as sp
from sympy import symbols, Function, Eq, solve, diff, simplify, sqrt
from sympy import laplace_transform, inverse_laplace_transform
from typing import Dict, List, Tuple, Any, Optional, Union
import warnings

# Importar transformada de Laplace do módulo correspondente
try:
    from .laplace_transform import LaplaceTransformer, TransformationHistory
except ImportError:
    # Fallback básico
    LaplaceTransformer = None
    TransformationHistory = None


class PhysicalSystemHistory:
    """Histórico da derivação de sistemas físicos"""
    
    def __init__(self):
        self.steps = []
        self.physical_laws = []
        self.equations = []
        self.assumptions = []
        
    def add_physical_law(self, law_name: str, description: str, equation: Any):
        law = {
            'name': law_name,
            'description': description,
            'equation': equation
        }
        self.physical_laws.append(law)
        
    def add_derivation_step(self, description: str, equation: Any, explanation: str = ""):
        step = {
            'step': len(self.steps) + 1,
            'description': description,
            'equation': equation,
            'explanation': explanation
        }
        self.steps.append(step)
        
    def add_assumption(self, assumption: str):
        self.assumptions.append(assumption)
        
    def get_formatted_derivation(self) -> str:
        if not self.steps:
            return "Nenhuma derivação registrada."
            
        derivation = "🏗️ DERIVAÇÃO DO SISTEMA FÍSICO\n"
        derivation += "=" * 50 + "\n"
        
        if self.assumptions:
            derivation += "\n📋 HIPÓTESES ASSUMIDAS:\n"
            for i, assumption in enumerate(self.assumptions, 1):
                derivation += f"  {i}. {assumption}\n"
        
        if self.physical_laws:
            derivation += "\n⚖️ LEIS FÍSICAS APLICADAS:\n"
            for law in self.physical_laws:
                derivation += f"  • {law['name']}: {law['description']}\n"
                derivation += f"    Equação: {law['equation']}\n"
        
        derivation += "\n🔧 PASSOS DA DERIVAÇÃO:\n"
        for step in self.steps:
            derivation += f"\n{step['step']}. {step['description']}\n"
            derivation += f"   Equação: {step['equation']}\n"
            if step['explanation']:
                derivation += f"   Explicação: {step['explanation']}\n"
            derivation += "-" * 30 + "\n"
            
        return derivation


class PhysicalSystemBase:
    """Classe base para todos os sistemas físicos"""
    
    def __init__(self, system_name: str):
        self.system_name = system_name
        self.t = symbols('t', real=True, positive=True)
        self.s = symbols('s', complex=True)
        self.history = PhysicalSystemHistory()
        self.parameters = {}
        self.variables = {}
        self.differential_equation = None
        self.transfer_function = None
        
    def add_parameter(self, name: str, symbol: Any, description: str, units: str = ""):
        """Adiciona um parâmetro físico ao sistema"""
        self.parameters[name] = {
            'symbol': symbol,
            'description': description,
            'units': units
        }
        
    def add_variable(self, name: str, function: Any, description: str, units: str = ""):
        """Adiciona uma variável do sistema"""
        self.variables[name] = {
            'function': function,
            'description': description,
            'units': units
        }
        
    def derive_equations(self):
        """Método abstrato para derivar equações - deve ser implementado pelas subclasses"""
        raise NotImplementedError("Subclasses devem implementar derive_equations()")
        
    def apply_laplace_modeling(self, show_steps: bool = True):
        """Aplica transformada de Laplace para obter função de transferência"""
        if self.differential_equation is None:
            raise ValueError("Equação diferencial não foi derivada ainda")
            
        if show_steps:
            self.history.add_derivation_step(
                "Aplicação da Transformada de Laplace",
                self.differential_equation,
                "Convertendo EDO para função de transferência"
            )
        
        try:
            # Usar transformador de Laplace se disponível
            if LaplaceTransformer:
                transformer = LaplaceTransformer()
                transformer.history = self.history if hasattr(self.history, 'steps') else TransformationHistory()
                
                # Para simplificar, vamos assumir condições iniciais nulas
                # Em uma implementação completa, isso seria parametrizável
                
                # Aplicar transformada de Laplace simbolicamente
                # Esta é uma implementação simplificada
                tf_expr = self._derive_transfer_function()
                
                if show_steps:
                    self.history.add_derivation_step(
                        "Função de transferência obtida",
                        tf_expr,
                        "G(s) = Y(s)/U(s) com condições iniciais nulas"
                    )
                
                self.transfer_function = tf_expr
                return tf_expr
            else:
                # Fallback básico
                tf_expr = self._derive_transfer_function()
                self.transfer_function = tf_expr
                return tf_expr
                
        except Exception as e:
            if show_steps:
                self.history.add_derivation_step(
                    "Erro na aplicação de Laplace",
                    str(e),
                    "Falha na conversão para função de transferência"
                )
            raise ValueError(f"Erro na modelagem com Laplace: {e}")
    
    def _derive_transfer_function(self):
        """Método abstrato para derivar função de transferência"""
        raise NotImplementedError("Subclasses devem implementar _derive_transfer_function()")
    
    def get_system_summary(self) -> str:
        """Retorna resumo completo do sistema"""
        summary = f"📊 RESUMO DO SISTEMA: {self.system_name.upper()}\n"
        summary += "=" * 60 + "\n"
        
        if self.parameters:
            summary += "\n🔧 PARÂMETROS:\n"
            for name, info in self.parameters.items():
                summary += f"  {name}: {info['description']}"
                if info['units']:
                    summary += f" [{info['units']}]"
                summary += f" = {info['symbol']}\n"
        
        if self.variables:
            summary += "\n📈 VARIÁVEIS:\n"
            for name, info in self.variables.items():
                summary += f"  {name}: {info['description']}"
                if info['units']:
                    summary += f" [{info['units']}]"
                summary += f" = {info['function']}\n"
        
        if self.differential_equation:
            summary += f"\n⚖️ EQUAÇÃO DIFERENCIAL:\n"
            summary += f"  {self.differential_equation}\n"
        
        if self.transfer_function:
            summary += f"\n🔄 FUNÇÃO DE TRANSFERÊNCIA:\n"
            summary += f"  G(s) = {self.transfer_function}\n"
        
        return summary


class MechanicalSystem(PhysicalSystemBase):
    """Sistema mecânico massa-mola-amortecedor"""
    
    def __init__(self, mass=None, damping=None, stiffness=None):
        super().__init__("Sistema Mecânico")
        
        # Definir parâmetros
        m = symbols('m', positive=True) if mass is None else mass
        b = symbols('b', positive=True) if damping is None else damping  
        k = symbols('k', positive=True) if stiffness is None else stiffness
        
        self.add_parameter('mass', m, 'Massa do sistema', 'kg')
        self.add_parameter('damping', b, 'Coeficiente de amortecimento', 'N·s/m')
        self.add_parameter('stiffness', k, 'Constante da mola', 'N/m')
        
        # Definir variáveis
        x = Function('x')(self.t)  # Posição
        F = Function('F')(self.t)  # Força aplicada
        
        self.add_variable('position', x, 'Posição da massa', 'm')
        self.add_variable('force', F, 'Força aplicada', 'N')
        
        # Adicionar hipóteses
        self.history.add_assumption("Sistema linear e invariante no tempo")
        self.history.add_assumption("Massa concentrada (corpo rígido)")
        self.history.add_assumption("Amortecimento viscoso (proporcional à velocidade)")
        self.history.add_assumption("Mola ideal (força proporcional ao deslocamento)")
        
    def derive_equations(self, show_steps: bool = True):
        """Deriva a equação diferencial do sistema massa-mola-amortecedor"""
        
        m = self.parameters['mass']['symbol']
        b = self.parameters['damping']['symbol']
        k = self.parameters['stiffness']['symbol']
        x = self.variables['position']['function']
        F = self.variables['force']['function']
        
        if show_steps:
            # Leis físicas
            self.history.add_physical_law(
                "Segunda Lei de Newton",
                "Força resultante = massa × aceleração",
                Eq(sp.symbols('F_net'), m * x.diff(self.t, 2))
            )
            
            self.history.add_physical_law(
                "Lei de Hooke",
                "Força da mola proporcional ao deslocamento",
                Eq(sp.symbols('F_mola'), -k * x)
            )
            
            self.history.add_physical_law(
                "Amortecimento Viscoso",
                "Força de amortecimento proporcional à velocidade",
                Eq(sp.symbols('F_amort'), -b * x.diff(self.t))
            )
        
        # Somatório de forças
        if show_steps:
            self.history.add_derivation_step(
                "Diagrama de corpo livre",
                "F(t) - b·ẋ(t) - k·x(t) = m·ẍ(t)",
                "Força aplicada - força amortecimento - força mola = força inercial"
            )
        
        # Equação diferencial
        ode = Eq(m * x.diff(self.t, 2) + b * x.diff(self.t) + k * x, F)
        
        if show_steps:
            self.history.add_derivation_step(
                "Equação diferencial do movimento",
                ode,
                "EDO de 2ª ordem que governa o sistema"
            )
        
        # Forma padrão
        standard_form = Eq(x.diff(self.t, 2) + (b/m) * x.diff(self.t) + (k/m) * x, F/m)
        
        if show_steps:
            self.history.add_derivation_step(
                "Forma padrão (normalizada)",
                standard_form,
                "Dividindo por m para forma canônica"
            )
        
        self.differential_equation = ode
        return ode
    
    def _derive_transfer_function(self):
        """Deriva a função de transferência G(s) = X(s)/F(s)"""
        
        m = self.parameters['mass']['symbol']
        b = self.parameters['damping']['symbol']
        k = self.parameters['stiffness']['symbol']
        
        # G(s) = 1/(m·s² + b·s + k)
        transfer_function = 1 / (m * self.s**2 + b * self.s + k)
        
        return transfer_function
    
    def analyze_natural_response(self):
        """Analisa a resposta natural do sistema"""
        m = self.parameters['mass']['symbol']
        b = self.parameters['damping']['symbol']
        k = self.parameters['stiffness']['symbol']
        
        # Frequência natural não amortecida
        wn = sqrt(k/m)
        
        # Razão de amortecimento
        zeta = b / (2 * sqrt(k * m))
        
        # Frequência natural amortecida
        wd = wn * sqrt(1 - zeta**2)
        
        analysis = {
            'natural_frequency': wn,
            'damping_ratio': zeta, 
            'damped_frequency': wd,
            'characteristic_equation': m * self.s**2 + b * self.s + k
        }
        
        # Classificar tipo de resposta
        if zeta < 0:
            analysis['response_type'] = "Instável"
        elif zeta == 0:
            analysis['response_type'] = "Não amortecido (oscilatório)"
        elif 0 < zeta < 1:
            analysis['response_type'] = "Subamortecido"
        elif zeta == 1:
            analysis['response_type'] = "Criticamente amortecido"
        else:
            analysis['response_type'] = "Sobreamortecido"
        
        return analysis


class ElectricalSystem(PhysicalSystemBase):
    """Sistema elétrico RLC"""
    
    def __init__(self, resistance=None, inductance=None, capacitance=None, circuit_type='series'):
        super().__init__("Sistema Elétrico")
        
        # Definir parâmetros
        R = symbols('R', positive=True) if resistance is None else resistance
        L = symbols('L', positive=True) if inductance is None else inductance
        C = symbols('C', positive=True) if capacitance is None else capacitance
        
        self.add_parameter('resistance', R, 'Resistência', 'Ω')
        self.add_parameter('inductance', L, 'Indutância', 'H')
        self.add_parameter('capacitance', C, 'Capacitância', 'F')
        
        self.circuit_type = circuit_type
        
        # Definir variáveis dependendo do tipo de circuito
        if circuit_type == 'series':
            i = Function('i')(self.t)  # Corrente
            v = Function('v')(self.t)  # Tensão de entrada
            vc = Function('vc')(self.t)  # Tensão no capacitor (saída)
            
            self.add_variable('current', i, 'Corrente do circuito', 'A')
            self.add_variable('input_voltage', v, 'Tensão de entrada', 'V')
            self.add_variable('output_voltage', vc, 'Tensão no capacitor', 'V')
        else:
            # Circuito paralelo - implementação futura
            raise NotImplementedError("Circuito paralelo não implementado ainda")
        
        # Hipóteses
        self.history.add_assumption("Componentes ideais e lineares")
        self.history.add_assumption("Temperatura constante")
        self.history.add_assumption("Frequência de operação dentro da faixa linear")
        
    def derive_equations(self, show_steps: bool = True):
        """Deriva equações do circuito RLC série"""
        
        if self.circuit_type != 'series':
            raise NotImplementedError("Apenas circuito série implementado")
            
        R = self.parameters['resistance']['symbol']
        L = self.parameters['inductance']['symbol'] 
        C = self.parameters['capacitance']['symbol']
        
        i = self.variables['current']['function']
        v = self.variables['input_voltage']['function']
        vc = self.variables['output_voltage']['function']
        
        if show_steps:
            # Leis de Kirchhoff
            self.history.add_physical_law(
                "Lei de Kirchhoff das Tensões (LKT)",
                "Soma das tensões em malha fechada = 0",
                Eq(sp.symbols('sum_V'), 0)
            )
            
            self.history.add_physical_law(
                "Lei de Ohm",
                "Tensão no resistor = R × corrente",
                Eq(sp.symbols('VR'), R * i)
            )
            
            self.history.add_physical_law(
                "Lei do Indutor",
                "Tensão no indutor = L × di/dt",
                Eq(sp.symbols('VL'), L * i.diff(self.t))
            )
            
            self.history.add_physical_law(
                "Lei do Capacitor",
                "Corrente no capacitor = C × dv/dt",
                Eq(i, C * vc.diff(self.t))
            )
        
        # Aplicar LKT: v(t) = vR + vL + vC
        if show_steps:
            self.history.add_derivation_step(
                "Aplicação da LKT",
                "v(t) = R·i(t) + L·di/dt + vc(t)",
                "Soma das tensões na malha"
            )
        
        # Substituir i = C·dvc/dt
        lkt_equation = Eq(v, R * C * vc.diff(self.t) + L * C * vc.diff(self.t, 2) + vc)
        
        if show_steps:
            self.history.add_derivation_step(
                "Substituição da corrente",
                lkt_equation,
                "Usando i = C·dvc/dt para eliminar i"
            )
        
        # Reordenar para forma padrão
        standard_form = Eq(L * C * vc.diff(self.t, 2) + R * C * vc.diff(self.t) + vc, v)
        
        if show_steps:
            self.history.add_derivation_step(
                "Equação diferencial final",
                standard_form,
                "EDO de 2ª ordem para tensão no capacitor"
            )
        
        self.differential_equation = standard_form
        return standard_form
    
    def _derive_transfer_function(self):
        """Deriva G(s) = Vc(s)/V(s)"""
        
        R = self.parameters['resistance']['symbol']
        L = self.parameters['inductance']['symbol']
        C = self.parameters['capacitance']['symbol']
        
        # G(s) = 1/(LCs² + RCs + 1)
        transfer_function = 1 / (L * C * self.s**2 + R * C * self.s + 1)
        
        return transfer_function


class ThermalSystem(PhysicalSystemBase):
    """Sistema térmico de primeira ordem"""
    
    def __init__(self, thermal_resistance=None, thermal_capacitance=None):
        super().__init__("Sistema Térmico")
        
        # Parâmetros térmicos
        R_th = symbols('R_th', positive=True) if thermal_resistance is None else thermal_resistance
        C_th = symbols('C_th', positive=True) if thermal_capacitance is None else thermal_capacitance
        
        self.add_parameter('thermal_resistance', R_th, 'Resistência térmica', 'K/W')
        self.add_parameter('thermal_capacitance', C_th, 'Capacitância térmica', 'J/K')
        
        # Variáveis
        T = Function('T')(self.t)  # Temperatura
        Q = Function('Q')(self.t)  # Fluxo de calor
        T_amb = symbols('T_amb')   # Temperatura ambiente
        
        self.add_variable('temperature', T, 'Temperatura do sistema', 'K')
        self.add_variable('heat_flow', Q, 'Fluxo de calor aplicado', 'W')
        self.add_parameter('ambient_temp', T_amb, 'Temperatura ambiente', 'K')
        
        # Hipóteses
        self.history.add_assumption("Temperatura uniforme no sistema (análise concentrada)")
        self.history.add_assumption("Propriedades térmicas constantes")
        self.history.add_assumption("Transferência de calor linear")
        
    def derive_equations(self, show_steps: bool = True):
        """Deriva equação do sistema térmico"""
        
        R_th = self.parameters['thermal_resistance']['symbol']
        C_th = self.parameters['thermal_capacitance']['symbol']
        T_amb = self.parameters['ambient_temp']['symbol']
        
        T = self.variables['temperature']['function']
        Q = self.variables['heat_flow']['function']
        
        if show_steps:
            self.history.add_physical_law(
                "Primeira Lei da Termodinâmica",
                "Taxa de armazenamento = Fluxo de entrada - Fluxo de saída",
                Eq(sp.symbols('dE/dt'), sp.symbols('Q_in - Q_out'))
            )
            
            self.history.add_physical_law(
                "Lei de Fourier",
                "Fluxo de calor proporcional ao gradiente de temperatura",
                Eq(sp.symbols('Q_cond'), sp.symbols('(T1-T2)/R_th'))
            )
        
        # Balanço de energia: C_th * dT/dt = Q(t) - (T(t) - T_amb)/R_th
        if show_steps:
            self.history.add_derivation_step(
                "Balanço de energia",
                "C_th·dT/dt = Q(t) - (T(t) - T_amb)/R_th",
                "Energia armazenada = entrada - saída por condução"
            )
        
        energy_balance = Eq(C_th * T.diff(self.t), Q - (T - T_amb)/R_th)
        
        # Forma padrão com T_amb como referência
        if show_steps:
            self.history.add_derivation_step(
                "Reorganização",
                "R_th·C_th·dT/dt + T(t) = R_th·Q(t) + T_amb",
                "Multiplicando por R_th e reorganizando"
            )
        
        standard_form = Eq(R_th * C_th * T.diff(self.t) + T, R_th * Q + T_amb)
        
        self.differential_equation = standard_form
        return standard_form
    
    def _derive_transfer_function(self):
        """Deriva G(s) = T(s)/Q(s) (assumindo T_amb = 0)"""
        
        R_th = self.parameters['thermal_resistance']['symbol']
        C_th = self.parameters['thermal_capacitance']['symbol']
        
        # G(s) = R_th/(R_th·C_th·s + 1)
        transfer_function = R_th / (R_th * C_th * self.s + 1)
        
        return transfer_function
    
    def analyze_thermal_response(self):
        """Analisa características da resposta térmica"""
        R_th = self.parameters['thermal_resistance']['symbol']
        C_th = self.parameters['thermal_capacitance']['symbol']
        
        # Constante de tempo
        tau = R_th * C_th
        
        # Ganho estático
        K = R_th
        
        analysis = {
            'time_constant': tau,
            'steady_state_gain': K,
            'system_type': 'Primeira ordem',
            'pole': -1/tau,
            'rise_time': 2.2 * tau,  # 10% a 90%
            'settling_time': 4 * tau  # 2% do valor final
        }
        
        return analysis


# Funções de conveniência para criação rápida de sistemas
def create_mass_spring_damper(m=1, b=0.5, k=1):
    """Cria sistema massa-mola-amortecedor com valores específicos"""
    system = MechanicalSystem(mass=m, damping=b, stiffness=k)
    system.derive_equations()
    system.apply_laplace_modeling()
    return system


def create_rlc_circuit(R=1, L=1, C=1):
    """Cria circuito RLC série com valores específicos"""
    system = ElectricalSystem(resistance=R, inductance=L, capacitance=C)
    system.derive_equations()
    system.apply_laplace_modeling()
    return system


def create_thermal_system(R_th=1, C_th=1):
    """Cria sistema térmico com valores específicos"""
    system = ThermalSystem(thermal_resistance=R_th, thermal_capacitance=C_th)
    system.derive_equations()
    system.apply_laplace_modeling()
    return system
