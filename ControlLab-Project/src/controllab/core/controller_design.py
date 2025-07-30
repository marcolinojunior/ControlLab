"""
Módulo de design de controladores para sistemas de controle
"""

import sympy as sp
from typing import List, Dict, Union, Tuple, Optional
from .symbolic_tf import SymbolicTransferFunction
from .history import OperationHistory


class PIDController:
    """Design de controladores PID"""
    
    def __init__(self):
        self.history = OperationHistory()
    
    def design_pid(self, plant: SymbolicTransferFunction, 
                   kp: Union[sp.Symbol, float] = None,
                   ki: Union[sp.Symbol, float] = None,
                   kd: Union[sp.Symbol, float] = None) -> SymbolicTransferFunction:
        """
        Projeta controlador PID
        
        Args:
            plant: Planta a ser controlada
            kp: Ganho proporcional
            ki: Ganho integral
            kd: Ganho derivativo
            
        Returns:
            SymbolicTransferFunction: Controlador PID
        """
        # Define símbolos se não fornecidos
        if kp is None:
            kp = sp.Symbol('K_p', real=True)
        if ki is None:
            ki = sp.Symbol('K_i', real=True)
        if kd is None:
            kd = sp.Symbol('K_d', real=True)
        
        self.history.add_step(
            "DESIGN_PID",
            f"Projetando controlador PID",
            f"Planta: {plant}",
            f"Kp={kp}, Ki={ki}, Kd={kd}"
        )
        
        # Controlador PID: C(s) = Kp + Ki/s + Kd*s
        s = plant.variable
        pid_num = kd * s**2 + kp * s + ki
        pid_den = s
        
        controller = SymbolicTransferFunction(pid_num, pid_den, s)
        controller.history.steps = self.history.steps.copy()
        
        return controller
    
    def tune_ziegler_nichols(self, plant: SymbolicTransferFunction, 
                           critical_gain: Union[sp.Symbol, float],
                           critical_period: Union[sp.Symbol, float]) -> dict:
        """
        Sintonia de Ziegler-Nichols
        
        Args:
            plant: Planta
            critical_gain: Ganho crítico
            critical_period: Período crítico
            
        Returns:
            dict: Parâmetros sintonizados
        """
        self.history.add_step(
            "ZIEGLER_NICHOLS",
            "Aplicando método de Ziegler-Nichols",
            f"Ku={critical_gain}, Tu={critical_period}",
            "Calculando parâmetros PID"
        )
        
        # Regras de Ziegler-Nichols
        kp_p = 0.5 * critical_gain
        
        kp_pi = 0.45 * critical_gain
        ki_pi = 1.2 * kp_pi / critical_period
        
        kp_pid = 0.6 * critical_gain
        ki_pid = 2 * kp_pid / critical_period
        kd_pid = kp_pid * critical_period / 8
        
        return {
            'P': {'Kp': kp_p},
            'PI': {'Kp': kp_pi, 'Ki': ki_pi},
            'PID': {'Kp': kp_pid, 'Ki': ki_pid, 'Kd': kd_pid}
        }


class LeadLagCompensator:
    """Design de compensadores lead-lag"""
    
    def __init__(self):
        self.history = OperationHistory()
    
    def design_lead(self, desired_phase_margin: float,
                   crossover_frequency: Union[sp.Symbol, float],
                   alpha: Union[sp.Symbol, float] = None) -> SymbolicTransferFunction:
        """
        Projeta compensador lead
        
        Args:
            desired_phase_margin: Margem de fase desejada (graus)
            crossover_frequency: Frequência de cruzamento
            alpha: Parâmetro do compensador
            
        Returns:
            SymbolicTransferFunction: Compensador lead
        """
        s = sp.Symbol('s', complex=True)
        
        if alpha is None:
            # Calcula alpha baseado na margem de fase
            phi_max = sp.rad(desired_phase_margin - 5)  # 5° de margem adicional
            alpha = (1 - sp.sin(phi_max)) / (1 + sp.sin(phi_max))
        
        self.history.add_step(
            "DESIGN_LEAD",
            f"Projetando compensador lead",
            f"Margem de fase: {desired_phase_margin}°",
            f"α = {alpha}"
        )
        
        # Frequência de máximo avanço de fase
        omega_max = crossover_frequency / sp.sqrt(alpha)
        
        # Compensador lead: C(s) = (s + 1/T) / (s + 1/(α*T))
        T = 1 / (omega_max * sp.sqrt(alpha))
        
        lead_num = alpha * (T * s + 1)
        lead_den = alpha * T * s + 1
        
        compensator = SymbolicTransferFunction(lead_num, lead_den, s)
        compensator.history.steps = self.history.steps.copy()
        
        return compensator
    
    def design_lag(self, steady_state_error_requirement: float,
                  beta: Union[sp.Symbol, float] = None) -> SymbolicTransferFunction:
        """
        Projeta compensador lag
        
        Args:
            steady_state_error_requirement: Requisito de erro em regime
            beta: Parâmetro do compensador (β > 1)
            
        Returns:
            SymbolicTransferFunction: Compensador lag
        """
        s = sp.Symbol('s', complex=True)
        
        if beta is None:
            beta = sp.Symbol('beta', real=True, positive=True)
        
        self.history.add_step(
            "DESIGN_LAG",
            f"Projetando compensador lag",
            f"Erro em regime: {steady_state_error_requirement}",
            f"β = {beta}"
        )
        
        # Compensador lag: C(s) = β * (T*s + 1) / (β*T*s + 1)
        T = sp.Symbol('T', real=True, positive=True)
        
        lag_num = beta * (T * s + 1)
        lag_den = beta * T * s + 1
        
        compensator = SymbolicTransferFunction(lag_num, lag_den, s)
        compensator.history.steps = self.history.steps.copy()
        
        return compensator


class StateSpaceController:
    """Design de controladores no espaço de estados"""
    
    def __init__(self):
        self.history = OperationHistory()
    
    def pole_placement(self, A: sp.Matrix, B: sp.Matrix, 
                      desired_poles: List[sp.Expr]) -> sp.Matrix:
        """
        Alocação de polos por realimentação de estados
        
        Args:
            A: Matriz de estados
            B: Matriz de entrada
            desired_poles: Polos desejados
            
        Returns:
            sp.Matrix: Matriz de ganho K
        """
        self.history.add_step(
            "POLE_PLACEMENT",
            "Iniciando alocação de polos",
            f"A: {A}, B: {B}",
            f"Polos desejados: {desired_poles}"
        )
        
        try:
            s = sp.Symbol('s', complex=True)
            n = A.rows
            
            # Polinômio característico desejado
            desired_char_poly = sp.prod([s - pole for pole in desired_poles])
            
            # Verifica controlabilidade
            controllability_matrix = self._controllability_matrix(A, B)
            
            if controllability_matrix.det() == 0:
                self.history.add_step(
                    "ERRO_CONTROLABILIDADE",
                    "Sistema não é controlável",
                    str(controllability_matrix),
                    "det(C) = 0"
                )
                return None
            
            # Usa fórmula de Ackermann para cálculo de K
            # K = [0 0 ... 1] * inv(C) * φ(A)
            # onde φ(A) é A substituído no polinômio característico desejado
            
            e_n = sp.zeros(1, n)
            e_n[n-1] = 1
            
            # Calcula φ(A)
            phi_A = self._evaluate_polynomial_at_matrix(desired_char_poly, A, s)
            
            K = e_n * controllability_matrix.inv() * phi_A
            
            self.history.add_step(
                "RESULTADO_POLE_PLACEMENT",
                "Ganho K calculado",
                f"K = {K}",
                f"Polos de malha fechada: {desired_poles}"
            )
            
            return K
            
        except Exception as e:
            self.history.add_step(
                "ERRO_POLE_PLACEMENT",
                f"Erro no cálculo: {str(e)}",
                f"A: {A}, B: {B}",
                None
            )
            return None
    
    def lqr_design(self, A: sp.Matrix, B: sp.Matrix, 
                  Q: sp.Matrix, R: sp.Matrix) -> dict:
        """
        Design LQR (Linear Quadratic Regulator)
        
        Args:
            A: Matriz de estados
            B: Matriz de entrada
            Q: Matriz de peso dos estados
            R: Matriz de peso do controle
            
        Returns:
            dict: Resultado do design LQR
        """
        self.history.add_step(
            "LQR_DESIGN",
            "Iniciando design LQR",
            f"A: {A}, B: {B}",
            f"Q: {Q}, R: {R}"
        )
        
        try:
            # Para solução analítica da equação de Riccati seria necessário
            # resolver: A^T*P + P*A - P*B*R^(-1)*B^T*P + Q = 0
            # Aqui fornecemos a estrutura simbólica
            
            P = sp.MatrixSymbol('P', A.rows, A.cols)
            riccati_eq = A.T * P + P * A - P * B * R.inv() * B.T * P + Q
            
            # Ganho ótimo: K = R^(-1) * B^T * P
            K_optimal = R.inv() * B.T * P
            
            result = {
                'riccati_equation': riccati_eq,
                'optimal_gain': K_optimal,
                'cost_matrices': {'Q': Q, 'R': R},
                'note': 'Resolva a equação de Riccati para obter P e depois K'
            }
            
            self.history.add_step(
                "RESULTADO_LQR",
                "Estrutura LQR definida",
                f"Equação de Riccati: {riccati_eq}",
                f"Ganho ótimo: K = {K_optimal}"
            )
            
            return result
            
        except Exception as e:
            self.history.add_step(
                "ERRO_LQR",
                f"Erro no design: {str(e)}",
                f"A: {A}, B: {B}",
                None
            )
            return {'error': str(e)}
    
    def _controllability_matrix(self, A: sp.Matrix, B: sp.Matrix) -> sp.Matrix:
        """Calcula matriz de controlabilidade"""
        n = A.rows
        C = B
        
        for i in range(1, n):
            C = C.row_join(A**i * B)
        
        return C
    
    def _evaluate_polynomial_at_matrix(self, poly: sp.Expr, 
                                     matrix: sp.Matrix, 
                                     variable: sp.Symbol) -> sp.Matrix:
        """Avalia polinômio em uma matriz"""
        # Expande o polinômio
        expanded = sp.expand(poly)
        
        # Substitui a variável pela matriz
        result = sp.zeros(matrix.rows, matrix.cols)
        
        # Para cada termo do polinômio
        terms = sp.Add.make_args(expanded)
        for term in terms:
            coeff = term.as_coeff_mul(variable)[0]
            power = sp.degree(term, variable)
            
            if power == 0:
                result += coeff * sp.eye(matrix.rows)
            else:
                result += coeff * (matrix ** power)
        
        return result


class ObserverDesign:
    """Design de observadores"""
    
    def __init__(self):
        self.history = OperationHistory()
    
    def luenberger_observer(self, A: sp.Matrix, C: sp.Matrix,
                          desired_poles: List[sp.Expr]) -> sp.Matrix:
        """
        Design de observador de Luenberger
        
        Args:
            A: Matriz de estados
            C: Matriz de saída
            desired_poles: Polos desejados do observador
            
        Returns:
            sp.Matrix: Matriz de ganho L do observador
        """
        self.history.add_step(
            "LUENBERGER_OBSERVER",
            "Projetando observador de Luenberger",
            f"A: {A}, C: {C}",
            f"Polos desejados: {desired_poles}"
        )
        
        try:
            # Verifica observabilidade
            observability_matrix = self._observability_matrix(A, C)
            
            if observability_matrix.det() == 0:
                self.history.add_step(
                    "ERRO_OBSERVABILIDADE",
                    "Sistema não é observável",
                    str(observability_matrix),
                    "det(O) = 0"
                )
                return None
            
            # Design dual ao problema de alocação de polos
            # L^T = pole_placement(A^T, C^T, desired_poles)
            controller = StateSpaceController()
            L_transpose = controller.pole_placement(A.T, C.T, desired_poles)
            
            if L_transpose is not None:
                L = L_transpose.T
                
                self.history.add_step(
                    "RESULTADO_OBSERVER",
                    "Observador projetado",
                    f"L = {L}",
                    f"Polos do observador: {desired_poles}"
                )
                
                return L
            else:
                return None
                
        except Exception as e:
            self.history.add_step(
                "ERRO_OBSERVER",
                f"Erro no projeto: {str(e)}",
                f"A: {A}, C: {C}",
                None
            )
            return None
    
    def _observability_matrix(self, A: sp.Matrix, C: sp.Matrix) -> sp.Matrix:
        """Calcula matriz de observabilidade"""
        n = A.rows
        O = C
        
        for i in range(1, n):
            O = O.col_join(C * (A**i))
        
        return O
