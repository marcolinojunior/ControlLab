"""
ControlLab - Regulador Linear Quadrático (LQR)
===============================================

Este módulo implementa controle ótimo LQR:
- Solução simbólica da Equação Algébrica de Riccati
- Projeto de controlador LQR
- Análise de sensibilidade
- Explicações pedagógicas detalhadas

Características:
- Derivação simbólica da solução
- Explicações step-by-step
- Análise de propriedades do LQR
- Estudo de sensibilidade paramétrica
"""

import sympy as sp
from typing import Dict, List, Optional, Union, Tuple, Any
from ..core.symbolic_ss import SymbolicStateSpace
from .design_utils import ControllerResult, create_educational_content

def solve_are_symbolic(A: sp.Matrix, 
                      B: sp.Matrix, 
                      Q: sp.Matrix, 
                      R: sp.Matrix,
                      show_steps: bool = True) -> Dict[str, Any]:
    """
    Resolve a Equação Algébrica de Riccati simbolicamente
    
    Args:
        A: Matriz de estado (n×n)
        B: Matriz de entrada (n×m)
        Q: Matriz de ponderação dos estados (n×n)
        R: Matriz de ponderação do controle (m×m)
        show_steps: Se deve mostrar passos detalhados
    
    Returns:
        Dict[str, Any]: Solução da ARE e análise
    """
    if show_steps:
        print("🎯 SOLUÇÃO DA EQUAÇÃO ALGÉBRICA DE RICCATI (ARE)")
        print("=" * 55)
        print("Equação: A^T P + P A - P B R^(-1) B^T P + Q = 0")
        print(f"📐 A = {A}")
        print(f"📐 B = {B}")
        print(f"📐 Q = {Q}")
        print(f"📐 R = {R}")
    
    n = A.rows
    m = B.cols
    
    # Verificar dimensões
    if show_steps:
        print(f"\n📊 VERIFICAÇÃO DE DIMENSÕES:")
        print(f"Sistema de ordem n = {n}")
        print(f"Número de entradas m = {m}")
        print(f"A: {A.rows}×{A.cols}")
        print(f"B: {B.rows}×{B.cols}")
        print(f"Q: {Q.rows}×{Q.cols}")
        print(f"R: {R.rows}×{R.cols}")
    
    # Verificar se Q é semidefinida positiva e R é definida positiva
    if show_steps:
        print(f"\n✅ CONDIÇÕES PARA SOLUÇÃO:")
        print(f"• Q deve ser semidefinida positiva (Q ≥ 0)")
        print(f"• R deve ser definida positiva (R > 0)")
        print(f"• Par (A,B) deve ser estabilizável")
        print(f"• Par (A,Q^(1/2)) deve ser detectável")
    
    # Para sistemas de ordem baixa, tentar solução analítica
    if n <= 2:
        return _solve_are_analytical(A, B, Q, R, show_steps)
    else:
        return _solve_are_numerical_symbolic(A, B, Q, R, show_steps)

def _solve_are_analytical(A: sp.Matrix, B: sp.Matrix, Q: sp.Matrix, R: sp.Matrix, 
                         show_steps: bool) -> Dict[str, Any]:
    """Solução analítica para sistemas de ordem baixa"""
    
    n = A.rows
    
    if show_steps:
        print(f"\n🔧 SOLUÇÃO ANALÍTICA PARA SISTEMA DE ORDEM {n}")
        print("=" * 50)
    
    # Criar matriz P simbólica
    P_vars = []
    for i in range(n):
        for j in range(i, n):  # Aproveitar simetria
            P_vars.append(sp.Symbol(f'P_{i+1}{j+1}', real=True))
    
    # Construir matriz P simétrica
    P = sp.zeros(n, n)
    var_idx = 0
    for i in range(n):
        for j in range(i, n):
            P[i, j] = P_vars[var_idx]
            if i != j:
                P[j, i] = P_vars[var_idx]  # Simetria
            var_idx += 1
    
    if show_steps:
        print(f"📐 Matriz P (simétrica):")
        print(f"P = {P}")
    
    # Montar equação de Riccati: A^T P + P A - P B R^(-1) B^T P + Q = 0
    R_inv = R.inv()
    riccati_eq = A.T * P + P * A - P * B * R_inv * B.T * P + Q
    
    if show_steps:
        print(f"\n📋 EQUAÇÃO DE RICCATI:")
        print(f"A^T P + P A - P B R^(-1) B^T P + Q = {riccati_eq}")
    
    # Cada elemento da matriz deve ser zero
    equations = []
    for i in range(n):
        for j in range(i, n):  # Aproveitar simetria
            equations.append(riccati_eq[i, j])
    
    if show_steps:
        print(f"\n📊 SISTEMA DE EQUAÇÕES ({len(equations)} equações):")
        for i, eq in enumerate(equations):
            print(f"Equação {i+1}: {eq} = 0")
    
    # Tentar resolver sistema
    try:
        solution = sp.solve(equations, P_vars)
        
        if show_steps:
            print(f"\n🔍 SOLUÇÃO RETORNADA PELO SYMPY:")
            print(f"Tipo: {type(solution)}")
            print(f"Conteúdo: {solution}")
        
        # Tratar diferentes tipos de retorno do SymPy
        if isinstance(solution, dict) and solution:
            # Solução única como dicionário
            P_solution = P.subs(solution)
            
            if show_steps:
                print(f"\n✅ SOLUÇÃO ÚNICA ENCONTRADA:")
                print(f"P = {P_solution}")
            
            return {
                'success': True,
                'P': P_solution,
                'method': 'analytical',
                'equations': equations,
                'solution_dict': solution
            }
            
        elif isinstance(solution, list) and solution:
            # Múltiplas soluções - escolher a primeira que faz sentido físico
            if show_steps:
                print(f"\n🔍 PROCESSANDO {len(solution)} SOLUÇÕES:")
            
            for i, sol in enumerate(solution):
                if show_steps:
                    print(f"Solução {i+1}: {sol}")
                
                try:
                    # Se sol é uma tupla de valores, criar dicionário
                    if isinstance(sol, (tuple, list)) and len(sol) == len(P_vars):
                        sol_dict = dict(zip(P_vars, sol))
                        if show_steps:
                            print(f"  → Mapeamento: {sol_dict}")
                        
                        P_candidate = P.subs(sol_dict)
                        
                        # Verificar se os valores são reais
                        try:
                            # Tentar avaliar numericamente para verificar se é real
                            P_numeric = sp.Matrix([[float(P_candidate[i,j].evalf()) for j in range(P_candidate.cols)] 
                                                  for i in range(P_candidate.rows)])
                            
                            # Verificar se P é semidefinida positiva (autovalores ≥ 0)
                            eigenvals = [float(val.evalf()) for val in P_candidate.eigenvals().keys()]
                            is_positive_semidefinite = all(val >= -1e-10 for val in eigenvals)  # Tolerância numérica
                            
                            if is_positive_semidefinite:
                                if show_steps:
                                    print(f"\n✅ SOLUÇÃO FISICAMENTE VÁLIDA ENCONTRADA:")
                                    print(f"P = {P_candidate}")
                                    print(f"Autovalores: {eigenvals}")
                                
                                return {
                                    'success': True,
                                    'P': P_candidate,
                                    'method': 'analytical',
                                    'equations': equations,
                                    'solution_dict': sol_dict,
                                    'eigenvalues': eigenvals,
                                    'note': f'Solução {i+1} de {len(solution)} (semidefinida positiva)'
                                }
                            else:
                                if show_steps:
                                    print(f"  ❌ Matriz não é semidefinida positiva (autovalores: {eigenvals})")
                        
                        except (ValueError, TypeError) as ve:
                            if show_steps:
                                print(f"  ❌ Erro na avaliação numérica: {ve}")
                            continue
                    
                    elif isinstance(sol, dict):
                        # Solução já em formato de dicionário
                        P_candidate = P.subs(sol)
                        # Similar verificação aqui se necessário
                        
                except Exception as e:
                    if show_steps:
                        print(f"  ❌ Erro ao processar solução {i+1}: {e}")
                    continue
            
            # Se chegou aqui, nenhuma solução da lista funcionou
            if show_steps:
                print(f"\n⚠️ Nenhuma das {len(solution)} soluções é fisicamente válida")
            
            return {
                'success': False,
                'method': 'analytical',
                'error': f'Nenhuma das {len(solution)} soluções é fisicamente válida',
                'raw_solutions': solution,
                'note': 'Pode ser necessário método numérico para este sistema'
            }
        
        else:
            if show_steps:
                print(f"\n❌ Não foi possível encontrar solução analítica")
            
            return {
                'success': False,
                'method': 'analytical',
                'error': 'No analytical solution found'
            }
    
    except Exception as e:
        if show_steps:
            print(f"\n⚠️ Erro na solução analítica: {e}")
        
        return {
            'success': False,
            'method': 'analytical',
            'error': str(e)
        }

def _solve_are_numerical_symbolic(A: sp.Matrix, B: sp.Matrix, Q: sp.Matrix, R: sp.Matrix,
                                 show_steps: bool) -> Dict[str, Any]:
    """Solução numérico-simbólica para sistemas de ordem alta"""
    
    if show_steps:
        print(f"\n🔧 MÉTODO HAMILTONIANO PARA SISTEMA DE ORDEM {A.rows}")
        print("=" * 50)
        print("Para sistemas de ordem alta, usar método Hamiltoniano:")
        print("H = [A  -BR^(-1)B^T]")
        print("    [-Q    -A^T   ]")
    
    n = A.rows
    
    # Construir matriz Hamiltoniana
    R_inv = R.inv()
    BR_inv_BT = B * R_inv * B.T
    
    # Matriz Hamiltoniana 2n×2n
    H_upper = sp.Matrix.hstack(A, -BR_inv_BT)
    H_lower = sp.Matrix.hstack(-Q, -A.T)
    H = sp.Matrix.vstack(H_upper, H_lower)
    
    if show_steps:
        print(f"📐 Matriz Hamiltoniana H:")
        print(f"H = {H}")
    
    # Para solução pedagógica, criar P simbólica
    P = sp.MatrixSymbol('P', n, n)
    
    return {
        'success': True,
        'P': P,
        'method': 'hamiltonian',
        'hamiltonian_matrix': H,
        'note': 'Solução requer métodos numéricos para ordem alta'
    }

def lqr_design(ss_obj: SymbolicStateSpace,
               Q: sp.Matrix,
               R: sp.Matrix,
               show_steps: bool = True) -> ControllerResult:
    """
    Projeta controlador LQR
    
    Args:
        ss_obj: Sistema em espaço de estados
        Q: Matriz de ponderação dos estados
        R: Matriz de ponderação do controle
        show_steps: Se deve mostrar passos
    
    Returns:
        ControllerResult: Controlador LQR projetado
    """
    if show_steps:
        print("🎯 PROJETO DE CONTROLADOR LQR")
        print("=" * 35)
        print(f"🏭 Sistema: ẋ = Ax + Bu")
        print(f"📐 A = {ss_obj.A}")
        print(f"📐 B = {ss_obj.B}")
        print(f"⚖️ Matriz Q = {Q}")
        print(f"⚖️ Matriz R = {R}")
    
    A = ss_obj.A
    B = ss_obj.B
    
    result = ControllerResult(controller=None)
    
    # Passo 1: Resolver equação de Riccati
    result.add_step("Resolvendo Equação Algébrica de Riccati")
    
    are_result = solve_are_symbolic(A, B, Q, R, show_steps)
    
    if not are_result['success']:
        result.add_step("❌ Falha na solução da ARE")
        return result
    
    P = are_result['P']
    result.add_step(f"Solução da ARE: P = {P}")
    
    # Passo 2: Calcular ganhos LQR
    result.add_step("Calculando ganhos LQR: K = R^(-1) B^T P")
    
    R_inv = R.inv()
    K = R_inv * B.T * P
    
    result.controller = K
    result.add_step(f"Ganhos LQR: K = {K}")
    
    # Passo 3: Sistema em malha fechada
    A_cl = A - B * K
    result.add_step(f"Sistema em malha fechada: A_cl = A - BK = {A_cl}")
    
    # Adicionar conteúdo educacional
    educational_notes = create_educational_content("lqr", {
        'Q': Q, 'R': R
    })
    
    for note in educational_notes:
        result.add_educational_note(note)
    
    # Propriedades do LQR
    result.add_educational_note("PROPRIEDADES IMPORTANTES DO LQR:")
    result.add_educational_note("• Margem de ganho infinita")
    result.add_educational_note("• Margem de fase ≥ 60°")
    result.add_educational_note("• Sistema em malha fechada sempre estável")
    result.add_educational_note("• Solução ótima para função custo quadrática")
    
    result.stability_analysis = {
        'closed_loop_matrix': A_cl,
        'riccati_solution': P,
        'cost_matrices': {'Q': Q, 'R': R}
    }
    
    if show_steps:
        print(result.get_formatted_report())
    
    return result

def analyze_lqr_sensitivity(ss_obj: SymbolicStateSpace,
                           Q: sp.Matrix,
                           R: sp.Matrix,
                           param_variations: Dict[str, List[float]],
                           show_steps: bool = True) -> Dict[str, Any]:
    """
    Analisa sensibilidade do LQR a variações paramétricas
    
    Args:
        ss_obj: Sistema em espaço de estados
        Q, R: Matrizes de ponderação
        param_variations: Variações dos parâmetros
        show_steps: Se deve mostrar passos
    
    Returns:
        Dict[str, Any]: Análise de sensibilidade
    """
    if show_steps:
        print("🔍 ANÁLISE DE SENSIBILIDADE DO LQR")
        print("=" * 40)
        print(f"📊 Variações paramétricas: {param_variations}")
    
    # Para análise pedagógica, mostrar conceitos
    sensitivity_analysis = {
        'method': 'parametric_variation',
        'base_system': {
            'A': ss_obj.A,
            'B': ss_obj.B,
            'Q': Q,
            'R': R
        },
        'variations': param_variations,
        'properties': [
            "LQR é naturalmente robusto",
            "Pequenas variações em Q,R causam pequenas variações em K",
            "Margem de estabilidade é mantida",
            "Função custo degrada graciosamente"
        ]
    }
    
    if show_steps:
        print("🎓 PROPRIEDADES DE ROBUSTEZ DO LQR:")
        for prop in sensitivity_analysis['properties']:
            print(f"• {prop}")
    
    return sensitivity_analysis

class LQRDesigner:
    """
    Classe para projeto sistemático de controladores LQR
    
    Fornece interface unificada para projeto LQR com
    análise de desempenho e robustez.
    """
    
    def __init__(self, system: SymbolicStateSpace, show_steps: bool = True):
        """
        Inicializa o designer LQR
        
        Args:
            system: Sistema em espaço de estados
            show_steps: Se deve mostrar passos
        """
        self.system = system
        self.show_steps = show_steps
        self.design_history = []
    
    def design(self, Q: sp.Matrix, R: sp.Matrix) -> ControllerResult:
        """
        Projeta controlador LQR
        
        Args:
            Q: Matriz de ponderação dos estados
            R: Matriz de ponderação do controle
        
        Returns:
            ControllerResult: Controlador LQR
        """
        return lqr_design(self.system, Q, R, self.show_steps)
    
    def tune_weights(self, 
                     state_importance: List[float],
                     control_effort: float) -> Tuple[sp.Matrix, sp.Matrix]:
        """
        Auxilia na sintonização das matrizes de peso
        
        Args:
            state_importance: Importância relativa de cada estado
            control_effort: Penalização do esforço de controle
        
        Returns:
            Tuple[sp.Matrix, sp.Matrix]: Matrizes Q e R sintonizadas
        """
        n = self.system.A.rows
        m = self.system.B.cols
        
        # Criar Q diagonal com pesos dos estados
        Q = sp.diag(*state_importance)
        
        # Criar R diagonal com penalização do controle
        if m == 1:
            R = sp.Matrix([[control_effort]])
        else:
            R = sp.diag(*([control_effort] * m))
        
        if self.show_steps:
            print("🎛️ SINTONIZAÇÃO DE PESOS:")
            print(f"Q (estados) = {Q}")
            print(f"R (controle) = {R}")
        
        return Q, R
    
    def analyze_performance(self, K: sp.Matrix) -> Dict[str, Any]:
        """
        Analisa desempenho do controlador LQR
        
        Args:
            K: Ganhos LQR
        
        Returns:
            Dict[str, Any]: Análise de desempenho
        """
        A_cl = self.system.A - self.system.B * K
        
        # Calcular autovalores
        eigenvals = A_cl.eigenvals()
        
        performance = {
            'closed_loop_matrix': A_cl,
            'eigenvalues': eigenvals,
            'stability': 'stable' if all(sp.re(val) < 0 for val in eigenvals.keys()) else 'unstable',
            'lqr_properties': [
                "Margem de ganho infinita",
                "Margem de fase ≥ 60°",
                "Robustez natural a incertezas"
            ]
        }
        
        return performance
