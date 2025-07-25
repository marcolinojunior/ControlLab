"""
Módulo de visualização simbólica para sistemas de controle
"""

import sympy as sp
from typing import Dict, List, Union, Tuple, Optional
from .symbolic_tf import SymbolicTransferFunction
from .symbolic_ss import SymbolicStateSpace
from .history import OperationHistory


class SymbolicPlotter:
    """Classe para geração de gráficos simbólicos"""
    
    def __init__(self):
        self.history = OperationHistory()
    
    def generate_bode_expressions(self, transfer_function: SymbolicTransferFunction) -> dict:
        """
        Gera expressões simbólicas para diagrama de Bode
        
        Args:
            transfer_function: Função de transferência
            
        Returns:
            dict: Expressões para magnitude e fase
        """
        self.history.add_step(
            "BODE_SIMBÓLICO",
            "Gerando expressões para Bode",
            str(transfer_function),
            "Calculando magnitude e fase..."
        )
        
        try:
            # Substitui s por jω
            omega = sp.Symbol('omega', real=True, positive=True)
            s = transfer_function.variable
            jw_expr = transfer_function.numerator / transfer_function.denominator
            jw_expr = jw_expr.subs(s, sp.I * omega)
            
            # Magnitude em dB
            magnitude_linear = sp.Abs(jw_expr)
            magnitude_db = 20 * sp.log(magnitude_linear, 10)
            
            # Fase em graus
            phase_rad = sp.arg(jw_expr)
            phase_deg = phase_rad * 180 / sp.pi
            
            # Simplifica as expressões
            magnitude_db = sp.simplify(magnitude_db)
            phase_deg = sp.simplify(phase_deg)
            
            result = {
                'magnitude_db': magnitude_db,
                'phase_deg': phase_deg,
                'magnitude_linear': magnitude_linear,
                'phase_rad': phase_rad,
                'frequency_var': omega,
                'complex_expression': jw_expr
            }
            
            self.history.add_step(
                "RESULTADO_BODE_SIMBÓLICO",
                "Expressões de Bode geradas",
                f"|G(jω)| = {magnitude_linear}",
                f"∠G(jω) = {phase_rad} rad = {phase_deg}°"
            )
            
            return result
            
        except Exception as e:
            self.history.add_step(
                "ERRO_BODE_SIMBÓLICO",
                f"Erro: {str(e)}",
                str(transfer_function),
                None
            )
            return {'error': str(e)}
    
    def generate_nyquist_expression(self, transfer_function: SymbolicTransferFunction) -> dict:
        """
        Gera expressão simbólica para diagrama de Nyquist
        
        Args:
            transfer_function: Função de transferência
            
        Returns:
            dict: Expressões para parte real e imaginária
        """
        self.history.add_step(
            "NYQUIST_SIMBÓLICO",
            "Gerando expressão para Nyquist",
            str(transfer_function),
            "Separando parte real e imaginária..."
        )
        
        try:
            # Substitui s por jω
            omega = sp.Symbol('omega', real=True)
            s = transfer_function.variable
            jw_expr = transfer_function.numerator / transfer_function.denominator
            jw_expr = jw_expr.subs(s, sp.I * omega)
            
            # Separa parte real e imaginária
            real_part = sp.re(jw_expr)
            imag_part = sp.im(jw_expr)
            
            # Simplifica
            real_part = sp.simplify(real_part)
            imag_part = sp.simplify(imag_part)
            
            result = {
                'real_part': real_part,
                'imaginary_part': imag_part,
                'complex_expression': jw_expr,
                'frequency_var': omega,
                'magnitude': sp.sqrt(real_part**2 + imag_part**2),
                'phase': sp.atan2(imag_part, real_part)
            }
            
            self.history.add_step(
                "RESULTADO_NYQUIST_SIMBÓLICO",
                "Expressão de Nyquist gerada",
                f"Re[G(jω)] = {real_part}",
                f"Im[G(jω)] = {imag_part}"
            )
            
            return result
            
        except Exception as e:
            self.history.add_step(
                "ERRO_NYQUIST_SIMBÓLICO",
                f"Erro: {str(e)}",
                str(transfer_function),
                None
            )
            return {'error': str(e)}
    
    def generate_root_locus_equations(self, open_loop_tf: SymbolicTransferFunction,
                                    gain_symbol: sp.Symbol = None) -> dict:
        """
        Gera equações para lugar das raízes
        
        Args:
            open_loop_tf: Função de transferência de malha aberta
            gain_symbol: Símbolo do ganho (default: K)
            
        Returns:
            dict: Equações e informações do lugar das raízes
        """
        if gain_symbol is None:
            gain_symbol = sp.Symbol('K', real=True, positive=True)
        
        self.history.add_step(
            "ROOT_LOCUS_SIMBÓLICO",
            "Gerando equações do lugar das raízes",
            str(open_loop_tf),
            f"Ganho variável: {gain_symbol}"
        )
        
        try:
            s = open_loop_tf.variable
            
            # Equação característica: 1 + K*G(s)*H(s) = 0
            char_equation = open_loop_tf.denominator + gain_symbol * open_loop_tf.numerator
            
            # Polos de malha aberta (raízes do denominador)
            open_poles = sp.solve(open_loop_tf.denominator, s)
            
            # Zeros de malha aberta (raízes do numerador)
            open_zeros = sp.solve(open_loop_tf.numerator, s)
            
            # Número de assintotas
            n_poles = len(open_poles)
            n_zeros = len(open_zeros)
            n_asymptotes = n_poles - n_zeros
            
            result = {
                'characteristic_equation': char_equation,
                'gain_symbol': gain_symbol,
                'open_loop_poles': open_poles,
                'open_loop_zeros': open_zeros,
                'num_asymptotes': n_asymptotes,
                'variable': s
            }
            
            # Centroide das assintotas
            if n_asymptotes > 0:
                sum_poles = sum(open_poles) if open_poles else 0
                sum_zeros = sum(open_zeros) if open_zeros else 0
                centroid = (sum_poles - sum_zeros) / n_asymptotes
                result['asymptote_centroid'] = centroid
                
                # Ângulos das assintotas
                asymptote_angles = [(2*k + 1) * sp.pi / n_asymptotes 
                                  for k in range(n_asymptotes)]
                result['asymptote_angles'] = asymptote_angles
                result['asymptote_angles_degrees'] = [angle * 180 / sp.pi 
                                                    for angle in asymptote_angles]
            
            # Pontos de breakaway/break-in (onde dK/ds = 0)
            if n_asymptotes > 0:
                # dK/ds = 0 onde K = -1/G(s)
                K_expr = -open_loop_tf.denominator / open_loop_tf.numerator
                dK_ds = sp.diff(K_expr, s)
                breakaway_equation = sp.Eq(dK_ds, 0)
                result['breakaway_equation'] = breakaway_equation
            
            self.history.add_step(
                "RESULTADO_ROOT_LOCUS_SIMBÓLICO",
                "Equações do lugar das raízes geradas",
                f"Equação característica: {char_equation}",
                f"Polos: {open_poles}, Zeros: {open_zeros}"
            )
            
            return result
            
        except Exception as e:
            self.history.add_step(
                "ERRO_ROOT_LOCUS_SIMBÓLICO",
                f"Erro: {str(e)}",
                str(open_loop_tf),
                None
            )
            return {'error': str(e)}
    
    def generate_step_response_expression(self, transfer_function: SymbolicTransferFunction) -> dict:
        """
        Gera expressão simbólica para resposta ao degrau
        
        Args:
            transfer_function: Função de transferência
            
        Returns:
            dict: Expressão da resposta ao degrau
        """
        self.history.add_step(
            "RESPOSTA_DEGRAU_SIMBÓLICA",
            "Gerando expressão da resposta ao degrau",
            str(transfer_function),
            "Aplicando transformada inversa de Laplace..."
        )
        
        try:
            s = transfer_function.variable
            t = sp.Symbol('t', real=True, positive=True)
            
            # Resposta ao degrau: Y(s) = G(s) * (1/s)
            step_response_s = (transfer_function.numerator / transfer_function.denominator) / s
            
            # Aplica transformada inversa de Laplace
            try:
                step_response_t = sp.inverse_laplace_transform(step_response_s, s, t)
            except:
                # Se falhar, usa frações parciais
                partial_fractions = sp.apart(step_response_s, s)
                step_response_t = sp.inverse_laplace_transform(partial_fractions, s, t)
            
            result = {
                'step_response_s_domain': step_response_s,
                'step_response_t_domain': step_response_t,
                'time_variable': t,
                's_variable': s
            }
            
            # Calcula valor final (se existe)
            try:
                final_value = sp.limit(s * step_response_s, s, 0)
                result['final_value'] = final_value
            except:
                result['final_value'] = 'indefinido'
            
            # Calcula valor inicial
            try:
                initial_value = sp.limit(s * step_response_s, s, sp.oo)
                result['initial_value'] = initial_value
            except:
                result['initial_value'] = 0
            
            self.history.add_step(
                "RESULTADO_RESPOSTA_DEGRAU",
                "Resposta ao degrau calculada",
                f"Y(s) = {step_response_s}",
                f"y(t) = {step_response_t}"
            )
            
            return result
            
        except Exception as e:
            self.history.add_step(
                "ERRO_RESPOSTA_DEGRAU",
                f"Erro: {str(e)}",
                str(transfer_function),
                None
            )
            return {'error': str(e)}
    
    def generate_impulse_response_expression(self, transfer_function: SymbolicTransferFunction) -> dict:
        """
        Gera expressão simbólica para resposta ao impulso
        
        Args:
            transfer_function: Função de transferência
            
        Returns:
            dict: Expressão da resposta ao impulso
        """
        self.history.add_step(
            "RESPOSTA_IMPULSO_SIMBÓLICA",
            "Gerando expressão da resposta ao impulso",
            str(transfer_function),
            "Aplicando transformada inversa de Laplace..."
        )
        
        try:
            s = transfer_function.variable
            t = sp.Symbol('t', real=True, positive=True)
            
            # Resposta ao impulso: y(t) = L⁻¹[G(s)]
            impulse_response_s = transfer_function.numerator / transfer_function.denominator
            
            # Aplica transformada inversa de Laplace
            try:
                impulse_response_t = sp.inverse_laplace_transform(impulse_response_s, s, t)
            except:
                # Se falhar, usa frações parciais
                partial_fractions = sp.apart(impulse_response_s, s)
                impulse_response_t = sp.inverse_laplace_transform(partial_fractions, s, t)
            
            result = {
                'impulse_response_s_domain': impulse_response_s,
                'impulse_response_t_domain': impulse_response_t,
                'time_variable': t,
                's_variable': s
            }
            
            self.history.add_step(
                "RESULTADO_RESPOSTA_IMPULSO",
                "Resposta ao impulso calculada",
                f"G(s) = {impulse_response_s}",
                f"g(t) = {impulse_response_t}"
            )
            
            return result
            
        except Exception as e:
            self.history.add_step(
                "ERRO_RESPOSTA_IMPULSO",
                f"Erro: {str(e)}",
                str(transfer_function),
                None
            )
            return {'error': str(e)}


class LaTeXGenerator:
    """Classe para geração de código LaTeX para documentação"""
    
    def __init__(self):
        self.history = OperationHistory()
    
    def transfer_function_to_latex(self, tf: SymbolicTransferFunction) -> str:
        """
        Converte função de transferência para LaTeX
        
        Args:
            tf: Função de transferência
            
        Returns:
            str: Código LaTeX
        """
        try:
            # Usa o método to_latex do SymPy
            num_latex = sp.latex(tf.numerator)
            den_latex = sp.latex(tf.denominator)
            var_latex = sp.latex(tf.variable)
            
            latex_code = f"G({var_latex}) = \\frac{{{num_latex}}}{{{den_latex}}}"
            
            self.history.add_step(
                "LATEX_TRANSFER_FUNCTION",
                "Convertido para LaTeX",
                str(tf),
                latex_code
            )
            
            return latex_code
            
        except Exception as e:
            return f"\\text{{Erro na conversão: {str(e)}}}"
    
    def state_space_to_latex(self, ss: SymbolicStateSpace) -> str:
        """
        Converte sistema em espaço de estados para LaTeX
        
        Args:
            ss: Sistema em espaço de estados
            
        Returns:
            str: Código LaTeX
        """
        try:
            A_latex = sp.latex(ss.A)
            B_latex = sp.latex(ss.B)
            C_latex = sp.latex(ss.C)
            D_latex = sp.latex(ss.D)
            
            latex_code = f"""\\begin{{align}}
\\dot{{x}} &= {A_latex} x + {B_latex} u \\\\
y &= {C_latex} x + {D_latex} u
\\end{{align}}"""
            
            self.history.add_step(
                "LATEX_STATE_SPACE",
                "Convertido para LaTeX",
                f"A: {ss.A}, B: {ss.B}, C: {ss.C}, D: {ss.D}",
                latex_code
            )
            
            return latex_code
            
        except Exception as e:
            return f"\\text{{Erro na conversão: {str(e)}}}"
    
    def equation_to_latex(self, equation: sp.Expr, label: str = "") -> str:
        """
        Converte equação para LaTeX
        
        Args:
            equation: Equação simbólica
            label: Rótulo opcional
            
        Returns:
            str: Código LaTeX
        """
        try:
            eq_latex = sp.latex(equation)
            
            if label:
                latex_code = f"\\begin{{equation}}\\label{{{label}}}\n{eq_latex}\n\\end{{equation}}"
            else:
                latex_code = f"$${eq_latex}$$"
            
            return latex_code
            
        except Exception as e:
            return f"\\text{{Erro na conversão: {str(e)}}}"


class BlockDiagramGenerator:
    """Classe para geração de diagramas de blocos em formato texto/ASCII"""
    
    def __init__(self):
        self.history = OperationHistory()
    
    def generate_feedback_diagram(self, forward_tf: SymbolicTransferFunction,
                                 feedback_tf: SymbolicTransferFunction = None) -> str:
        """
        Gera diagrama de blocos para sistema em malha fechada
        
        Args:
            forward_tf: Função de transferência direta
            feedback_tf: Função de transferência de realimentação
            
        Returns:
            str: Diagrama ASCII
        """
        if feedback_tf is None:
            # Realimentação unitária
            diagram = f"""
    R(s) ───┬──►[ {forward_tf} ]──┬───► Y(s)
            │                     │
            └───────[ -1 ]◄───────┘
"""
        else:
            diagram = f"""
    R(s) ───┬──►[ {forward_tf} ]──┬───► Y(s)
            │                     │
            └───[ {feedback_tf} ]◄─┘
"""
        
        self.history.add_step(
            "DIAGRAMA_BLOCOS",
            "Diagrama de blocos gerado",
            f"G(s) = {forward_tf}, H(s) = {feedback_tf}",
            "Diagrama ASCII criado"
        )
        
        return diagram
    
    def generate_series_diagram(self, transfer_functions: List[SymbolicTransferFunction]) -> str:
        """
        Gera diagrama para conexão em série
        
        Args:
            transfer_functions: Lista de funções de transferência
            
        Returns:
            str: Diagrama ASCII
        """
        if not transfer_functions:
            return "Nenhuma função de transferência fornecida"
        
        # Constrói diagrama em série
        diagram = "X(s) "
        
        for i, tf in enumerate(transfer_functions):
            diagram += f"──►[ {tf} ]"
            if i < len(transfer_functions) - 1:
                diagram += "──"
        
        diagram += "──► Y(s)"
        
        self.history.add_step(
            "DIAGRAMA_SÉRIE",
            "Diagrama em série gerado",
            f"Funções: {transfer_functions}",
            "Diagrama ASCII criado"
        )
        
        return diagram
    
    def generate_parallel_diagram(self, transfer_functions: List[SymbolicTransferFunction]) -> str:
        """
        Gera diagrama para conexão em paralelo
        
        Args:
            transfer_functions: Lista de funções de transferência
            
        Returns:
            str: Diagrama ASCII
        """
        if not transfer_functions:
            return "Nenhuma função de transferência fornecida"
        
        # Constrói diagrama em paralelo
        diagram = "       ┌──►[ " + str(transfer_functions[0]) + " ]──┐\n"
        diagram += "X(s) ──┤                           ├── Y(s)\n"
        
        for i in range(1, len(transfer_functions)):
            if i == len(transfer_functions) - 1:
                diagram += "       └──►[ " + str(transfer_functions[i]) + " ]──┘\n"
            else:
                diagram += "       ├──►[ " + str(transfer_functions[i]) + " ]──┤\n"
        
        self.history.add_step(
            "DIAGRAMA_PARALELO",
            "Diagrama em paralelo gerado",
            f"Funções: {transfer_functions}",
            "Diagrama ASCII criado"
        )
        
        return diagram
