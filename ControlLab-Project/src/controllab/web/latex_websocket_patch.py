
# 🧮 PATCH: LaTeX Integration for WebSocket Server
# Integra LaTeXGenerator com respostas do servidor WebSocket

# Adicionar após as importações existentes do websocket_server.py:

try:
    from ..core.visualization import LaTeXGenerator
    from ..core.symbolic_tf import SymbolicTransferFunction
    LATEX_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ LaTeX Generator não disponível: {e}")
    LATEX_AVAILABLE = False
    
    # Mock LaTeXGenerator para evitar erros
    class LaTeXGenerator:
        def transfer_function_to_latex(self, tf):
            return f"G(s) = {str(tf)}"


class LaTeXResponseEnhancer:
    """Enhancer para adicionar LaTeX às respostas do WebSocket"""
    
    def __init__(self):
        self.latex_gen = LaTeXGenerator() if LATEX_AVAILABLE else LaTeXGenerator()
        
    def enhance_response_with_latex(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adiciona campos LaTeX à resposta do servidor
        
        Args:
            response: Resposta original do servidor
            
        Returns:
            Dict: Resposta aprimorada com LaTeX
        """
        enhanced_response = response.copy()
        
        # Se há dados de sistema, adiciona LaTeX
        if 'system' in response:
            system_data = response['system']
            
            # LaTeX para função de transferência
            if 'transfer_function' in system_data:
                tf_str = system_data['transfer_function']
                enhanced_response['latex'] = {
                    'transfer_function': self._convert_tf_to_latex(tf_str),
                    'original_expression': tf_str
                }
            
            # LaTeX para polos e zeros
            if 'poles' in system_data:
                poles = system_data['poles']
                enhanced_response['latex']['poles'] = self._format_poles_latex(poles)
                
            if 'zeros' in system_data:
                zeros = system_data['zeros']
                enhanced_response['latex']['zeros'] = self._format_zeros_latex(zeros)
                
            # LaTeX para especificações
            if 'specifications' in response:
                specs = response['specifications']
                enhanced_response['latex']['specifications'] = self._format_specs_latex(specs)
        
        return enhanced_response
    
    def _convert_tf_to_latex(self, tf_string: str) -> str:
        """Converte string de função de transferência para LaTeX"""
        try:
            # Parse da função de transferência
            if '=' in tf_string:
                tf_string = tf_string.split('=')[1].strip()
            
            # Converte para SymPy e depois LaTeX
            import sympy as sp
            s = sp.Symbol('s')
            expr = sp.sympify(tf_string.replace('^', '**'))
            
            if hasattr(expr, 'as_numer_denom'):
                num, den = expr.as_numer_denom()
                num_latex = sp.latex(num)
                den_latex = sp.latex(den)
                return f"G(s) = \\frac{{{num_latex}}}{{{den_latex}}}"
            else:
                return f"G(s) = {sp.latex(expr)}"
                
        except Exception as e:
            # Fallback para LaTeX simples
            return f"G(s) = \\text{{{tf_string}}}"
    
    def _format_poles_latex(self, poles: list) -> str:
        """Formata polos em LaTeX"""
        if not poles:
            return "\\text{Nenhum polo}"
        
        poles_latex = []
        for pole in poles:
            if isinstance(pole, (int, float)):
                poles_latex.append(f"{pole:.3f}")
            else:
                poles_latex.append(str(pole))
        
        return f"p = [{', '.join(poles_latex)}]"
    
    def _format_zeros_latex(self, zeros: list) -> str:
        """Formata zeros em LaTeX"""
        if not zeros:
            return "\\text{Nenhum zero}"
        
        zeros_latex = []
        for zero in zeros:
            if isinstance(zero, (int, float)):
                zeros_latex.append(f"{zero:.3f}")
            else:
                zeros_latex.append(str(zero))
        
        return f"z = [{', '.join(zeros_latex)}]"
    
    def _format_specs_latex(self, specs: dict) -> dict:
        """Formata especificações em LaTeX"""
        latex_specs = {}
        
        if 'overshoot' in specs:
            latex_specs['overshoot'] = f"M_p = {specs['overshoot']}"
        if 'settling_time' in specs:
            latex_specs['settling_time'] = f"t_s = {specs['settling_time']}"
        if 'rise_time' in specs:
            latex_specs['rise_time'] = f"t_r = {specs['rise_time']}"
        if 'steady_state_error' in specs:
            latex_specs['steady_state_error'] = f"e_{ss} = {specs['steady_state_error']}"
            
        return latex_specs
