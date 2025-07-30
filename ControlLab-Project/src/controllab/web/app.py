# src/controllab/web/app.py

import sys
import os
import sympy as sp
from flask import Flask, request, jsonify, render_template

# --- Configuração de Path ---
# Adiciona o diretório 'src' ao path para que possamos importar 'controllab'
# Isso assume que o app.py está em src/controllab/web/
project_src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_src_path not in sys.path:
    sys.path.insert(0, project_src_path)
# -----------------------------

# --- Importações do seu Módulo ControlLab ---
# Importamos as classes e funções que o frontend irá utilizar
from controllab.core import SymbolicTransferFunction
from controllab.analysis.routh_hurwitz import RouthHurwitzAnalyzer
from controllab.analysis.root_locus import RootLocusAnalyzer
from controllab.analysis.frequency_response import FrequencyAnalyzer
from controllab.modeling.partial_fractions import PartialFractionExpander
# Adicione mais importações conforme necessário...
# ---------------------------------------------

app = Flask(__name__, template_folder='../templates') # Aponta para a pasta de templates

# Variável simbólica global
s = sp.symbols('s')

def parse_latex_to_sympy(latex_str: str) -> sp.Expr:
    """
    Converte uma string LaTeX em uma expressão SymPy.
    NOTA: Esta é uma simplificação. Uma implementação robusta precisaria
    de um parser mais avançado (ex: antlr) ou uma biblioteca como `latex2sympy`.
    Por enquanto, faremos substituições simples.
    """
    # Substituições comuns de LaTeX para formato SymPy
    replacements = {
        '\\frac': '(',
        '{': '',
        '}': ')',
        '\\left': '',
        '\\right': '',
        '^': '**',
        # Adicione mais regras conforme necessário
    }
    for old, new in replacements.items():
        latex_str = latex_str.replace(old, new)
    
    # Adicionar parênteses ausentes se a conversão do \frac falhar
    if latex_str.count('(') > latex_str.count(')'):
        latex_str += ')' * (latex_str.count('(') - latex_str.count(')'))

    try:
        # sympify tenta converter a string em uma expressão sympy
        return sp.sympify(latex_str, locals={'s': s})
    except (sp.SympifyError, SyntaxError) as e:
        print(f"Erro ao fazer o parse da expressão: {latex_str}. Erro: {e}")
        raise ValueError("A expressão matemática não pôde ser interpretada.")


@app.route('/')
def index():
    """Renderiza a página HTML principal."""
    # O Flask irá procurar por 'controllab_interface.html' na pasta 'templates'
    return render_template('controllab_interface.html')

@app.route('/calculate', methods=['POST'])
def calculate():
    """
    Endpoint principal que recebe o problema, chama o módulo ControlLab
    e retorna a solução passo a passo.
    """
    data = request.json
    expression_str = data.get('expression')
    analysis_type = data.get('analysis')

    if not expression_str or not analysis_type:
        return jsonify({'error': 'Expressão ou tipo de análise ausente.'}), 400

    try:
        # Converte a expressão de string para um objeto simbólico
        # Futuramente, isso viria do parser LaTeX
        sym_expr = sp.sympify(expression_str, locals={'s': s})
        
        # Cria a Função de Transferência simbólica
        tf = SymbolicTransferFunction(sp.fraction(sym_expr)[0], sp.fraction(sym_expr)[1], s)

        # Chama a análise apropriada baseada na seleção do usuário
        steps = []
        if analysis_type == 'routh_hurwitz':
            analyzer = RouthHurwitzAnalyzer()
            poly = tf.denominator # Analisa o polinômio característico
            result = analyzer.analyze_stability(analyzer.build_routh_array(poly, show_steps=True), show_steps=True)
            steps.append({
                'title': 'Análise de Estabilidade por Routh-Hurwitz',
                'explanation': result.history.get_formatted_report(),
                'latex_result': f"\\text{{Estável: {result.is_stable}, Trocas de Sinal: {result.unstable_poles_count}}}"
            })

        elif analysis_type == 'root_locus':
            analyzer = RootLocusAnalyzer()
            features = analyzer.get_locus_features(tf, show_steps=True)
            steps.append({
                'title': 'Análise pelo Lugar Geométrico das Raízes (LGR)',
                'explanation': features.history.get_formatted_report(),
                'latex_result': f"\\text{{Assíntotas: {len(features.asymptotes['angles'])}, Centroide: {features.asymptotes['centroid']}}}"
            })
            # AQUI VOCÊ ADICIONARIA O CÓDIGO PARA GERAR A IMAGEM DO PLOT

        elif analysis_type == 'partial_fractions':
            expander = PartialFractionExpander()
            # Usando o método pedagógico que você pode criar
            result = expander.expand_with_explanation(sym_expr, s) # Supondo que este método exista
            steps.append({
                'title': 'Expansão em Frações Parciais',
                'explanation': result['explanation'], # Supondo que ele retorne um dict com explicação
                'latex_result': sp.latex(result['expression'])
            })
            
        else:
            return jsonify({'error': 'Tipo de análise não suportado.'}), 400

        return jsonify({'steps': steps})

    except Exception as e:
        return jsonify({'error': f'Ocorreu um erro: {str(e)}'}), 500

if __name__ == '__main__':
    # Garante que estamos executando no modo de desenvolvimento para ter recarregamento automático
    app.run(debug=True, port=5000)
