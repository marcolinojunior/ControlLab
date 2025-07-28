# DENTRO DE: app.py
from flask import Flask, request, jsonify
from controllab.core.symbolic_tf import SymbolicTransferFunction
from controllab.numerical.factory import NumericalSystemFactory # Supondo que esta seja a ponte para simulação
import sympy as sp

app = Flask(__name__)

@app.route('/api/recalculate_analysis', methods=['POST'])
def recalculate_analysis():
    data = request.get_json()

    # 1. Validação Robusta do Input
    required_keys = ['system_definition', 'param_name', 'param_value', 'analysis_type']
    if not all(key in data for key in required_keys):
        return jsonify({"error": "Parâmetros em falta na requisição."}), 400

    try:
        # 2. Parsing e Criação do Objeto Simbólico
        s = sp.symbols('s')
        param_symbol = sp.symbols(data['param_name'])

        # 1. Converte a string inteira numa única expressão SymPy.
        #    O sympify é inteligente o suficiente para entender os parênteses.
        full_expr = sp.sympify(data['system_definition'], locals={'s': s, 'K': param_symbol})

        # 2. Usa o SymPy para extrair o numerador e o denominador de forma segura.
        #    O método .as_numer_den() é a forma canónica de fazer isto.
        num_expr, den_expr = full_expr.as_numer_den()

        symbolic_tf = SymbolicTransferFunction(num_expr, den_expr, s)

        # 3. Chamada à nossa nova função do ControlLab (Task VI-01)
        tf_substituted = symbolic_tf.substitute_param(param_symbol, float(data['param_value']))

        # 4. Execução da Análise (usando a ponte numérica)
        factory = NumericalSystemFactory()
        numeric_sys = factory.create_from_symbolic(tf_substituted)

        if data['analysis_type'] == "step_response":
            # Supondo que o seu sistema numérico tenha um método de simulação
            time, response = numeric_sys.step_response()
            analysis_result = {"time": time.tolist(), "response": response.tolist()}
        else:
            return jsonify({"error": f"Tipo de análise '{data['analysis_type']}' não suportado."}), 400

        return jsonify(analysis_result)

    except Exception as e:
        # O nosso sistema de diagnóstico inteligente pode ser útil aqui no futuro
        return jsonify({"error": f"Ocorreu um erro no servidor: {e}"}), 500
