from flask import Flask, request, jsonify
from src.controllab.core.symbolic_tf import SymbolicTransferFunction
from src.controllab.numerical.factory import NumericalSystemFactory
from src.controllab.numerical.simulation import simulate_system_response
import sympy as sp
import numpy as np

app = Flask(__name__)

def run_step_response_analysis(tf_symbolic):
    """
    Usa o pipeline completo do ControlLab para converter um sistema simbólico
    para numérico e calcular a sua resposta ao degrau.
    """
    # 1. Usa a nossa "fábrica" para criar um sistema numérico
    factory = NumericalSystemFactory()
    numeric_sys = factory.create_from_symbolic(tf_symbolic)

    # 2. Usa o nosso simulador para obter a resposta
    # (Assumindo que simulate_system_response retorna tempo e resposta)
    time, response = simulate_system_response(numeric_sys, input_type="step")

    # 3. Formata a saída para JSON
    return {"time": time.tolist(), "response": response.tolist()}

@app.route('/api/recalculate_analysis', methods=['POST'])
def recalculate_analysis():
    data = request.get_json()

    system_definition_str = data.get('system_definition')
    param_name_str = data.get('param_name')
    param_value = data.get('param_value')
    analysis_type = data.get('analysis_type')

    if not all([system_definition_str, param_name_str, param_value is not None, analysis_type]):
        return jsonify({"error": "Missing required parameters"}), 400

    try:
        s = sp.symbols('s')
        K = sp.symbols(param_name_str)

        # 1. Converte a string inteira numa única expressão SymPy.
        #    O sympify é inteligente o suficiente para lidar com parênteses e operadores.
        full_expr = sp.sympify(system_definition_str, locals={'s': s, param_name_str: K})

        # 2. Usa o SymPy para extrair o numerador e o denominador de forma segura.
        #    Este é o método à prova de falhas.
        num_expr, den_expr = full_expr.as_numer_den()

        tf = SymbolicTransferFunction(num_expr, den_expr, s)

        # Substitute the parameter
        tf_numeric = tf.substitute_param(K, float(param_value))

        # Run the requested analysis
        if analysis_type == "step_response":
            analysis_result = run_step_response_analysis(tf_numeric)
        else:
            return jsonify({"error": f"Analysis type '{analysis_type}' not supported"}), 400

        return jsonify(analysis_result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
