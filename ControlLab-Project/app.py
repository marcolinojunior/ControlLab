from flask import Flask, request, jsonify
from src.controllab.core import SymbolicTransferFunction
import sympy as sp
import numpy as np

# Placeholder for analysis functions
def run_step_response_analysis(tf):
    # In a real implementation, this would call the actual analysis function
    # and return data formatted for plotting.
    # For now, we'll just return a simple dictionary.
    return {"time": np.linspace(0, 10, 100).tolist(), "response": np.random.rand(100).tolist()}

app = Flask(__name__)

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
        # Create symbolic transfer function from string
        s = sp.symbols('s')
        K = sp.symbols(param_name_str)

        # This is a simplified way to create the transfer function from a string.
        # A more robust solution would be needed for complex expressions.
        num, den = system_definition_str.split('/')
        num_expr = sp.sympify(num, locals={'s': s, param_name_str: K})
        den_expr = sp.sympify(den, locals={'s': s, param_name_str: K})

        tf = SymbolicTransferFunction(num_expr, den_expr)

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
