import pytest
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_recalculate_analysis_endpoint(client):
    """Testa o endpoint /api/recalculate_analysis."""
    # Dados de exemplo para a requisição
    request_data = {
        "system_definition": "K/(s*(s+5))",
        "param_name": "K",
        "param_value": 10.0,
        "analysis_type": "step_response"
    }

    # Envia a requisição POST
    response = client.post('/api/recalculate_analysis', json=request_data)

    # Tenta obter o JSON da resposta
    try:
        response_data = response.get_json()
    except Exception as e:
        # Se falhar, imprime o conteúdo da resposta para depuração
        print(f"Erro ao decodificar JSON: {e}")
        print(f"Conteúdo da Resposta: {response.data.decode('utf-8')}")
        response_data = None

    # Verifica se a resposta foi bem-sucedida (código 200)
    assert response.status_code == 200, f"A API retornou um erro: {response_data.get('error') if response_data else 'N/A'}"

    # Verifica se a resposta é um JSON válido
    assert isinstance(response_data, dict)

    # Verifica se a resposta contém as chaves esperadas para o gráfico
    assert "time" in response_data
    assert "response" in response_data

    # Verifica se os valores são listas (como esperado para dados de gráfico)
    assert isinstance(response_data["time"], list)
    assert isinstance(response_data["response"], list)
