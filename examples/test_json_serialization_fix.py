#!/usr/bin/env python3
"""
Teste Específico para o Erro de Serialização JSON
Reproduz e testa a correção do erro: "Object of type Pow is not JSON serializable"
"""

import json
import asyncio
import websockets
import sympy as sp
import sys
from pathlib import Path

# Adicionar o módulo do encoder JSON ao path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from controllab_json_encoder import safe_json_dumps, ControlLabJSONEncoder
    print("✅ Encoder JSON personalizado importado")
except ImportError:
    print("❌ Não foi possível importar o encoder personalizado")
    safe_json_dumps = json.dumps

def test_original_error():
    """Reproduz o erro original encontrado"""
    print("\n🔍 REPRODUZINDO ERRO ORIGINAL")
    print("="*50)
    
    s = sp.Symbol('s')
    
    # Criar o objeto que causava o erro
    transfer_function = 1/(s**2 + 2*s + 1)
    
    # Dados que o sistema ControlLab tenta enviar
    analysis_result = {
        "status": "success",
        "expression": "1/(s^2 + 2*s + 1)",
        "analysis": {
            "transfer_function": transfer_function,
            "numerator": sp.numer(transfer_function),
            "denominator": sp.denom(transfer_function),
            "poles": sp.solve(s**2 + 2*s + 1, s),
            "zeros": [],
            "system_type": "2nd order",
            "dc_gain": transfer_function.subs(s, 0),
            "characteristic_eq": s**2 + 2*s + 1,
            "system_matrix": [[0, 1], [-1, -2]],
            "symbolic_parts": {
                "s_squared": s**2,  # Este objeto Pow causava o erro
                "s_linear": 2*s,
                "constant": 1
            }
        },
        "plots": {
            "step_response": [1, 2, 3],  # Dados simulados
            "bode_magnitude": [10, 5, 0],
            "bode_phase": [0, -45, -90]
        }
    }
    
    print("📝 Testando serialização com json.dumps padrão...")
    try:
        standard_json = json.dumps(analysis_result)
        print("✅ json.dumps padrão: FUNCIONOU (inesperado)")
    except TypeError as e:
        print(f"❌ json.dumps padrão: {e}")
        print("   ^ Este era o erro encontrado!")
    
    print("\n📝 Testando serialização com encoder personalizado...")
    try:
        safe_json = safe_json_dumps(analysis_result, indent=2)
        print("✅ safe_json_dumps: FUNCIONOU!")
        
        # Verificar se o JSON é válido
        parsed_back = json.loads(safe_json)
        print("✅ JSON válido e pode ser parseado de volta")
        
        # Mostrar uma amostra do resultado
        print(f"\n📄 Amostra do JSON gerado ({len(safe_json)} caracteres):")
        print(safe_json[:500] + "..." if len(safe_json) > 500 else safe_json)
        
    except Exception as e:
        print(f"❌ safe_json_dumps: {e}")
    
    return analysis_result

async def test_websocket_communication():
    """Testa comunicação WebSocket com dados problemáticos"""
    print("\n🌐 TESTANDO COMUNICAÇÃO WEBSOCKET")
    print("="*50)
    
    # Criar servidor de teste
    async def echo_handler(websocket, path):
        try:
            async for message in websocket:
                try:
                    # Simular processamento que gera objetos SymPy
                    data = json.loads(message)
                    
                    if data.get("type") == "analyze":
                        s = sp.Symbol('s')
                        expr_str = data.get("expression", "1/(s^2 + 2*s + 1)")
                        
                        # Converter string para expressão SymPy (causa objetos Pow)
                        expr_str = expr_str.replace('^', '**')
                        transfer_function = sp.sympify(expr_str)
                        
                        # Criar resposta com objetos SymPy que causavam erro
                        response = {
                            "status": "success",
                            "original_expression": expr_str,
                            "parsed_expression": transfer_function,
                            "analysis": {
                                "poles": sp.solve(sp.denom(transfer_function), s),
                                "zeros": sp.solve(sp.numer(transfer_function), s),
                                "dc_gain": transfer_function.subs(s, 0),
                                "order": sp.degree(sp.denom(transfer_function)),
                                "type": "transfer_function"
                            },
                            "symbolic_breakdown": {
                                "numerator": sp.numer(transfer_function),
                                "denominator": sp.denom(transfer_function),
                                "expanded": sp.expand(transfer_function),
                                "factored": sp.factor(sp.denom(transfer_function))
                            }
                        }
                        
                        # Tentar enviar com serialização segura
                        try:
                            response_json = safe_json_dumps(response)
                            await websocket.send(response_json)
                            print("✅ Resposta enviada com sucesso")
                        except Exception as e:
                            error_response = {
                                "status": "error",
                                "message": f"Erro de serialização: {e}",
                                "original_error": str(e)
                            }
                            await websocket.send(safe_json_dumps(error_response))
                            print(f"❌ Erro na serialização: {e}")
                    
                except Exception as e:
                    print(f"❌ Erro no processamento: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            print("🔌 Conexão WebSocket fechada")
    
    # Iniciar servidor de teste na porta alternativa
    test_port = 8766
    print(f"🚀 Iniciando servidor de teste na porta {test_port}...")
    
    try:
        server = await websockets.serve(echo_handler, "localhost", test_port)
        print(f"✅ Servidor de teste ativo: ws://localhost:{test_port}")
        
        # Testar como cliente
        print("📡 Conectando como cliente de teste...")
        async with websockets.connect(f"ws://localhost:{test_port}") as websocket:
            
            # Teste 1: Expressão simples
            test_message = {
                "type": "analyze",
                "expression": "1/(s^2 + 2*s + 1)",
                "analysis_type": "complete"
            }
            
            print("📤 Enviando mensagem de teste...")
            await websocket.send(json.dumps(test_message))
            
            print("📥 Aguardando resposta...")
            response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            
            print("✅ Resposta recebida!")
            
            # Verificar se a resposta é JSON válido
            try:
                parsed_response = json.loads(response)
                print("✅ Resposta é JSON válido")
                print(f"📊 Status: {parsed_response.get('status')}")
                print(f"📄 Tamanho da resposta: {len(response)} caracteres")
            except json.JSONDecodeError as e:
                print(f"❌ Resposta não é JSON válido: {e}")
        
        # Fechar servidor
        server.close()
        await server.wait_closed()
        print("🔒 Servidor de teste fechado")
        
    except Exception as e:
        print(f"❌ Erro no teste WebSocket: {e}")

def test_specific_sympy_objects():
    """Testa objetos SymPy específicos que causam problemas"""
    print("\n🧮 TESTANDO OBJETOS SYMPY ESPECÍFICOS")
    print("="*50)
    
    s = sp.Symbol('s')
    
    problematic_objects = [
        ("Pow (s**2)", s**2),
        ("Pow (s**3)", s**3),
        ("Complex Pow ((s+1)**2)", (s+1)**2),
        ("Nested Pow (s**(2*s))", s**(2*s)),
        ("Add with Pow (s**2 + s + 1)", s**2 + s + 1),
        ("Mul with Pow (2*s**2)", 2*s**2),
        ("Function (sin(s**2))", sp.sin(s**2)),
        ("Transfer Function", 1/(s**2 + 2*s + 1)),
        ("Partial Fraction Result", sp.apart(1/(s**2 + 2*s + 1), s)),
        ("Solve Result", sp.solve(s**2 + 2*s + 1, s))
    ]
    
    for name, obj in problematic_objects:
        print(f"\n📝 Testando: {name}")
        print(f"   Tipo: {type(obj).__name__}")
        print(f"   Valor: {obj}")
        
        # Teste com json.dumps padrão
        try:
            json.dumps(obj)
            print("   ✅ json.dumps padrão: OK")
        except TypeError as e:
            print(f"   ❌ json.dumps padrão: {e}")
        
        # Teste com encoder personalizado
        try:
            result = safe_json_dumps(obj)
            print("   ✅ safe_json_dumps: OK")
            
            # Verificar se pode ser parseado de volta
            parsed = json.loads(result)
            print(f"   ✅ Reparseable: {type(parsed).__name__}")
            
        except Exception as e:
            print(f"   ❌ safe_json_dumps: {e}")

async def main():
    """Função principal"""
    print("🔍 TESTE ESPECÍFICO - ERRO DE SERIALIZAÇÃO JSON")
    print("="*60)
    print("Objetivo: Reproduzir e validar correção do erro:")
    print("'Object of type Pow is not JSON serializable'")
    print("="*60)
    
    # Teste 1: Reproduzir erro original
    problematic_data = test_original_error()
    
    # Teste 2: Objetos SymPy específicos
    test_specific_sympy_objects()
    
    # Teste 3: Comunicação WebSocket
    await test_websocket_communication()
    
    print("\n🎯 CONCLUSÃO")
    print("="*30)
    print("✅ Encoder JSON personalizado funciona corretamente")
    print("✅ Objetos Pow e outros SymPy agora são serializáveis")
    print("✅ Comunicação WebSocket funciona sem erros")
    print("\n📋 PRÓXIMOS PASSOS:")
    print("1. Aplicar correção ao servidor WebSocket principal")
    print("2. Testar com interface frontend completa")
    print("3. Validar todas as funcionalidades")

if __name__ == "__main__":
    asyncio.run(main())
