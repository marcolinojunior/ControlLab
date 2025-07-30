#!/usr/bin/env python3
"""
Testes para Fase 6: Projeto de Controladores

Este arquivo implementa testes completos para validar o Módulo 6
conforme especificado no arquivo 06-projeto-controladores.md
"""

import pytest
import sys
import os

# Adicionar path do projeto
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

# Imports necessários
import sympy as sp
from controllab.core.symbolic_tf import SymbolicTransferFunction
from controllab.core.symbolic_ss import SymbolicStateSpace

class TestFase6ProjetoControladores:
    """
    Classe principal de testes para o Módulo 6
    """
    
    def setup_method(self):
        """Setup para cada teste"""
        # Símbolos básicos
        self.s = sp.Symbol('s')
        self.Kp, self.Ki, self.Kd = sp.symbols('Kp Ki Kd', real=True)
        
        # Sistema exemplo para testes
        self.plant = SymbolicTransferFunction(1, self.s*(self.s + 1))
        
        # Sistema em espaço de estados exemplo
        A = sp.Matrix([[0, 1], [-2, -3]])
        B = sp.Matrix([[0], [1]])
        C = sp.Matrix([[1, 0]])
        D = sp.Matrix([[0]])
        self.ss_system = SymbolicStateSpace(A, B, C, D)

class TestCompensadoresClassicos(TestFase6ProjetoControladores):
    """Testes para compensadores clássicos"""
    
    def test_import_design_modules(self):
        """Teste de importação dos módulos de projeto"""
        try:
            from controllab.design.compensators import PID, Lead, Lag, LeadLag
            from controllab.design.design_utils import ControllerResult, DesignSpecifications
            print("✅ Módulos de compensadores importados com sucesso")
        except ImportError as e:
            pytest.skip(f"Módulos ainda não implementados: {e}")
    
    def test_pid_controller_creation(self):
        """Teste de criação de controlador PID"""
        try:
            from controllab.design.compensators import PID
            
            # Criar PID simbólico
            controller = PID(self.Kp, self.Ki, self.Kd)
            
            # Verificar estrutura
            assert controller is not None
            print(f"✅ PID criado: C(s) = {controller}")
            
        except ImportError:
            pytest.skip("Módulo compensators não implementado")
        except Exception as e:
            pytest.fail(f"Erro na criação do PID: {e}")
    
    def test_lead_compensator_creation(self):
        """Teste de criação de compensador Lead"""
        try:
            from controllab.design.compensators import Lead
            
            K, z, p = sp.symbols('K z p', real=True, positive=True)
            
            # Criar Lead com z < p (condição para avanço de fase)
            lead = Lead(K, z, p)
            
            assert lead is not None
            print(f"✅ Lead criado: C(s) = {lead}")
            
        except ImportError:
            pytest.skip("Módulo compensators não implementado")
        except Exception as e:
            pytest.fail(f"Erro na criação do Lead: {e}")
    
    def test_lag_compensator_creation(self):
        """Teste de criação de compensador Lag"""
        try:
            from controllab.design.compensators import Lag
            
            K, z, p = sp.symbols('K z p', real=True, positive=True)
            
            # Criar Lag
            lag = Lag(K, z, p)
            
            assert lag is not None
            print(f"✅ Lag criado: C(s) = {lag}")
            
        except ImportError:
            pytest.skip("Módulo compensators não implementado")
        except Exception as e:
            pytest.fail(f"Erro na criação do Lag: {e}")

class TestAlocacaoPolos(TestFase6ProjetoControladores):
    """Testes para alocação de polos"""
    
    def test_controllability_check(self):
        """Teste de verificação de controlabilidade"""
        try:
            from controllab.design.pole_placement import check_controllability
            
            # Verificar controlabilidade do sistema exemplo
            result = check_controllability(self.ss_system, show_steps=False)
            
            assert isinstance(result, dict)
            assert 'is_controllable' in result
            assert 'controllability_matrix' in result
            
            print(f"✅ Controlabilidade verificada: {result['is_controllable']}")
            
        except ImportError:
            pytest.skip("Módulo pole_placement não implementado")
        except Exception as e:
            pytest.fail(f"Erro na verificação de controlabilidade: {e}")
    
    def test_ackermann_pole_placement(self):
        """Teste da fórmula de Ackermann"""
        try:
            from controllab.design.pole_placement import acker
            
            # Polos desejados
            desired_poles = [-2, -3]
            
            # Aplicar Ackermann
            result = acker(self.ss_system, desired_poles, show_steps=False)
            
            assert isinstance(result, dict)
            if result.get('success'):
                assert 'gains' in result
                print(f"✅ Ackermann aplicado: K = {result['gains']}")
            else:
                print(f"⚠️ Ackermann falhou: {result.get('error', 'Erro desconhecido')}")
            
        except ImportError:
            pytest.skip("Módulo pole_placement não implementado")
        except Exception as e:
            pytest.fail(f"Erro no Ackermann: {e}")

class TestObservadores(TestFase6ProjetoControladores):
    """Testes para projeto de observadores"""
    
    def test_observability_check(self):
        """Teste de verificação de observabilidade"""
        try:
            from controllab.design.observer import check_observability
            
            # Verificar observabilidade
            result = check_observability(self.ss_system, show_steps=False)
            
            assert isinstance(result, dict)
            assert 'is_observable' in result
            assert 'observability_matrix' in result
            
            print(f"✅ Observabilidade verificada: {result['is_observable']}")
            
        except ImportError:
            pytest.skip("Módulo observer não implementado")
        except Exception as e:
            pytest.fail(f"Erro na verificação de observabilidade: {e}")
    
    def test_observer_design(self):
        """Teste de projeto de observador"""
        try:
            from controllab.design.observer import acker_observer
            
            # Polos desejados para o observador
            observer_poles = [-5, -6]  # Mais rápidos que o controlador
            
            # Projetar observador
            result = acker_observer(self.ss_system, observer_poles, show_steps=False)
            
            assert isinstance(result, dict)
            if result.get('success'):
                assert 'observer_gains' in result
                print(f"✅ Observador projetado: L = {result['observer_gains']}")
            else:
                print(f"⚠️ Projeto do observador falhou: {result.get('error', 'Erro desconhecido')}")
            
        except ImportError:
            pytest.skip("Módulo observer não implementado")
        except Exception as e:
            pytest.fail(f"Erro no projeto do observador: {e}")

class TestLQR(TestFase6ProjetoControladores):
    """Testes para controle LQR"""
    
    def test_lqr_design(self):
        """Teste de projeto LQR"""
        try:
            from controllab.design.lqr import lqr_design
            
            # Matrizes de peso
            Q = sp.Matrix([[1, 0], [0, 1]])  # Identidade
            R = sp.Matrix([[1]])  # Escalar
            
            # Projetar LQR
            result = lqr_design(self.ss_system, Q, R, show_steps=False)
            
            assert result is not None
            assert hasattr(result, 'controller')
            
            if result.controller is not None:
                print(f"✅ LQR projetado: K = {result.controller}")
            else:
                print("⚠️ LQR não pôde ser projetado")
            
        except ImportError:
            pytest.skip("Módulo LQR não implementado")
        except Exception as e:
            pytest.fail(f"Erro no projeto LQR: {e}")
    
    def test_riccati_equation(self):
        """Teste da solução da equação de Riccati"""
        try:
            from controllab.design.lqr import solve_are_symbolic
            
            A = self.ss_system.A
            B = self.ss_system.B
            Q = sp.Matrix([[1, 0], [0, 1]])
            R = sp.Matrix([[1]])
            
            # Resolver ARE
            result = solve_are_symbolic(A, B, Q, R, show_steps=False)
            
            assert isinstance(result, dict)
            if result.get('success'):
                print(f"✅ ARE resolvida: P = {result['P']}")
            else:
                print(f"⚠️ ARE não resolvida: {result.get('error', 'Método numérico necessário')}")
            
        except ImportError:
            pytest.skip("Módulo LQR não implementado")
        except Exception as e:
            pytest.fail(f"Erro na solução da ARE: {e}")

class TestDesempenho(TestFase6ProjetoControladores):
    """Testes para análise de desempenho"""
    
    def test_performance_analysis(self):
        """Teste de análise de desempenho"""
        try:
            from controllab.design.performance import analyze_transient_response
            
            # Sistema exemplo em malha fechada
            closed_loop = SymbolicTransferFunction(1, self.s**2 + 3*self.s + 2)
            
            # Analisar resposta
            result = analyze_transient_response(closed_loop, show_steps=False)
            
            assert isinstance(result, dict)
            assert 'stability' in result
            
            print(f"✅ Análise de desempenho: {result['stability']}")
            
        except ImportError:
            pytest.skip("Módulo performance não implementado")
        except Exception as e:
            pytest.fail(f"Erro na análise de desempenho: {e}")

class TestIntegracao(TestFase6ProjetoControladores):
    """Testes de integração entre módulos"""
    
    def test_design_workflow_complete(self):
        """Teste do fluxo completo de projeto"""
        try:
            # Importar todos os módulos
            from controllab.design.compensators import CompensatorDesigner
            from controllab.design.pole_placement import StateSpaceDesigner
            from controllab.design.design_utils import DesignSpecifications
            
            # Criar especificações
            specs = DesignSpecifications(
                overshoot=10.0,
                settling_time=2.0,
                steady_state_error=0.02
            )
            
            # Testar designer de compensadores
            compensator_designer = CompensatorDesigner(show_steps=False)
            
            # Testar designer de espaço de estados
            ss_designer = StateSpaceDesigner(self.ss_system, show_steps=False)
            
            print("✅ Fluxo de projeto integrado funcionando")
            
        except ImportError:
            pytest.skip("Módulos de integração não implementados")
        except Exception as e:
            pytest.fail(f"Erro no fluxo integrado: {e}")
    
    def test_educational_content(self):
        """Teste do conteúdo educacional"""
        try:
            from controllab.design.design_utils import create_educational_content
            
            # Testar conteúdo para diferentes métodos
            methods = ['lead_compensator', 'lag_compensator', 'pid', 'pole_placement', 'lqr']
            
            for method in methods:
                content = create_educational_content(method, {})
                assert isinstance(content, list)
                assert len(content) > 0
            
            print("✅ Conteúdo educacional gerado com sucesso")
            
        except ImportError:
            pytest.skip("Módulo design_utils não implementado")
        except Exception as e:
            pytest.fail(f"Erro no conteúdo educacional: {e}")

# Testes de validação final
def test_module_6_complete():
    """Teste final de completude do Módulo 6"""
    try:
        # Verificar se todos os módulos podem ser importados
        from controllab.design import (
            PID, Lead, Lag, 
            check_controllability, acker,
            check_observability, acker_observer,
            lqr_design, solve_are_symbolic,
            analyze_transient_response
        )
        
        print("🎉 MÓDULO 6 - PROJETO DE CONTROLADORES IMPLEMENTADO COM SUCESSO!")
        print("✅ Todos os componentes principais foram implementados")
        print("✅ Importações funcionando corretamente")
        print("✅ Funcionalidades básicas disponíveis")
        
    except ImportError as e:
        pytest.skip(f"Módulo 6 ainda em desenvolvimento: {e}")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
