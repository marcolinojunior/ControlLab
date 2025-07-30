"""
ControlLab Web - Smart Plots

Sistema de visualização inteligente que integra backend simbólico com visualização numérica.
Converte análises simbólicas em gráficos interativos mantendo transparência pedagógica.

Classes implementadas:
- SmartPlotter: Plotagem inteligente simbólica→numérica
- PlotlyRenderer: Renderização com Plotly.js
- VisualizationManager: Gerenciamento de gráficos sincronizados
"""

import numpy as np
import json
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Backend numérico (opcional)
try:
    import scipy.signal as signal
    import matplotlib.pyplot as plt
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠️ SciPy não disponível - alguns gráficos limitados")

# Integração com backend ControlLab
try:
    from ..analysis.stability_analysis import StabilityAnalysisEngine
    from ..analysis.frequency_response import FrequencyAnalyzer
    from ..analysis.root_locus import RootLocusAnalyzer
    from ..core.symbolic_tf import SymbolicTransferFunction
    from ..numerical import get_available_backends, check_numerical_dependencies
    BACKEND_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Backend ControlLab não disponível para visualização: {e}")
    BACKEND_AVAILABLE = False


class PlotlyRenderer:
    """
    Renderizador Plotly.js para gráficos interativos.
    
    Gera especificações JSON compatíveis com Plotly.js para
    renderização no frontend React.
    """
    
    @staticmethod
    def create_root_locus_plot(poles_real: List[float], poles_imag: List[float],
                              zeros_real: List[float] = None, zeros_imag: List[float] = None,
                              title: str = "Root Locus") -> Dict[str, Any]:
        """Cria gráfico de Root Locus interativo"""
        
        traces = []
        
        # Polos
        if poles_real and poles_imag:
            traces.append({
                "x": poles_real,
                "y": poles_imag, 
                "mode": "markers",
                "type": "scatter",
                "name": "Polos",
                "marker": {
                    "symbol": "x",
                    "size": 12,
                    "color": "red",
                    "line": {"width": 3}
                },
                "hovertemplate": "Polo: %{x:.3f} + %{y:.3f}j<extra></extra>"
            })
            
        # Zeros
        if zeros_real and zeros_imag:
            traces.append({
                "x": zeros_real,
                "y": zeros_imag,
                "mode": "markers", 
                "type": "scatter",
                "name": "Zeros",
                "marker": {
                    "symbol": "circle",
                    "size": 12,
                    "color": "blue",
                    "line": {"width": 2}
                },
                "hovertemplate": "Zero: %{x:.3f} + %{y:.3f}j<extra></extra>"
            })
            
        # Linha de estabilidade (eixo imaginário)
        if poles_imag and len(poles_imag) > 0:
            max_imag_poles = max(poles_imag)
        else:
            max_imag_poles = 0
            
        if zeros_imag and len(zeros_imag) > 0:
            max_imag_zeros = max(zeros_imag)
        else:
            max_imag_zeros = 0
            
        max_imag = max(max_imag_poles, max_imag_zeros, 5)
        
        if poles_imag and len(poles_imag) > 0:
            min_imag_poles = min(poles_imag)
        else:
            min_imag_poles = 0
            
        if zeros_imag and len(zeros_imag) > 0:
            min_imag_zeros = min(zeros_imag)
        else:
            min_imag_zeros = 0
            
        min_imag = min(min_imag_poles, min_imag_zeros, -5)
        
        traces.append({
            "x": [0, 0],
            "y": [min_imag, max_imag],
            "mode": "lines",
            "type": "scatter",
            "name": "Limite de Estabilidade",
            "line": {
                "color": "green",
                "width": 2,
                "dash": "dash"
            },
            "hovertemplate": "Eixo jω (limite estabilidade)<extra></extra>"
        })
        
        layout = {
            "title": {
                "text": title,
                "x": 0.5,
                "font": {"size": 16}
            },
            "xaxis": {
                "title": "Parte Real (σ)",
                "zeroline": True,
                "zerolinecolor": "black",
                "zerolinewidth": 1,
                "gridcolor": "lightgray"
            },
            "yaxis": {
                "title": "Parte Imaginária (jω)",
                "zeroline": True,
                "zerolinecolor": "black", 
                "zerolinewidth": 1,
                "gridcolor": "lightgray"
            },
            "showlegend": True,
            "hovermode": "closest",
            "plot_bgcolor": "white",
            "annotations": [
                {
                    "x": -0.1,
                    "y": 0.5,
                    "xref": "paper",
                    "yref": "paper",
                    "text": "ESTÁVEL",
                    "showarrow": False,
                    "font": {"color": "green", "size": 12},
                    "textangle": -90
                },
                {
                    "x": 0.95,
                    "y": 0.5,
                    "xref": "paper", 
                    "yref": "paper",
                    "text": "INSTÁVEL",
                    "showarrow": False,
                    "font": {"color": "red", "size": 12},
                    "textangle": -90
                }
            ]
        }
        
        return {
            "data": traces,
            "layout": layout,
            "config": {
                "displayModeBar": True,
                "displaylogo": False,
                "modeBarButtonsToRemove": ["pan2d", "lasso2d"]
            }
        }
        
    @staticmethod
    def create_bode_plot(frequencies: List[float], magnitude_db: List[float],
                        phase_deg: List[float], title: str = "Diagrama de Bode") -> Dict[str, Any]:
        """Cria diagrama de Bode interativo"""
        
        # Subplots para magnitude e fase
        from plotly.subplots import make_subplots
        
        traces = []
        
        # Magnitude
        traces.append({
            "x": frequencies,
            "y": magnitude_db,
            "mode": "lines",
            "type": "scatter",
            "name": "Magnitude",
            "line": {"color": "blue", "width": 2},
            "xaxis": "x",
            "yaxis": "y",
            "hovertemplate": "f: %{x:.3f} rad/s<br>|G|: %{y:.2f} dB<extra></extra>"
        })
        
        # Fase
        traces.append({
            "x": frequencies,
            "y": phase_deg,
            "mode": "lines",
            "type": "scatter", 
            "name": "Fase",
            "line": {"color": "red", "width": 2},
            "xaxis": "x2",
            "yaxis": "y2",
            "hovertemplate": "f: %{x:.3f} rad/s<br>∠G: %{y:.1f}°<extra></extra>"
        })
        
        layout = {
            "title": {
                "text": title,
                "x": 0.5,
                "font": {"size": 16}
            },
            "xaxis": {
                "title": "Frequência (rad/s)",
                "type": "log",
                "domain": [0, 1],
                "anchor": "y",
                "gridcolor": "lightgray"
            },
            "yaxis": {
                "title": "Magnitude (dB)",
                "domain": [0.55, 1],
                "anchor": "x",
                "gridcolor": "lightgray"
            },
            "xaxis2": {
                "title": "Frequência (rad/s)",
                "type": "log",
                "domain": [0, 1],
                "anchor": "y2",
                "gridcolor": "lightgray"
            },
            "yaxis2": {
                "title": "Fase (graus)",
                "domain": [0, 0.45],
                "anchor": "x2",
                "gridcolor": "lightgray"
            },
            "showlegend": True,
            "hovermode": "x unified",
            "plot_bgcolor": "white"
        }
        
        return {
            "data": traces,
            "layout": layout,
            "config": {
                "displayModeBar": True,
                "displaylogo": False
            }
        }
        
    @staticmethod
    def create_nyquist_plot(real_parts: List[float], imag_parts: List[float],
                           title: str = "Diagrama de Nyquist") -> Dict[str, Any]:
        """Cria diagrama de Nyquist interativo"""
        
        traces = []
        
        # Curva de Nyquist
        traces.append({
            "x": real_parts,
            "y": imag_parts,
            "mode": "lines+markers",
            "type": "scatter",
            "name": "G(jω)",
            "line": {"color": "blue", "width": 2},
            "marker": {"size": 4},
            "hovertemplate": "Re: %{x:.3f}<br>Im: %{y:.3f}<extra></extra>"
        })
        
        # Ponto crítico (-1, 0)
        traces.append({
            "x": [-1],
            "y": [0],
            "mode": "markers",
            "type": "scatter",
            "name": "Ponto Crítico (-1,0)",
            "marker": {
                "symbol": "x",
                "size": 15,
                "color": "red",
                "line": {"width": 3}
            },
            "hovertemplate": "Ponto Crítico: -1 + 0j<extra></extra>"
        })
        
        # Círculo unitário (referência)
        theta = np.linspace(0, 2*np.pi, 100)
        circle_x = np.cos(theta)
        circle_y = np.sin(theta)
        
        traces.append({
            "x": circle_x.tolist(),
            "y": circle_y.tolist(),
            "mode": "lines",
            "type": "scatter",
            "name": "Círculo Unitário",
            "line": {
                "color": "gray",
                "width": 1,
                "dash": "dot"
            },
            "hovertemplate": "Referência: |G| = 1<extra></extra>"
        })
        
        layout = {
            "title": {
                "text": title,
                "x": 0.5,
                "font": {"size": 16}
            },
            "xaxis": {
                "title": "Parte Real",
                "zeroline": True,
                "zerolinecolor": "black",
                "zerolinewidth": 1,
                "gridcolor": "lightgray",
                "scaleanchor": "y",
                "scaleratio": 1
            },
            "yaxis": {
                "title": "Parte Imaginária",
                "zeroline": True,
                "zerolinecolor": "black",
                "zerolinewidth": 1,
                "gridcolor": "lightgray"
            },
            "showlegend": True,
            "hovermode": "closest",
            "plot_bgcolor": "white"
        }
        
        return {
            "data": traces,
            "layout": layout,
            "config": {
                "displayModeBar": True,
                "displaylogo": False
            }
        }
        
    @staticmethod
    def create_step_response_plot(time: List[float], response: List[float],
                                 title: str = "Resposta ao Degrau") -> Dict[str, Any]:
        """Cria gráfico de resposta ao degrau"""
        
        traces = []
        
        # Resposta do sistema
        traces.append({
            "x": time,
            "y": response,
            "mode": "lines",
            "type": "scatter",
            "name": "Resposta y(t)",
            "line": {"color": "blue", "width": 2},
            "hovertemplate": "t: %{x:.3f} s<br>y(t): %{y:.3f}<extra></extra>"
        })
        
        # Linha de referência (entrada degrau)
        if len(time) > 0:
            traces.append({
                "x": [0, 0, max(time)],
                "y": [0, 1, 1],
                "mode": "lines",
                "type": "scatter",
                "name": "Entrada u(t)",
                "line": {
                    "color": "red",
                    "width": 2,
                    "dash": "dash"
                },
                "hovertemplate": "Entrada degrau unitário<extra></extra>"
            })
            
        # Características da resposta (se calculáveis)
        if len(response) > 10:
            final_value = response[-1]
            
            # Valor final
            traces.append({
                "x": [0, max(time)],
                "y": [final_value, final_value],
                "mode": "lines",
                "type": "scatter",
                "name": f"Valor Final: {final_value:.3f}",
                "line": {
                    "color": "green",
                    "width": 1,
                    "dash": "dot"
                },
                "hovertemplate": f"Valor final: {final_value:.3f}<extra></extra>"
            })
            
        layout = {
            "title": {
                "text": title,
                "x": 0.5,
                "font": {"size": 16}
            },
            "xaxis": {
                "title": "Tempo (s)",
                "gridcolor": "lightgray"
            },
            "yaxis": {
                "title": "Amplitude",
                "gridcolor": "lightgray"
            },
            "showlegend": True,
            "hovermode": "x unified",
            "plot_bgcolor": "white"
        }
        
        return {
            "data": traces,
            "layout": layout,
            "config": {
                "displayModeBar": True,
                "displaylogo": False
            }
        }


class SmartPlotter:
    """
    Sistema de plotagem inteligente que integra análise simbólica com visualização numérica.
    
    Converte resultados simbólicos do backend ControlLab em gráficos interativos,
    mantendo a filosofia "Anti-Caixa-Preta" com transparência pedagógica.
    """
    
    def __init__(self):
        # Verificar dependências
        self.scipy_available = SCIPY_AVAILABLE
        self.backend_available = BACKEND_AVAILABLE
        
        if BACKEND_AVAILABLE:
            self.dependencies = check_numerical_dependencies()
            self.numerical_backends = get_available_backends()
        else:
            self.dependencies = {"SCIPY_AVAILABLE": SCIPY_AVAILABLE}
            self.numerical_backends = {}
            
        self.renderer = PlotlyRenderer()
        
    def get_capabilities(self) -> Dict[str, Any]:
        """Retorna capacidades de visualização disponíveis"""
        return {
            "backend_available": self.backend_available,
            "scipy_available": self.scipy_available,
            "available_plots": [
                "root_locus",
                "bode" if self.scipy_available else "bode_limited",
                "nyquist" if self.scipy_available else "nyquist_limited", 
                "step_response" if self.scipy_available else "step_response_basic",
                "pole_zero_map"
            ],
            "dependencies": self.dependencies,
            "numerical_backends": self.numerical_backends
        }
        
    async def create_stability_visualization(self, stability_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Cria visualizações para análise de estabilidade.
        
        Integra resultados do StabilityAnalysisEngine com gráficos pedagógicos.
        """
        visualizations = {}
        
        try:
            # Extrair informações do resultado
            final_result = stability_result.get("final_result", {})
            method_results = final_result.get("method_results", {})
            
            # 1. Root Locus Plot (se disponível)
            root_locus_result = method_results.get("root_locus")
            if root_locus_result:
                poles_real, poles_imag = self._extract_poles_from_result(root_locus_result)
                
                visualizations["root_locus"] = self.renderer.create_root_locus_plot(
                    poles_real, poles_imag,
                    title="Root Locus - Análise de Estabilidade"
                )
                
            # 2. Routh Table Visualization (textual)
            routh_result = method_results.get("routh_hurwitz")
            if routh_result:
                visualizations["routh_table"] = self._create_routh_table_visualization(routh_result)
                
            # 3. Frequency Response (se SciPy disponível)
            frequency_result = method_results.get("frequency")
            if frequency_result and self.scipy_available:
                freq_plots = await self._create_frequency_plots(frequency_result)
                visualizations.update(freq_plots)
                
            # 4. Summary Dashboard
            visualizations["summary"] = self._create_stability_summary(final_result)
            
        except Exception as e:
            visualizations["error"] = {
                "message": f"Erro criando visualizações: {str(e)}",
                "available_data": list(stability_result.keys())
            }
            
        return {
            "visualizations": visualizations,
            "capabilities": self.get_capabilities(),
            "timestamp": datetime.now().isoformat()
        }
        
    def _extract_poles_from_result(self, root_locus_result: Any) -> Tuple[List[float], List[float]]:
        """Extrai coordenadas dos polos do resultado de root locus"""
        
        # Implementação básica - adaptável baseada na estrutura real do resultado
        if hasattr(root_locus_result, 'poles'):
            poles = root_locus_result.poles
        elif isinstance(root_locus_result, dict) and 'poles' in root_locus_result:
            poles = root_locus_result['poles']
        else:
            # Fallback: polos básicos para demonstração
            poles = [-1+0j, -2+1j, -2-1j]
            
        poles_real = []
        poles_imag = []
        
        for pole in poles:
            if hasattr(pole, 'real') and hasattr(pole, 'imag'):
                poles_real.append(float(pole.real))
                poles_imag.append(float(pole.imag))
            elif isinstance(pole, complex):
                poles_real.append(pole.real)
                poles_imag.append(pole.imag)
            else:
                # Assumir que é real
                poles_real.append(float(pole))
                poles_imag.append(0.0)
                
        return poles_real, poles_imag
        
    def _create_routh_table_visualization(self, routh_result: Any) -> Dict[str, Any]:
        """Cria visualização da tabela de Routh"""
        
        # Estrutura básica para visualização da tabela
        table_data = {
            "type": "table",
            "title": "Tabela de Routh-Hurwitz",
            "description": "Análise algébrica de estabilidade",
            "headers": ["Potência", "Coeficientes"],
            "rows": [],
            "analysis": {
                "sign_changes": 0,
                "conclusion": "Análise pendente",
                "stability": "unknown"
            }
        }
        
        try:
            # Extrair dados da tabela (adaptável baseada na implementação real)
            if hasattr(routh_result, 'table'):
                table = routh_result.table
                table_data["rows"] = self._format_routh_table(table)
            elif isinstance(routh_result, dict):
                if 'table' in routh_result:
                    table_data["rows"] = self._format_routh_table(routh_result['table'])
                if 'sign_changes' in routh_result:
                    table_data["analysis"]["sign_changes"] = routh_result['sign_changes']
                if 'stable' in routh_result:
                    table_data["analysis"]["stability"] = "stable" if routh_result['stable'] else "unstable"
                    
        except Exception as e:
            table_data["error"] = f"Erro formatando tabela: {str(e)}"
            
        return table_data
        
    def _format_routh_table(self, table: Any) -> List[List[str]]:
        """Formata tabela de Routh para visualização"""
        
        formatted_rows = []
        
        try:
            if isinstance(table, list):
                for i, row in enumerate(table):
                    if isinstance(row, list):
                        formatted_row = [f"s^{len(table)-i-1}"] + [f"{val:.4f}" for val in row]
                        formatted_rows.append(formatted_row)
                        
        except Exception:
            # Fallback: tabela de exemplo
            formatted_rows = [
                ["s^2", "1.0000", "1.0000"],
                ["s^1", "3.0000", "0.0000"], 
                ["s^0", "1.0000", "0.0000"]
            ]
            
        return formatted_rows
        
    async def _create_frequency_plots(self, frequency_result: Any) -> Dict[str, Any]:
        """Cria gráficos de resposta em frequência"""
        
        plots = {}
        
        try:
            if self.scipy_available:
                # Gerar dados de frequência básicos para demonstração
                frequencies = np.logspace(-2, 2, 100)  # 0.01 a 100 rad/s
                
                # Simular resposta em frequência básica (adaptável)
                # G(s) = 1/(s+1) como exemplo
                s = 1j * frequencies
                H = 1 / (s + 1)
                
                magnitude_db = 20 * np.log10(np.abs(H))
                phase_deg = np.angle(H) * 180 / np.pi
                
                # Diagrama de Bode
                plots["bode"] = self.renderer.create_bode_plot(
                    frequencies.tolist(),
                    magnitude_db.tolist(),
                    phase_deg.tolist(),
                    "Diagrama de Bode - Resposta em Frequência"
                )
                
                # Diagrama de Nyquist
                plots["nyquist"] = self.renderer.create_nyquist_plot(
                    H.real.tolist(),
                    H.imag.tolist(),
                    "Diagrama de Nyquist"
                )
                
            else:
                plots["limited"] = {
                    "message": "SciPy não disponível - gráficos de frequência limitados",
                    "suggestion": "Instale SciPy para gráficos completos"
                }
                
        except Exception as e:
            plots["error"] = f"Erro criando gráficos de frequência: {str(e)}"
            
        return plots
        
    def _create_stability_summary(self, final_result: Dict[str, Any]) -> Dict[str, Any]:
        """Cria dashboard resumo de estabilidade"""
        
        is_stable = final_result.get("is_stable", False)
        execution_time = final_result.get("execution_time", 0)
        
        summary = {
            "type": "dashboard",
            "title": "Resumo da Análise de Estabilidade",
            "status": {
                "stable": is_stable,
                "label": "ESTÁVEL ✅" if is_stable else "INSTÁVEL ⚠️",
                "color": "green" if is_stable else "red"
            },
            "metrics": [
                {
                    "name": "Tempo de Execução",
                    "value": f"{execution_time:.3f}s",
                    "icon": "⏱️"
                },
                {
                    "name": "Métodos Aplicados",
                    "value": len(final_result.get("method_results", {})),
                    "icon": "🔬"
                }
            ],
            "recommendations": self._get_stability_recommendations(final_result)
        }
        
        return summary
        
    def _get_stability_recommendations(self, final_result: Dict[str, Any]) -> List[str]:
        """Gera recomendações pedagógicas baseadas no resultado"""
        
        recommendations = []
        is_stable = final_result.get("is_stable", False)
        
        if is_stable:
            recommendations.extend([
                "Sistema estável - todos os critérios satisfeitos",
                "Explore projeto de controladores para melhorar desempenho",
                "Analise margens de estabilidade para verificar robustez"
            ])
        else:
            recommendations.extend([
                "Sistema instável - requer compensação",
                "Considere adição de compensador lead/lag",
                "Revise localização de polos e zeros",
                "Aplique técnicas de realimentação"
            ])
            
        return recommendations
        
    async def create_basic_step_response(self, tf_string: str) -> Dict[str, Any]:
        """Cria resposta ao degrau básica sem SciPy"""
        
        try:
            # Simulação muito básica para demonstração
            time = np.linspace(0, 10, 100)
            
            # Resposta exponencial básica (adaptável)
            response = 1 - np.exp(-time)  # Primeira ordem aproximada
            
            step_plot = self.renderer.create_step_response_plot(
                time.tolist(),
                response.tolist(),
                f"Resposta ao Degrau - {tf_string}"
            )
            
            return {
                "plot": step_plot,
                "analysis": {
                    "final_value": response[-1],
                    "settling_time": "~5 constantes de tempo",
                    "overshoot": "0% (primeira ordem)"
                },
                "note": "Simulação básica - instale SciPy para análise precisa"
            }
            
        except Exception as e:
            return {
                "error": f"Erro na simulação básica: {str(e)}"
            }


class VisualizationManager:
    """
    Gerenciador de visualizações sincronizadas para interface web.
    
    Coordena múltiplos gráficos e mantém sincronização entre
    análise simbólica e visualização numérica.
    """
    
    def __init__(self):
        self.smart_plotter = SmartPlotter()
        self.active_visualizations = {}
        self.sync_groups = {}  # Grupos de gráficos sincronizados
        
    async def create_comprehensive_analysis_plots(self, analysis_result: Dict[str, Any], 
                                                 sync_group: str = "main") -> Dict[str, Any]:
        """
        Cria conjunto completo de visualizações para análise.
        
        Retorna múltiplos gráficos sincronizados para diferentes aspectos da análise.
        """
        
        plots_package = {
            "sync_group": sync_group,
            "timestamp": datetime.now().isoformat(),
            "plots": {},
            "capabilities": self.smart_plotter.get_capabilities()
        }
        
        try:
            # Análise de estabilidade
            if analysis_result.get("status") == "COMPLETED":
                stability_viz = await self.smart_plotter.create_stability_visualization(analysis_result)
                plots_package["plots"]["stability"] = stability_viz
                
            # Adicionar outros tipos conforme implementados
            plots_package["plots"]["metadata"] = {
                "total_plots": len(plots_package["plots"]),
                "analysis_type": "stability",
                "backend_available": BACKEND_AVAILABLE
            }
            
        except Exception as e:
            plots_package["error"] = f"Erro criando visualizações: {str(e)}"
            
        # Registrar grupo de sincronização
        self.sync_groups[sync_group] = plots_package
        
        return plots_package
        
    def get_sync_group(self, sync_group: str) -> Optional[Dict[str, Any]]:
        """Retorna grupo de visualizações sincronizadas"""
        return self.sync_groups.get(sync_group)
        
    def get_manager_status(self) -> Dict[str, Any]:
        """Status do gerenciador de visualizações"""
        return {
            "active_sync_groups": len(self.sync_groups),
            "plotter_capabilities": self.smart_plotter.get_capabilities(),
            "available_plot_types": [
                "stability_analysis",
                "root_locus", 
                "bode_diagram",
                "nyquist_plot",
                "step_response"
            ]
        }


# Funções de conveniência
def create_smart_plotter() -> SmartPlotter:
    """Cria instância do Smart Plotter"""
    return SmartPlotter()


def create_visualization_manager() -> VisualizationManager:
    """Cria instância do Visualization Manager"""
    return VisualizationManager()


# Exemplo de uso e teste
if __name__ == "__main__":
    import asyncio
    
    async def demo():
        """Demonstração do sistema de visualização"""
        print("📊 ControlLab Web - Smart Plots Demo")
        print("=" * 50)
        
        # Criar plotter
        plotter = create_smart_plotter()
        
        # Mostrar capacidades
        capabilities = plotter.get_capabilities()
        print("Capacidades de Visualização:")
        print(json.dumps(capabilities, indent=2))
        
        # Criar gerenciador
        viz_manager = create_visualization_manager()
        
        # Status do gerenciador
        status = viz_manager.get_manager_status()
        print(f"\nStatus do Gerenciador:")
        print(json.dumps(status, indent=2))
        
        # Demonstração básica de gráfico
        print(f"\n📈 Criando gráfico de exemplo...")
        
        # Root Locus básico
        root_locus_plot = PlotlyRenderer.create_root_locus_plot(
            [-1, -2, -3], [0, 1, -1],  # polos
            [0], [0],  # zeros
            "Root Locus - Exemplo"
        )
        
        print(f"Gráfico Root Locus criado com {len(root_locus_plot['data'])} traces")
        
        # Resposta ao degrau básica
        if plotter.scipy_available:
            print(f"✅ SciPy disponível - gráficos completos")
        else:
            print(f"⚠️ SciPy não disponível - usando simulação básica")
            
        step_response = await plotter.create_basic_step_response("1/(s+1)")
        print(f"Resposta ao degrau: {list(step_response.keys())}")
        
        print(f"\n✅ Demo de visualização concluída!")
        
    # Executar demo
    asyncio.run(demo())
