# ControlLab - Documentação COMPLETA de Funções (Atualizada)
## Módulos 1-7: Referência Técnica e Pedagógica Detalhada

**Data:** 30 de Julho de 2025
**Versão:** 8.0 ATUALIZADA
**Status:** Módulos 1-7 Implementados - TODAS AS FUNÇÕES CATALOGADAS E VERIFICADAS

---

## 📚 Índice

1.  [Módulo Analysis - Análise de Estabilidade](#módulo-analysis---análise-de-estabilidade)
2.  [Módulo Core - Núcleo Simbólico](#módulo-core---núcleo-simbólico)
3.  [Módulo Design - Projeto de Controladores](#módulo-design---projeto-de-controladores)
4.  [Módulo Modeling - Modelagem e Transformadas](#módulo-modeling---modelagem-e-transformadas)
5.  [Módulo Numerical - Interface Numérica](#módulo-numerical---interface-numérica)
6.  [Módulo Visualization - Visualização](#módulo-visualization---visualização)
7.  [Arquivos de Teste e Validação](#arquivos-de-teste-e-validação)

---

## Módulo Analysis - Análise de Estabilidade

> **Contexto Teórico:** Este módulo implementa os conceitos fundamentais do **Capítulo 6 - ESTABILIDADE** do livro de sistemas de controle. A estabilidade é uma propriedade fundamental de sistemas de controle que determina se o sistema irá convergir para um valor finito (estável), divergir para infinito (instável), ou permanecer na condição limítrofe (marginalmente estável).
>
> **Métodos Implementados:**
> - **Critério de Routh-Hurwitz (6.2-6.4):** Método algébrico para determinar estabilidade sem resolver equação característica
> - **Lugar Geométrico das Raízes (Caps 8-9):** Análise gráfica da localização dos polos em função de parâmetros
> - **Resposta em Frequência (Cap 10):** Análise de estabilidade através de diagramas de Bode e Nyquist
>
> **Fundamentos:** Um sistema linear invariante no tempo é estável se todos os polos da função de transferência em malha fechada estão localizados no semiplano esquerdo do plano complexo (parte real negativa).

### 📁 **Arquivo:** `src/controllab/analysis/stability_analysis.py`

### 🔧 **CLASSES IMPLEMENTADAS:**

#### **Classe:** `ComprehensiveStabilityReport`
**Localização:** `src/controllab/analysis/stability_analysis.py` linha 80
**Como chamar:**
```python
from controllab.analysis.stability_analysis import ComprehensiveStabilityReport
report = ComprehensiveStabilityReport()
```

##### **Método:** `__init__(self)`
**Entrada:** Nenhuma
**Saída:** Instância inicializada da classe
**Propósito:** Inicializa relatório pedagógico completo de análise de estabilidade
**Contexto Teórico:** Implementa estrutura para documentar análise multi-método conforme metodologia do Cap. 6, permitindo comparação entre Routh-Hurwitz, lugar das raízes e resposta em frequência

##### **Método:** `add_system_info(self, tf_obj, description: str = "")`
**Entrada:**
- `tf_obj`: Função de transferência ou sistema a ser analisado
- `description`: Descrição opcional do sistema
**Saída:** Nenhuma (modifica estado interno)
**Propósito:** Adiciona informações básicas do sistema ao relatório (numerador, denominador, ordem, polinômio característico)
**Contexto Teórico:** Extrai polinômio característico Q(s) = a₀sⁿ + a₁sⁿ⁻¹ + ... + aₙ fundamental para análise de Routh-Hurwitz (Seção 6.2)

##### **Método:** `add_educational_note(self, category: str, note: str)`
**Entrada:**
- `category`: Categoria da nota educacional
- `note`: Conteúdo da nota educacional
**Saída:** Nenhuma (modifica estado interno)
**Propósito:** Adiciona notas pedagógicas específicas por categoria para explicar conceitos
**Contexto Teórico:** Documenta explicações dos fundamentos teóricos: condições necessárias de estabilidade, interpretação física dos polos, relação com resposta temporal

##### **Método:** `add_conclusion(self, method: str, conclusion: str, confidence: str = "Alta")`
**Entrada:**
- `method`: Nome do método de análise
- `conclusion`: Conclusão obtida pelo método
- `confidence`: Nível de confiança ("Alta", "Média", "Baixa")
**Saída:** Nenhuma (modifica estado interno)
**Propósito:** Registra conclusões de cada método de análise com nível de confiança
**Contexto Teórico:** Compara resultados entre métodos (algébrico/Routh vs. gráfico/root locus vs. frequência/Nyquist) conforme abordagem multicritério

##### **Método:** `get_executive_summary(self) -> str`
**Entrada:** Nenhuma
**Saída:** String com resumo executivo formatado
**Propósito:** Gera resumo executivo da análise mostrando concordância entre métodos e conclusão final
**Contexto Teórico:** Sintetiza análise conforme critérios do Cap. 6: estável se Routh positivo E polos esquerda E margem de fase positiva

##### **Método:** `get_detailed_analysis(self) -> str`
**Entrada:** Nenhuma
**Saída:** String com análise detalhada por método
**Propósito:** Fornece análise detalhada por método (Routh-Hurwitz, Root Locus, Frequência) com histórico de operações
**Contexto Teórico:** Implementa metodologia completa Cap. 6: construção tabela Routh, análise de sinais, interpretação de casos especiais

##### **Método:** `get_educational_section(self) -> str`
**Entrada:** Nenhuma
**Saída:** String com seção educacional completa
**Propósito:** Gera seção pedagógica com conceitos fundamentais, conexões entre métodos, fórmulas e interpretação física
**Contexto Teórico:** Explica fundamentos: condições necessárias (todos coeficientes positivos), suficientes (tabela Routh), relação polos-estabilidade

##### **Método:** `get_full_report(self) -> str`
**Entrada:** Nenhuma
**Saída:** String com relatório completo
**Propósito:** Combina resumo executivo, análise detalhada, seção educacional e validação cruzada
**Contexto Teórico:** Relatório final integrando todas as técnicas de análise de estabilidade dos Caps. 6, 8 e 10

##### **Método:** `get_cross_validation_report(self) -> str`
**Entrada:** Nenhuma
**Saída:** String com relatório de validação cruzada
**Propósito:** Compara resultados entre métodos para verificar concordância e detectar discrepâncias
**Contexto Teórico:** Verifica consistência entre métodos: Routh ↔ localização polos ↔ margens estabilidade (fundamentação teórica sólida)

---

#### **Classe:** `StabilityAnalysisEngine`
**Localização:** `src/controllab/analysis/stability_analysis.py` linha 378
**Como chamar:**
```python
from controllab.analysis.stability_analysis import StabilityAnalysisEngine
engine = StabilityAnalysisEngine()
```

##### **Método:** `__init__(self)`
**Entrada:** Nenhuma
**Saída:** Instância inicializada do motor de análise
**Propósito:** Inicializa todos os analisadores (Routh, Root Locus, Frequência, Validação, Paramétrico)

##### **Método:** `comprehensive_analysis(self, tf_obj, show_all_steps: bool = True, include_validation: bool = True, include_parametric: bool = False) -> ComprehensiveStabilityReport`
**Entrada:**
- `tf_obj`: Função de transferência ou polinômio característico
- `show_all_steps`: Exibir todos os passos pedagógicos
- `include_validation`: Incluir validação cruzada
- `include_parametric`: Incluir análise paramétrica
**Saída:** `ComprehensiveStabilityReport` com análise completa
**Propósito:** Realiza análise de estabilidade usando todos os métodos disponíveis (Routh-Hurwitz, Root Locus, Margens, Validação)

##### **Método:** `analyze_complete_stability(self, tf_obj, show_steps: bool = True) -> ComprehensiveStabilityReport`
**Entrada:**
- `tf_obj`: Função de transferência
- `show_steps`: Exibir passos pedagógicos
**Saída:** `ComprehensiveStabilityReport` com análise completa
**Propósito:** Alias para comprehensive_analysis para compatibilidade com testes

##### **Método:** `quick_stability_check(self, tf_obj) -> Dict[str, Any]`
**Entrada:** `tf_obj`: Função de transferência
**Saída:** Dicionário com resultado rápido de estabilidade
**Propósito:** Verificação rápida de estabilidade usando método mais eficiente (prioritário: Routh-Hurwitz)

##### **Método:** `comparative_analysis(self, systems: List[Any], labels: List[str] = None) -> Dict[str, Any]`
**Entrada:**
- `systems`: Lista de funções de transferência
- `labels`: Rótulos opcionais para os sistemas
**Saída:** Dicionário com análise comparativa entre sistemas
**Propósito:** Análise comparativa de estabilidade entre múltiplos sistemas com resumo estatístico

### 🔧 **FUNÇÕES AUXILIARES:**

#### **Função:** `analyze_stability(tf_obj, show_steps: bool = True) -> ComprehensiveStabilityReport`
**Localização:** `src/controllab/analysis/stability_analysis.py` linha 644
**Como chamar:**
```python
from controllab.analysis.stability_analysis import analyze_stability
result = analyze_stability(transfer_function, show_steps=True)
```
**Entrada:**
- `tf_obj`: Função de transferência
- `show_steps`: Exibir passos pedagógicos
**Saída:** `ComprehensiveStabilityReport` com análise completa
**Propósito:** Função de conveniência para análise completa de estabilidade

#### **Função:** `quick_stability_check(tf_obj) -> bool`
**Localização:** `src/controllab/analysis/stability_analysis.py` linha 650
**Como chamar:**
```python
from controllab.analysis.stability_analysis import quick_stability_check
is_stable = quick_stability_check(transfer_function)
```
**Entrada:** `tf_obj`: Função de transferência
**Saída:** `bool` indicando se o sistema é estável
**Propósito:** Função de conveniência para verificação rápida de estabilidade

#### **Função:** `compare_systems_stability(systems: List[Any], labels: List[str] = None) -> Dict`
**Localização:** `src/controllab/analysis/stability_analysis.py` linha 657
**Como chamar:**
```python
from controllab.analysis.stability_analysis import compare_systems_stability
comparison = compare_systems_stability([tf1, tf2, tf3], ["Sistema 1", "Sistema 2", "Sistema 3"])
```
**Entrada:**
- `systems`: Lista de sistemas para comparar
- `labels`: Rótulos opcionais para identificação
**Saída:** Dicionário com análise comparativa
**Propósito:** Função de conveniência para comparação de estabilidade entre múltiplos sistemas

#### **Função:** `run_module_validation()`
**Localização:** `src/controllab/analysis/stability_analysis.py` linha 667
**Como chamar:**
```python
from controllab.analysis.stability_analysis import run_module_validation
validation_results = run_module_validation()
```
**Entrada:** Nenhuma
**Saída:** Dicionário com resultados da validação completa do módulo
**Propósito:** Executa validação completa do Módulo 5 testando importações, funcionalidades, pedagogia, integração e casos especiais

---

### 📁 **Arquivo:** `src/controllab/analysis/routh_hurwitz.py`

> **Contexto Teórico:** Este arquivo implementa o **Critério de Routh-Hurwitz (Seções 6.2-6.4)** - um método algébrico para determinar a estabilidade de sistemas lineares sem resolver a equação característica. O critério estabelece que um sistema é estável se e somente se todos os elementos da primeira coluna da tabela de Routh têm o mesmo sinal (todos positivos para coeficientes positivos).
>
> **Fundamentos Teóricos:**
> - **Condições Necessárias (6.2):** Todos os coeficientes do polinômio característico devem ser positivos e presentes
> - **Tabela de Routh (6.2):** Arranjo sistemático dos coeficientes que permite análise sem cálculo de raízes
> - **Casos Especiais (6.3):** Tratamento de zeros na primeira coluna e linhas de zeros usando métodos auxiliares
> - **Análise Paramétrica (6.4):** Determinação de faixas de parâmetros para estabilidade
>
> **Relação com Estabilidade:** O número de mudanças de sinal na primeira coluna equals número de polos no semiplano direito

### 🔧 **CLASSES IMPLEMENTADAS:**

#### **Classe:** `RouthAnalysisHistory`
**Localização:** `src/controllab/analysis/routh_hurwitz.py` linha 27
**Como chamar:**
```python
from controllab.analysis.routh_hurwitz import RouthAnalysisHistory
history = RouthAnalysisHistory()
```

##### **Método:** `__init__(self)`
**Entrada:** Nenhuma
**Saída:** Instância inicializada da classe
**Propósito:** Inicializa histórico pedagógico da análise de Routh-Hurwitz
**Contexto Teórico:** Documenta cada passo da construção da tabela de Routh conforme metodologia do Cap. 6, incluindo aplicação das regras de formação e casos especiais

##### **Método:** `add_step(self, step_type: str, description: str, data: Any, explanation: str = "")`
**Entrada:**
- `step_type`: Tipo do passo (ex: "INICIALIZAÇÃO", "COEFICIENTES")
- `description`: Descrição do passo
- `data`: Dados associados ao passo
- `explanation`: Explicação pedagógica opcional
**Saída:** Nenhuma (modifica estado interno)
**Propósito:** Adiciona passo pedagógico ao histórico com numeração automática
**Contexto Teórico:** Registra sequência de construção: 1) Coeficientes iniciais, 2) Primeira linha (sⁿ), 3) Segunda linha (sⁿ⁻¹), 4) Linhas subsequentes usando fórmula de recorrência

##### **Método:** `add_special_case(self, case_type: str, row: int, treatment: str, result: Any)`
**Entrada:**
- `case_type`: Tipo do caso especial ("ZERO_PRIMEIRA_COLUNA", "LINHA_DE_ZEROS")
- `row`: Número da linha onde ocorreu
- `treatment`: Tratamento aplicado
- `result`: Resultado do tratamento
**Saída:** Nenhuma (modifica estado internal)
**Propósito:** Registra casos especiais encontrados e como foram tratados
**Contexto Teórico:** Implementa tratamentos da Seção 6.3: substituição por ε para zeros isolados, método do polinômio auxiliar para linhas de zeros

##### **Método:** `get_formatted_report(self) -> str`
**Entrada:** Nenhuma
**Saída:** String com relatório pedagógico formatado
**Propósito:** Gera relatório completo mostrando polinômio, passos, casos especiais e conclusão
**Contexto Teórico:** Apresenta análise completa seguindo formato pedagógico: polinômio característico → tabela de Routh → análise de sinais → conclusão de estabilidade

---

#### **Classe:** `StabilityResult`
**Localização:** `src/controllab/analysis/routh_hurwitz.py` linha 83
**Como chamar:**
```python
from controllab.analysis.routh_hurwitz import StabilityResult
result = StabilityResult()
```

##### **Método:** `__init__(self)`
**Entrada:** Nenhuma
**Saída:** Instância inicializada da classe
**Propósito:** Inicializa resultado da análise de estabilidade com todos os campos
**Contexto Teórico:** Estrutura de dados para armazenar resultado conforme critério de Routh: estável (sem mudanças de sinal), instável (n mudanças = n polos instáveis), marginalmente estável

##### **Método:** `get_formatted_history(self) -> str`
**Entrada:** Nenhuma
**Saída:** String com histórico formatado da análise
**Propósito:** Retorna histórico pedagógico se disponível, senão mensagem informativa
**Contexto Teórico:** Fornece rastreabilidade pedagógica completa da aplicação do critério de Routh-Hurwitz

##### **Método:** `__str__(self)`
**Entrada:** Nenhuma
**Saída:** String representando o resultado da análise
**Propósito:** Representação textual do resultado (ESTÁVEL, INSTÁVEL, MARGINALMENTE ESTÁVEL)
**Contexto Teórico:** Classificação final conforme critério: estável (todos elementos primeira coluna mesmo sinal), instável (mudanças de sinal presentes)

---

#### **Classe:** `RouthArray`
**Localização:** `src/controllab/analysis/routh_hurwitz.py` linha 114
**Como chamar:**
```python
from controllab.analysis.routh_hurwitz import RouthArray
array = RouthArray(polynomial, variable='s')
```

##### **Método:** `__init__(self, polynomial, variable='s')`
**Entrada:**
- `polynomial`: Polinômio característico
- `variable`: Variável simbólica (padrão 's')
**Saída:** Instância inicializada da classe
**Propósito:** Inicializa representação da tabela de Routh com polinômio e variável
**Contexto Teórico:** Estrutura fundamental para aplicação do critério - organiza coeficientes do polinômio característico Q(s) = aₙsⁿ + aₙ₋₁sⁿ⁻¹ + ... + a₁s + a₀

##### **Método:** `get_coefficients(self)`
**Entrada:** Nenhuma
**Saída:** Lista de coeficientes em ordem decrescente de potência
**Propósito:** Extrai coeficientes do polinômio garantindo coeficientes para todas as potências
**Contexto Teórico:** Implementa verificação das condições necessárias: todos coeficientes devem estar presentes e ser positivos (primeira verificação antes da tabela)

##### **Método:** `display_array(self) -> str`
**Entrada:** Nenhuma
**Saída:** String com tabela de Routh formatada
**Propósito:** Exibe a tabela de Routh em formato tabular legível com potências e elementos
**Contexto Teórico:** Visualização pedagógica da tabela conforme layout padrão: linhas representam potências de s (sⁿ, sⁿ⁻¹, ..., s⁰), colunas contêm elementos calculados

---

#### **Classe:** `RouthHurwitzAnalyzer`
**Localização:** `src/controllab/analysis/routh_hurwitz.py` linha 160
**Como chamar:**
```python
from controllab.analysis.routh_hurwitz import RouthHurwitzAnalyzer
analyzer = RouthHurwitzAnalyzer()
```

##### **Método:** `__init__(self)`
**Entrada:** Nenhuma
**Saída:** Instância inicializada do analisador
**Propósito:** Inicializa analisador com histórico e símbolo epsilon para casos especiais
**Contexto Teórico:** Motor principal para aplicação completa do critério de Routh-Hurwitz incluindo casos especiais das Seções 6.2-6.4

##### **Método:** `build_routh_array(self, polynomial, variable='s', show_steps: bool = True) -> RouthArray`
**Entrada:**
- `polynomial`: Polinômio característico
- `variable`: Variável do polinômio (padrão 's')
- `show_steps`: Exibir passos pedagógicos
**Saída:** `RouthArray` com tabela construída
**Propósito:** Constrói a tabela de Routh completa com tratamento de casos especiais e histórico pedagógico
**Contexto Teórico:** Implementa algoritmo completo da Seção 6.2: 1) Primeira linha com coeficientes pares, 2) Segunda linha com coeficientes ímpares, 3) Linhas subsequentes com fórmula de recorrência

##### **Método:** `analyze_stability(self, routh_obj: RouthArray, show_steps: bool = True) -> StabilityResult`
**Entrada:**
- `routh_obj`: Objeto RouthArray com tabela construída
- `show_steps`: Exibir passos pedagógicos
**Saída:** `StabilityResult` com análise de estabilidade
**Propósito:** Analisa estabilidade contando mudanças de sinal na primeira coluna
**Contexto Teórico:** Aplica critério fundamental: sistema estável ↔ zero mudanças de sinal na primeira coluna, número de mudanças = número de polos no semiplano direito

##### **Método:** `parametric_stability_analysis(self, polynomial, parameter, show_steps: bool = True) -> Dict`
**Entrada:**
- `polynomial`: Polinômio com parâmetro simbólico
- `parameter`: Símbolo do parâmetro
- `show_steps`: Exibir passos pedagógicos
**Saída:** Dicionário com análise paramétrica (condições, faixas de estabilidade)
**Propósito:** Determina faixas de valores paramétricos para as quais o sistema é estável
**Contexto Teórico:** Implementa análise da Seção 6.4: determinação de condições sobre parâmetros K para estabilidade usando inequações da primeira coluna

### 🔧 **FUNÇÕES AUXILIARES:**

#### **Função:** `build_routh_array(polynomial, variable='s', show_steps: bool = True) -> RouthArray`
**Localização:** `src/controllab/analysis/routh_hurwitz.py` linha 553
**Como chamar:**
```python
from controllab.analysis.routh_hurwitz import build_routh_array
array = build_routh_array(s**3 + 2*s**2 + 3*s + 1, variable='s', show_steps=True)
```
**Entrada:**
- `polynomial`: Polinômio característico
- `variable`: Variável simbólica (padrão 's')
- `show_steps`: Exibir passos pedagógicos
**Saída:** `RouthArray` com tabela construída
**Propósito:** Função wrapper para construir tabela de Routh
**Contexto Teórico:** Interface simplificada para aplicação do critério de Routh-Hurwitz

#### **Função:** `analyze_stability(polynomial, variable='s', show_steps: bool = True) -> StabilityResult`
**Localização:** `src/controllab/analysis/routh_hurwitz.py` linha 559
**Como chamar:**
```python
from controllab.analysis.routh_hurwitz import analyze_stability
result = analyze_stability(s**3 + 2*s**2 + 3*s + 1, variable='s', show_steps=True)
```
**Entrada:**
- `polynomial`: Polinômio característico
- `variable`: Variável simbólica (padrão 's')
- `show_steps`: Exibir passos pedagógicos
**Saída:** `StabilityResult` com análise completa
**Propósito:** Função wrapper para análise completa de estabilidade (constrói array + analisa)
**Contexto Teórico:** Implementação completa do critério: verificação condições necessárias → construção tabela → análise sinais → conclusão estabilidade

#### **Função:** `handle_zero_in_first_column(array, row_index)`
**Localização:** `src/controllab/analysis/routh_hurwitz.py` linha 566
**Como chamar:**
```python
from controllab.analysis.routh_hurwitz import handle_zero_in_first_column
modified_array = handle_zero_in_first_column(array, row_index)
```
**Entrada:**
- `array`: Array da tabela de Routh
- `row_index`: Índice da linha com zero na primeira coluna
**Saída:** Array modificado
**Propósito:** Trata caso especial de zero na primeira coluna substituindo por epsilon
**Contexto Teórico:** Implementa primeiro caso especial da Seção 6.3: substitui zero por ε (número positivo pequeno) para continuar construção da tabela

#### **Função:** `handle_row_of_zeros(array, row_index)`
**Localização:** `src/controllab/analysis/routh_hurwitz.py` linha 572
**Como chamar:**
```python
from controllab.analysis.routh_hurwitz import handle_row_of_zeros
modified_array = handle_row_of_zeros(array, row_index)
```
**Entrada:**
- `array`: Array da tabela de Routh
- `row_index`: Índice da linha de zeros
**Saída:** Array modificado
**Propósito:** Trata caso especial de linha de zeros usando polinômio auxiliar
**Contexto Teórico:** Implementa segundo caso especial da Seção 6.3: linha de zeros indica pares de raízes simétricas; usa derivada do polinômio auxiliar para continuar análise
...
(O restante da documentação será gerado na próxima etapa, pois a saída é muito grande)

### 📁 **Arquivo:** `src/controllab/analysis/root_locus.py`

### 🔧 **CLASSES IMPLEMENTADAS:**

#### **Classe:** `LocusHistory`
**Localização:** `src/controllab/analysis/root_locus.py` linha 45
**Como chamar:**
```python
from controllab.analysis.root_locus import LocusHistory
history = LocusHistory()
```

##### **Método:** `__init__(self)`
**Entrada:** Nenhuma
**Saída:** Instância inicializada da classe
**Propósito:** Inicializa histórico pedagógico da análise de root locus

##### **Método:** `add_step(self, rule_number: int, rule_name: str, calculation: Any, result: Any, explanation: str = "")`
**Entrada:**
- `rule_number`: Número da regra do root locus (1-6)
- `rule_name`: Nome da regra
- `calculation`: Cálculo realizado
- `result`: Resultado obtido
- `explanation`: Explicação pedagógica opcional
**Saída:** Nenhuma (modifica estado interno)
**Propósito:** Adiciona passo pedagógico ao histórico com numeração automática e regra associada

##### **Método:** `add_rule(self, rule_number: int, description: str, result: Any)`
**Entrada:**
- `rule_number`: Número da regra (1-6)
- `description`: Descrição da regra
- `result`: Resultado da aplicação da regra
**Saída:** Nenhuma (modifica estado interno)
**Propósito:** Registra aplicação de uma das 6 regras fundamentais do root locus

##### **Método:** `get_formatted_report(self) -> str`
**Entrada:** Nenhuma
**Saída:** String com relatório pedagógico formatado
**Propósito:** Gera relatório completo mostrando função de transferência, regras aplicadas e passos de cálculo

---

#### **Classe:** `LocusFeatures`
**Localização:** `src/controllab/analysis/root_locus.py` linha 99
**Como chamar:**
```python
from controllab.analysis.root_locus import LocusFeatures
features = LocusFeatures()
```

##### **Método:** `__init__(self)`
**Entrada:** Nenhuma
**Saída:** Instância inicializada da classe
**Propósito:** Inicializa estrutura para armazenar todas as características do root locus

##### **Método:** `__str__(self)`
**Entrada:** Nenhuma
**Saída:** String com representação das características
**Propósito:** Exibe resumo das características (polos, zeros, ramos, assíntotas, pontos de separação, cruzamentos)

---

#### **Classe:** `RootLocusAnalyzer`
**Localização:** `src/controllab/analysis/root_locus.py` linha 126
**Como chamar:**
```python
from controllab.analysis.root_locus import RootLocusAnalyzer
analyzer = RootLocusAnalyzer()
```

##### **Método:** `__init__(self)`
**Entrada:** Nenhuma
**Saída:** Instância inicializada do analisador
**Propósito:** Inicializa analisador com histórico e símbolos (s, K) para análise de root locus

##### **Método:** `get_locus_features(self, tf_obj, show_steps: bool = True) -> LocusFeatures`
**Entrada:**
- `tf_obj`: Função de transferência (SymbolicTransferFunction ou expressão)
- `show_steps`: Exibir passos pedagógicos
**Saída:** `LocusFeatures` com todas as características extraídas
**Propósito:** Extrai todas as características do root locus aplicando as 6 regras fundamentais

##### **Método:** `analyze_comprehensive(self, tf_obj, show_steps: bool = True) -> LocusFeatures`
**Entrada:**
- `tf_obj`: Função de transferência
- `show_steps`: Exibir passos pedagógicos
**Saída:** `LocusFeatures` com análise completa incluindo avaliação de estabilidade
**Propósito:** Realiza análise completa do root locus com avaliação adicional de estabilidade e faixas de K

##### **Método:** `calculate_locus_points(self, tf_obj, k_range: List[float], show_steps: bool = False) -> Dict`
**Entrada:**
- `tf_obj`: Função de transferência
- `k_range`: Lista de valores de K para calcular
- `show_steps`: Exibir passos pedagógicos
**Saída:** Dicionário com pontos do locus para cada valor de K
**Propósito:** Calcula pontos específicos do root locus resolvendo equação característica para valores de K

### 🔧 **FUNÇÕES AUXILIARES:**

#### **Função:** `get_locus_features(tf_obj, show_steps: bool = True) -> LocusFeatures`
**Localização:** `src/controllab/analysis/root_locus.py` linha 636
**Como chamar:**
```python
from controllab.analysis.root_locus import get_locus_features
features = get_locus_features(transfer_function, show_steps=True)
```
**Entrada:**
- `tf_obj`: Função de transferência
- `show_steps`: Exibir passos pedagógicos
**Saída:** `LocusFeatures` com características do root locus
**Propósito:** Função wrapper para extrair características do root locus

#### **Função:** `calculate_asymptotes(zeros: List, poles: List) -> Dict`
**Localização:** `src/controllab/analysis/root_locus.py` linha 642
**Como chamar:**
```python
from controllab.analysis.root_locus import calculate_asymptotes
asymptotes = calculate_asymptotes(zeros_list, poles_list)
```
**Entrada:**
- `zeros`: Lista de zeros do sistema
- `poles`: Lista de polos do sistema
**Saída:** Dicionário com ângulos e centroide das assíntotas
**Propósito:** Calcula assíntotas do root locus (ângulos e centroide) usando fórmulas analíticas

#### **Função:** `find_breakaway_points(tf_obj) -> List`
**Localização:** `src/controllab/analysis/root_locus.py` linha 662
**Como chamar:**
```python
from controllab.analysis.root_locus import find_breakaway_points
breakaway = find_breakaway_points(transfer_function)
```
**Entrada:** `tf_obj`: Função de transferência
**Saída:** Lista de pontos de breakaway/break-in
**Propósito:** Encontra pontos de separação e chegada do root locus resolvendo dK/ds = 0

#### **Função:** `find_jw_crossings(tf_obj) -> List`
**Localização:** `src/controllab/analysis/root_locus.py` linha 669
**Como chamar:**
```python
from controllab.analysis.root_locus import find_jw_crossings
crossings = find_jw_crossings(transfer_function)
```
**Entrada:** `tf_obj`: Função de transferência
**Saída:** Lista de cruzamentos do eixo jω
**Propósito:** Encontra cruzamentos do eixo imaginário com valores de K e frequências correspondentes

#### **Função:** `calculate_locus_points(tf_obj, k_range: List[float]) -> Dict`
**Localização:** `src/controllab/analysis/root_locus.py` linha 676
**Como chamar:**
```python
from controllab.analysis.root_locus import calculate_locus_points
points = calculate_locus_points(transfer_function, [0.1, 1, 10, 100])
```
**Entrada:**
- `tf_obj`: Função de transferência
- `k_range`: Lista de valores de ganho K
**Saída:** Dicionário com pontos do root locus para cada K
**Propósito:** Função wrapper para calcular pontos específicos do root locus

### 📁 **Arquivo:** `src/controllab/analysis/frequency_response.py`

### 🔧 **CLASSES IMPLEMENTADAS:**

#### **Classe:** `FrequencyAnalysisHistory`
**Localização:** `src/controllab/analysis/frequency_response.py` linha 35
**Como chamar:**
```python
from controllab.analysis.frequency_response import FrequencyAnalysisHistory
history = FrequencyAnalysisHistory()
```

##### **Método:** `__init__(self)`
**Entrada:** Nenhuma
**Saída:** Instância inicializada da classe
**Propósito:** Inicializa histórico pedagógico da análise de resposta em frequência

##### **Método:** `add_step(self, step_type: str, description: str, calculation: Any, result: Any, explanation: str = "")`
**Entrada:**
- `step_type`: Tipo do passo na análise
- `description`: Descrição do passo
- `calculation`: Cálculo realizado
- `result`: Resultado obtido
- `explanation`: Explicação pedagógica opcional
**Saída:** Nenhuma (modifica estado interno)
**Propósito:** Adiciona passo pedagógico ao histórico com numeração automática

##### **Método:** `add_nyquist_step(self, description: str, encirclements: int, poles_rhp: int, conclusion: str)`
**Entrada:**
- `description`: Descrição da análise de Nyquist
- `encirclements`: Número de encerramentos
- `poles_rhp`: Polos no semiplano direito
- `conclusion`: Conclusão da análise
**Saída:** Nenhuma (modifica estado interno)
**Propósito:** Registra etapa específica da análise de Nyquist com dados pedagógicos
**Contexto Teórico:** Documenta aplicação do **Critério de Nyquist (Cap 10.3-10.5)**: N = Z - P, onde N = encerramentos de -1, Z = polos malha fechada instáveis, P = polos malha aberta instáveis

##### **Método:** `get_formatted_report(self) -> str`
**Entrada:** Nenhuma
**Saída:** String com relatório pedagógico formatado
**Propósito:** Gera relatório completo incluindo passos da análise e análise de Nyquist
**Contexto Teórico:** Apresenta análise pedagógica completa da estabilidade via resposta em frequência, conectando contorno de Nyquist com estabilidade de malha fechada

---

#### **Classe:** `StabilityMargins`
**Localização:** `src/controllab/analysis/frequency_response.py` linha 95
**Como chamar:**
```python
from controllab.analysis.frequency_response import StabilityMargins
margins = StabilityMargins()
```

##### **Método:** `__init__(self)`
**Entrada:** Nenhuma
**Saída:** Instância inicializada da classe
**Propósito:** Inicializa estrutura para armazenar margens de ganho e fase
**Contexto Teórico:** Implementa conceitos de **Margens de Estabilidade (Cap 10.6-10.7)**: margem de ganho = distância de |G(jω)| até instabilidade, margem de fase = distância de ∠G(jω) até instabilidade

##### **Método:** `__str__(self)`
**Entrada:** Nenhuma
**Saída:** String com representação das margens
**Propósito:** Exibe margens de ganho/fase, frequências de cruzamento e status de estabilidade
**Contexto Teórico:** Apresenta margens conforme definições: GM = 1/|G(jωπ)| em dB onde ∠G(jωπ) = -180°, PM = 180° + ∠G(jωc) onde |G(jωc)| = 1

---

#### **Classe:** `FrequencyResponse`
**Localização:** `src/controllab/analysis/frequency_response.py` linha 117
**Como chamar:**
```python
from controllab.analysis.frequency_response import FrequencyResponse
response = FrequencyResponse()
```

##### **Método:** `__init__(self)`
**Entrada:** Nenhuma
**Saída:** Instância inicializada da classe
**Propósito:** Inicializa estrutura para armazenar dados de resposta em frequência
**Contexto Teórico:** Estrutura para armazenar G(jω) onde ω varia de 0 a ∞, fundamental para análise de **Resposta em Frequência (Cap 10)**

##### **Método:** `add_point(self, freq: float, response: Complex)`
**Entrada:**
- `freq`: Frequência em rad/s
- `response`: Resposta complexa G(jω)
**Saída:** Nenhuma (modifica estado interno)
**Propósito:** Adiciona ponto de resposta em frequência calculando magnitude, fase e conversões
**Contexto Teórico:** Calcula |G(jω)| e ∠G(jω) para cada frequência, dados fundamentais para **Diagramas de Bode (Cap 10.2)** e **Nyquist (Cap 10.4)**

---

#### **Classe:** `NyquistContour`
**Localização:** `src/controllab/analysis/frequency_response.py` linha 142
**Como chamar:**
```python
from controllab.analysis.frequency_response import NyquistContour
contour = NyquistContour()
```

##### **Método:** `__init__(self)`
**Entrada:** Nenhuma
**Saída:** Instância inicializada da classe
**Propósito:** Inicializa contorno de Nyquist com caminhos e estruturas para encerramentos
**Contexto Teórico:** Implementa **Contorno de Nyquist (Cap 10.4)**: caminho fechado no plano s contornando semiplano direito para aplicação do Princípio do Argumento

##### **Método:** `count_encirclements_of_point(self, point: Complex = -1+0j) -> int`
**Entrada:** `point`: Ponto para contar encerramentos (padrão -1+0j)
**Saída:** Número inteiro de encerramentos
**Propósito:** Conta encerramentos do ponto especificado pelo contorno de Nyquist
**Contexto Teórico:** Implementa contagem de encerramentos fundamental para **Critério de Nyquist (Cap 10.5)**: N = número de encerramentos de -1 por G(s)H(s)

---

#### **Classe:** `FrequencyAnalyzer`
**Localização:** `src/controllab/analysis/frequency_response.py` linha 159
**Como chamar:**
```python
from controllab.analysis.frequency_response import FrequencyAnalyzer
analyzer = FrequencyAnalyzer()
```

##### **Método:** `__init__(self)`
**Entrada:** Nenhuma
**Saída:** Instância inicializada do analisador
**Propósito:** Inicializa analisador com histórico e símbolos (s, ω) para análise de frequência
**Contexto Teórico:** Motor principal para **Análise de Resposta em Frequência (Cap 10)** incluindo Bode, Nyquist, e cálculo de margens de estabilidade

##### **Método:** `get_nyquist_contour(self, tf_obj, radius: float = 1000, epsilon: float = 1e-6, show_steps: bool = True) -> NyquistContour`
**Entrada:**
- `tf_obj`: Função de transferência
- `radius`: Raio do semicírculo no infinito (padrão 1000)
- `epsilon`: Raio das indentações (padrão 1e-6)
- `show_steps`: Exibir passos pedagógicos
**Saída:** `NyquistContour` com contorno completo
**Propósito:** Constrói contorno de Nyquist completo incluindo eixo jω, indentações e semicírculo
**Contexto Teórico:** Implementa construção do **Contorno de Nyquist (Cap 10.4)**: 1) Eixo jω de 0 a ∞, 2) Indentações ao redor de polos no eixo jω, 3) Semicírculo no infinito

##### **Método:** `calculate_frequency_response(self, tf_obj, omega_range: np.ndarray, show_steps: bool = True) -> FrequencyResponse`
**Entrada:**
- `tf_obj`: Função de transferência
- `omega_range`: Array de frequências em rad/s
- `show_steps`: Exibir passos pedagógicos
**Saída:** `FrequencyResponse` ou dicionário com dados de resposta
**Propósito:** Calcula resposta em frequência G(jω) para faixa especificada

##### **Método:** `apply_nyquist_criterion(self, tf_obj, contour: NyquistContour = None, show_steps: bool = True) -> Dict`
**Entrada:**
- `tf_obj`: Função de transferência
- `contour`: Contorno de Nyquist opcional (será construído se None)
- `show_steps`: Exibir passos pedagógicos
**Saída:** Dicionário com resultado da análise (estabilidade, encerramentos, polos RHP)
**Propósito:** Aplica critério de Nyquist (Z = N + P) para determinar estabilidade em malha fechada

##### **Método:** `calculate_gain_phase_margins(self, tf_obj, show_steps: bool = True) -> StabilityMargins`
**Entrada:**
- `tf_obj`: Função de transferência
- `show_steps`: Exibir passos pedagógicos
**Saída:** `StabilityMargins` com margens calculadas
**Propósito:** Calcula margens de ganho e fase determinando frequências de cruzamento

### 🔧 **FUNÇÕES AUXILIARES:**

#### **Função:** `get_nyquist_contour(tf_obj, radius: float = 1000, epsilon: float = 1e-6) -> NyquistContour`
**Localização:** `src/controllab/analysis/frequency_response.py` linha 570
**Como chamar:**
```python
from controllab.analysis.frequency_response import get_nyquist_contour
contour = get_nyquist_contour(transfer_function, radius=1000, epsilon=1e-6)
```
**Entrada:**
- `tf_obj`: Função de transferência
- `radius`: Raio do semicírculo no infinito
- `epsilon`: Raio das indentações
**Saída:** `NyquistContour` completo
**Propósito:** Função wrapper para construir contorno de Nyquist

#### **Função:** `calculate_frequency_response(tf_obj, omega_range: np.ndarray) -> FrequencyResponse`
**Localização:** `src/controllab/analysis/frequency_response.py` linha 576
**Como chamar:**
```python
from controllab.analysis.frequency_response import calculate_frequency_response
import numpy as np
response = calculate_frequency_response(tf, np.logspace(-2, 2, 100))
```
**Entrada:**
- `tf_obj`: Função de transferência
- `omega_range`: Array de frequências
**Saída:** `FrequencyResponse` com dados calculados
**Propósito:** Função wrapper para calcular resposta em frequência
**Contexto Teórico:** Interface simplificada para calcular G(jω) em faixa de frequências, fundamental para construção de **Diagramas de Bode (Cap 10.2)** com escala logarítmica

#### **Função:** `apply_nyquist_criterion(tf_obj, contour: NyquistContour = None) -> Dict`
**Localização:** `src/controllab/analysis/frequency_response.py` linha 582
**Como chamar:**
```python
from controllab.analysis.frequency_response import apply_nyquist_criterion
result = apply_nyquist_criterion(transfer_function)
```
**Entrada:**
- `tf_obj`: Função de transferência
- `contour`: Contorno de Nyquist opcional
**Saída:** Dicionário com análise de estabilidade
**Propósito:** Função wrapper para aplicar critério de Nyquist
**Contexto Teórico:** Implementação completa do **Critério de Nyquist (Cap 10.5)**: constrói contorno → conta encerramentos → aplica Z = N + P para determinar estabilidade de malha fechada

#### **Função:** `calculate_gain_phase_margins(tf_obj) -> StabilityMargins`
**Localização:** `src/controllab/analysis/frequency_response.py` linha 588
**Como chamar:**
```python
from controllab.analysis.frequency_response import calculate_gain_phase_margins
margins = calculate_gain_phase_margins(transfer_function)
```
**Entrada:** `tf_obj`: Função de transferência
**Saída:** `StabilityMargins` com margens de estabilidade
**Propósito:** Função wrapper para calcular margens de ganho e fase
**Contexto Teórico:** Calcula **Margens de Estabilidade (Cap 10.7)**: GM e PM indicam "quão estável" é o sistema, relacionando-se com robustez do projeto de controle

### 📁 **Arquivo:** `src/controllab/analysis/stability_utils.py`

### 🔧 **CLASSES IMPLEMENTADAS:**

#### **Classe:** `ValidationHistory`
**Localização:** `src/controllab/analysis/stability_utils.py` linha 38
**Como chamar:**
```python
from controllab.analysis.stability_utils import ValidationHistory
history = ValidationHistory()
```

##### **Método:** `__init__(self)`
**Entrada:** Nenhuma
**Saída:** Instância inicializada da classe
**Propósito:** Inicializa histórico de validação cruzada entre métodos

##### **Método:** `add_validation(self, method1: str, method2: str, agreement: bool, details: str = "")`
**Entrada:**
- `method1`: Nome do primeiro método
- `method2`: Nome do segundo método
- `agreement`: Se os métodos concordam
- `details`: Detalhes opcionais da comparação
**Saída:** Nenhuma (modifica estado interno)
**Propósito:** Adiciona validação entre dois métodos de análise de estabilidade

##### **Método:** `add_discrepancy(self, description: str, methods: List[str], explanation: str = "")`
**Entrada:**
- `description`: Descrição da discrepância
- `methods`: Lista de métodos envolvidos
- `explanation`: Explicação opcional da discrepância
**Saída:** Nenhuma (modifica estado interno)
**Propósito:** Registra discrepâncias encontradas entre métodos

##### **Método:** `get_formatted_report(self) -> str`
**Entrada:** Nenhuma
**Saída:** String com relatório formatado
**Propósito:** Gera relatório completo de validação cruzada

---

#### **Classe:** `StabilityValidator`
**Localização:** `src/controllab/analysis/stability_utils.py` linha 83
**Como chamar:**
```python
from controllab.analysis.stability_utils import StabilityValidator
validator = StabilityValidator()
```

##### **Método:** `__init__(self)`
**Entrada:** Nenhuma
**Saída:** Instância inicializada do validador
**Propósito:** Inicializa validador cruzado para métodos de análise de estabilidade

##### **Método:** `validate_stability_methods(self, tf_obj, show_steps: bool = True) -> Dict`
**Entrada:**
- `tf_obj`: Função de transferência
- `show_steps`: Se deve mostrar os passos pedagógicos
**Saída:** Dicionário com resultados de todos os métodos
**Propósito:** Valida estabilidade usando múltiplos métodos (Routh-Hurwitz, cálculo direto, margens)
**Contexto Teórico:** Implementa validação cruzada entre métodos: 1) Critério de Routh-Hurwitz (algébrico), 2) Localização direta de polos (analítico), 3) Margens de estabilidade (frequência). Todos devem concordar para sistema bem modelado

##### **Método:** `_calculate_poles_directly(self, tf_obj) -> List[Complex]`
**Entrada:** `tf_obj`: Função de transferência
**Saída:** Lista de polos complexos
**Propósito:** Calcula polos diretamente resolvendo denominador = 0
**Contexto Teórico:** Implementa cálculo direto dos polos como raízes do denominador D(s) = 0. Método de referência para validar outros métodos de análise de estabilidade

##### **Método:** `_analyze_poles_stability(self, poles: List[Complex]) -> bool`
**Entrada:** `poles`: Lista de polos complexos
**Saída:** Boolean indicando estabilidade
**Propósito:** Analisa estabilidade baseada na localização dos polos
**Contexto Teórico:** Aplica critério fundamental: sistema é estável ↔ todos os polos têm parte real negativa (semiplano esquerdo). Base teórica para todos os outros métodos de análise

##### **Método:** `_perform_cross_validation(self, results: Dict)`
**Entrada:** `results`: Dicionário com resultados dos métodos
**Saída:** Nenhuma (modifica histórico interno)
**Propósito:** Realiza validação cruzada entre métodos e registra discrepâncias

---

#### **Classe:** `ParametricAnalyzer`
**Localização:** `src/controllab/analysis/stability_utils.py` linha 264
**Como chamar:**
```python
from controllab.analysis.stability_utils import ParametricAnalyzer
analyzer = ParametricAnalyzer()
```

##### **Método:** `__init__(self)`
**Entrada:** Nenhuma
**Saída:** Instância inicializada do analisador
**Propósito:** Inicializa analisador de estabilidade paramétrica

##### **Método:** `stability_region_2d(self, system, param1: sp.Symbol, param2: sp.Symbol, param1_range: Tuple[float, float], param2_range: Tuple[float, float], resolution: int = 50) -> Dict`
**Entrada:**
- `system`: Sistema ou polinômio característico
- `param1`, `param2`: Parâmetros a variar
- `param1_range`, `param2_range`: Faixas dos parâmetros
- `resolution`: Resolução da grade (padrão 50)
**Saída:** Dicionário com região de estabilidade 2D
**Propósito:** Determina região de estabilidade em espaço 2D de parâmetros

##### **Método:** `root_locus_3d(self, system, param1: sp.Symbol, param2: sp.Symbol, k_range: List[float]) -> Dict`
**Entrada:**
- `system`: Sistema com parâmetros
- `param1`, `param2`: Parâmetros adicionais
- `k_range`: Faixa de ganhos K
**Saída:** Dicionário com dados 3D do root locus
**Propósito:** Análise de root locus tridimensional

##### **Método:** `analyze_sensitivity(self, system, parameter: sp.Symbol, nominal_value: float = 1.0, perturbation: float = 0.1) -> Dict`
**Entrada:**
- `system`: Sistema a analisar
- `parameter`: Parâmetro para análise de sensibilidade
- `nominal_value`: Valor nominal do parâmetro
- `perturbation`: Perturbação relativa (0.1 = 10%)
**Saída:** Dicionário com análise de sensibilidade
**Propósito:** Analisa sensibilidade das margens de estabilidade

##### **Método:** `sensitivity_analysis(self, system, nominal_params: Dict[sp.Symbol, float], perturbation: float = 0.1) -> Dict`
**Entrada:**
- `system`: Sistema nominal
- `nominal_params`: Valores nominais dos parâmetros
- `perturbation`: Perturbação relativa
**Saída:** Dicionário com análise de sensibilidade
**Propósito:** Análise de sensibilidade das margens de estabilidade para múltiplos parâmetros

### 🔧 **FUNÇÕES AUXILIARES:**

#### **Função:** `validate_stability_methods(tf_obj, show_steps: bool = True) -> Dict`
**Localização:** `src/controllab/analysis/stability_utils.py` linha 516
**Como chamar:**
```python
from controllab.analysis.stability_utils import validate_stability_methods
results = validate_stability_methods(transfer_function, show_steps=True)
```
**Entrada:**
- `tf_obj`: Função de transferência
- `show_steps`: Se deve mostrar os passos
**Saída:** Dicionário com resultados de validação
**Propósito:** Função wrapper para validação cruzada de métodos

#### **Função:** `cross_validate_poles(tf_obj) -> Dict`
**Localização:** `src/controllab/analysis/stability_utils.py` linha 521
**Como chamar:**
```python
from controllab.analysis.stability_utils import cross_validate_poles
poles = cross_validate_poles(transfer_function)
```
**Entrada:** `tf_obj`: Função de transferência
**Saída:** Dicionário com polos validados
**Propósito:** Valida polos calculados por diferentes métodos

#### **Função:** `format_stability_report(results: Dict, include_details: bool = True) -> str`
**Localização:** `src/controllab/analysis/stability_utils.py` linha 526
**Como chamar:**
```python
from controllab.analysis.stability_utils import format_stability_report
report = format_stability_report(results, include_details=True)
```
**Entrada:**
- `results`: Dicionário com resultados de análise
- `include_details`: Se deve incluir detalhes
**Saída:** String com relatório formatado
**Propósito:** Formata relatório completo de análise de estabilidade

#### **Função:** `stability_region_2d(system, param1: sp.Symbol, param2: sp.Symbol, param1_range: Tuple[float, float], param2_range: Tuple[float, float], resolution: int = 50) -> Dict`
**Localização:** `src/controllab/analysis/stability_utils.py` linha 564
**Como chamar:**
```python
from controllab.analysis.stability_utils import stability_region_2d
import sympy as sp
k, tau = sp.symbols('k tau', real=True, positive=True)
region = stability_region_2d(system, k, tau, (0, 10), (0, 5), resolution=100)
```
**Entrada:**
- `system`: Sistema ou polinômio característico
- `param1`, `param2`: Símbolos dos parâmetros
- `param1_range`, `param2_range`: Faixas dos parâmetros
- `resolution`: Resolução da grade
**Saída:** Dicionário com região de estabilidade
**Propósito:** Função wrapper para análise de região de estabilidade 2D

#### **Função:** `root_locus_3d(system, param1: sp.Symbol, param2: sp.Symbol, k_range: List[float]) -> Dict`
**Localização:** `src/controllab/analysis/stability_utils.py` linha 570
**Como chamar:**
```python
from controllab.analysis.stability_utils import root_locus_3d
locus_3d = root_locus_3d(system, param1, param2, [0, 1, 2, 5, 10])
```
**Entrada:**
- `system`: Sistema com parâmetros
- `param1`, `param2`: Símbolos dos parâmetros
- `k_range`: Lista de ganhos K
**Saída:** Dicionário com dados 3D do root locus
**Propósito:** Função wrapper para root locus 3D

### 📁 **Arquivo:** `src/controllab/analysis/__init__.py`

### 🔧 **CLASSES IMPLEMENTADAS (PLACEHOLDERS):**

#### **Classe:** `ResponseCharacteristics`
**Localização:** `src/controllab/analysis/__init__.py` linha 100
**Como chamar:**
```python
from controllab.analysis import ResponseCharacteristics
# NOTA: Esta classe é um placeholder e levantará NotImplementedError
```

##### **Método:** `__init__(self)`
**Entrada:** Nenhuma
**Saída:** Levanta NotImplementedError
**Propósito:** Placeholder - será implementado no Módulo 4 (análise temporal)

---

#### **Classe:** `TransientParameters`
**Localização:** `src/controllab/analysis/__init__.py` linha 105
**Como chamar:**
```python
from controllab.analysis import TransientParameters
# NOTA: Esta classe é um placeholder e levantará NotImplementedError
```

##### **Método:** `__init__(self)`
**Entrada:** Nenhuma
**Saída:** Levanta NotImplementedError
**Propósito:** Placeholder - será implementado no Módulo 4 (análise temporal)

---

#### **Classe:** `ComparisonResult`
**Localização:** `src/controllab/analysis/__init__.py` linha 110
**Como chamar:**
```python
from controllab.analysis import ComparisonResult
# NOTA: Esta classe é um placeholder e levantará NotImplementedError
```

##### **Método:** `__init__(self)`
**Entrada:** Nenhuma
**Saída:** Levanta NotImplementedError
**Propósito:** Placeholder - será implementado no Módulo 4 (análise temporal)

### 🔧 **FUNÇÕES AUXILIARES:**

#### **Função:** `analyze_step_response(*args, **kwargs)`
**Localização:** `src/controllab/analysis/__init__.py` linha 78
**Como chamar:**
```python
from controllab.analysis import analyze_step_response
# NOTA: Esta função é um placeholder e levantará NotImplementedError
```
**Entrada:** Argumentos variados (não implementado)
**Saída:** Levanta NotImplementedError
**Propósito:** Placeholder - será implementado no Módulo 4 para análise de resposta ao degrau

#### **Função:** `analyze_impulse_response(*args, **kwargs)`
**Localização:** `src/controllab/analysis/__init__.py` linha 82
**Como chamar:**
```python
from controllab.analysis import analyze_impulse_response
# NOTA: Esta função é um placeholder e levantará NotImplementedError
```
**Entrada:** Argumentos variados (não implementado)
**Saída:** Levanta NotImplementedError
**Propósito:** Placeholder - será implementado no Módulo 4 para análise de resposta ao impulso

#### **Função:** `analyze_transient_response(*args, **kwargs)`
**Localização:** `src/controllab/analysis/__init__.py` linha 86
**Como chamar:**
```python
from controllab.analysis import analyze_transient_response
# NOTA: Esta função é um placeholder e levantará NotImplementedError
```
**Entrada:** Argumentos variados (não implementado)
**Saída:** Levanta NotImplementedError
**Propósito:** Placeholder - será implementado no Módulo 4 para análise de resposta transitória

#### **Função:** `compare_responses(*args, **kwargs)`
**Localização:** `src/controllab/analysis/__init__.py` linha 90
**Como chamar:**
```python
from controllab.analysis import compare_responses
# NOTA: Esta função é um placeholder e levantará NotImplementedError
```
**Entrada:** Argumentos variados (não implementado)
**Saída:** Levanta NotImplementedError
**Propósito:** Placeholder - será implementado no Módulo 4 para comparação de respostas

#### **Função:** `check_analysis_capabilities() -> Dict`
**Localização:** `src/controllab/analysis/__init__.py` linha 124
**Como chamar:**
```python
from controllab.analysis import check_analysis_capabilities
capabilities = check_analysis_capabilities()
```
**Entrada:** Nenhuma
**Saída:** Dicionário com status de cada módulo
**Propósito:** Verifica quais capacidades de análise estão disponíveis no sistema

### 🔧 **IMPORTS E EXPORTS:**

**Importações disponíveis:**
- `StabilityAnalysisEngine`: Motor principal de análise de estabilidade
- `ComprehensiveStabilityReport`: Relatório completo de estabilidade
- `RouthHurwitzAnalyzer`: Analisador Routh-Hurwitz
- `RootLocusAnalyzer`: Analisador de lugar geométrico
- `FrequencyAnalyzer`: Analisador de resposta em frequência
- `StabilityValidator`: Validador cruzado de métodos
- `ParametricAnalyzer`: Analisador paramétrico
- `analyze_stability`: Função principal de análise
- `quick_stability_check`: Verificação rápida de estabilidade
- `validate_stability_methods`: Validação cruzada de métodos

**Constantes:**
- `TEMPORAL_ANALYSIS_AVAILABLE`: False (módulo não implementado)
- `STABILITY_ANALYSIS_AVAILABLE`: True (módulo implementado)
- `AVAILABLE_ANALYSES`: Lista de análises disponíveis

---

## Módulo 2 - Núcleo Simbólico (Core)

> **Contexto Teórico:** Este módulo implementa os conceitos fundamentais dos **Capítulos 2-3 - MODELAGEM** do livro de sistemas de controle. Representa a base matemática para manipulação simbólica de sistemas de controle, implementando tanto representações no domínio da frequência (funções de transferência) quanto no domínio do tempo (espaço de estados).
>
> **Fundamentos Implementados:**
> - **Funções de Transferência (Cap 2.3):** G(s) = Y(s)/U(s) - representação entrada-saída no domínio de Laplace
> - **Espaço de Estados (Cap 3.3):** Representação moderna {A,B,C,D} - modelo mais geral para sistemas dinâmicos
> - **Transformadas de Laplace (Cap 2.2):** Fundamentação matemática para análise no domínio da frequência
> - **Conversões (Caps 3.5-3.6):** Algoritmos para conversão entre TF ↔ SS preservando características do sistema
>
> **Relação com Teoria:** Implementa manipulação algébrica exata preservando relações matemáticas precisas entre representações, permitindo análise pedagógica passo a passo das transformações.

### 📁 **Arquivo:** `src/controllab/core/__init__.py`

### 🔧 **FUNÇÕES AUXILIARES:**

#### **Função:** `get_advanced_modules() -> Dict`
**Localização:** `src/controllab/core/__init__.py` linha 22
**Como chamar:**
```python
from controllab.core import get_advanced_modules
advanced = get_advanced_modules()
```
**Entrada:** Nenhuma
**Saída:** Dicionário com módulos avançados disponíveis
**Propósito:** Importa módulos avançados somente quando necessário, evitando erros quando dependências não estão instaladas

### 🔧 **IMPORTS E EXPORTS:**

**Importações principais (sempre disponíveis):**
- `SymbolicTransferFunction`: Classe principal para funções de transferência
- `SymbolicStateSpace`: Classe para representação em espaço de estados
- `OperationHistory`, `OperationStep`: Sistema de histórico pedagógico
- Utilitários simbólicos: `create_laplace_variable`, `create_z_variable`, `poly_from_roots`, etc.

**Importações condicionais (dependem de disponibilidade):**
- `stability`: RouthHurwitzAnalyzer, NyquistAnalyzer, BodeAnalyzer, RootLocusAnalyzer
- `controllers`: PIDController, LeadLagCompensator, StateSpaceController, ObserverDesign
- `transforms`: LaplaceTransform, ZTransform, FourierTransform
- `visualization`: SymbolicPlotter, LaTeXGenerator, BlockDiagramGenerator

### 📁 **Arquivo:** `src/controllab/core/symbolic_tf.py`

### 🔧 **CLASSES IMPLEMENTADAS:**

#### **Classe:** `SymbolicTransferFunction`
**Localização:** `src/controllab/core/symbolic_tf.py` linha 18
**Como chamar:**
```python
from controllab.core.symbolic_tf import SymbolicTransferFunction
import sympy as sp
s = sp.symbols('s')
tf = SymbolicTransferFunction(s+1, s**2+2*s+1, s)
```

##### **Método:** `__init__(self, numerator: Union[sp.Expr, int, float], denominator: Union[sp.Expr, int, float], variable: Union[Symbol, str] = 's')`
**Entrada:**
- `numerator`: Numerador da função de transferência
- `denominator`: Denominador da função de transferência
- `variable`: Variável da função (padrão 's')
**Saída:** Instância inicializada da classe
**Propósito:** Inicializa função de transferência simbólica com histórico pedagógico
**Contexto Teórico:** Implementa definição fundamental G(s) = N(s)/D(s) do Cap. 2.3, onde N(s) e D(s) são polinômios em s. Corresponde à transformada de Laplace da resposta ao impulso

##### **Método:** `__str__(self)`
**Entrada:** Nenhuma
**Saída:** String com representação da função
**Propósito:** Representação em string formatada da função de transferência
**Contexto Teórico:** Exibe função na forma padrão de engenharia G(s) = K(s-z₁)(s-z₂).../(s-p₁)(s-p₂)... ou forma polinomial

##### **Método:** `__mul__(self, other: 'SymbolicTransferFunction') -> 'SymbolicTransferFunction'`
**Entrada:** `other`: Outra função de transferência ou escalar
**Saída:** Nova função de transferência produto
**Propósito:** Multiplicação de funções de transferência (conexão série)
**Contexto Teórico:** Implementa conexão em série G₁(s)×G₂(s) da Seção 5.2 - sistemas em cascata têm função de transferência igual ao produto das individuais

##### **Método:** `__add__(self, other) -> 'SymbolicTransferFunction'`
**Entrada:** `other`: Outra função de transferência ou número
**Saída:** Nova função de transferência soma
**Propósito:** Adição de funções de transferência (conexão paralela)
**Contexto Teórico:** Implementa conexão paralela G₁(s)+G₂(s) da Seção 5.2 - sistemas paralelos têm função de transferência igual à soma das individuais

##### **Método:** `poles(self) -> list`
**Entrada:** Nenhuma
**Saída:** Lista de polos
**Propósito:** Calcula os polos da função de transferência
**Contexto Teórico:** Implementa cálculo dos polos (raízes do denominador) que determinam estabilidade e características dinâmicas conforme Cap. 4.2

##### **Método:** `zeros(self) -> list`
**Entrada:** Nenhuma
**Saída:** Lista de zeros
**Propósito:** Calcula os zeros da função de transferência
**Contexto Teórico:** Implementa cálculo dos zeros (raízes do numerador) que afetam resposta transitória conforme Cap. 4.8

##### **Método:** `partial_fractions(self) -> sp.Expr`
**Entrada:** Nenhuma
**Saída:** Expressão em frações parciais
**Propósito:** Expande função de transferência em frações parciais
**Contexto Teórico:** Implementa expansão fundamental para transformada inversa de Laplace (Cap. 4.10) - cada termo corresponde a um modo natural do sistema

##### **Método:** `characteristic_equation(self) -> sp.Expr`
**Entrada:** Nenhuma
**Saída:** Expressão da equação característica
**Propósito:** Retorna equação característica (denominador = 0)
**Contexto Teórico:** Implementa equação característica fundamental 1 + GH = 0 para sistemas realimentados (Cap. 5.3) - suas raízes determinam estabilidade

##### **Método:** `apply_laplace_rules(self, time_expr: sp.Expr, initial_conditions: dict = None) -> 'SymbolicTransferFunction'`
**Entrada:**
- `time_expr`: Expressão no domínio do tempo
- `initial_conditions`: Condições iniciais opcionais
**Saída:** Resultado da transformada de Laplace
**Propósito:** Aplica regras da transformada de Laplace
**Contexto Teórico:** Implementa propriedades da transformada de Laplace (Cap. 2.2): linearidade, derivação (sY(s)-y(0)), integração (Y(s)/s), deslocamento temporal

### 🔧 **PROPRIEDADES:**

##### **Propriedade:** `is_proper -> bool`
**Propósito:** Verifica se a função de transferência é própria

##### **Propriedade:** `degree -> tuple`
**Propósito:** Retorna os graus do numerador e denominador

### 📁 **Arquivo:** `src/controllab/core/symbolic_ss.py`

### 🔧 **CLASSES IMPLEMENTADAS:**

#### **Classe:** `SymbolicStateSpace`
**Localização:** `src/controllab/core/symbolic_ss.py` linha 11
**Como chamar:**
```python
from controllab.core.symbolic_ss import SymbolicStateSpace
import sympy as sp
A = sp.Matrix([[0, 1], [-2, -3]])
B = sp.Matrix([[0], [1]])
C = sp.Matrix([[1, 0]])
D = sp.Matrix([[0]])
ss = SymbolicStateSpace(A, B, C, D)
```

##### **Método:** `__init__(self, A: Union[Matrix, list], B: Union[Matrix, list], C: Union[Matrix, list], D: Union[Matrix, list])`
**Entrada:**
- `A`: Matriz de estados (n×n)
- `B`: Matriz de entrada (n×m)
- `C`: Matriz de saída (p×n)
- `D`: Matriz de transmissão direta (p×m)
**Saída:** Instância inicializada da classe
**Propósito:** Inicializa sistema em espaço de estados simbólico com validação dimensional

##### **Método:** `_validate_dimensions(self)`
**Entrada:** Nenhuma
**Saída:** Nenhuma (levanta exceção se inválido)
**Propósito:** Valida as dimensões das matrizes do sistema

##### **Método:** `__str__(self)`
**Entrada:** Nenhuma
**Saída:** String com representação do sistema
**Propósito:** Representação em string formatada

##### **Método:** `__repr__(self)`
**Entrada:** Nenhuma
**Saída:** String com representação detalhada
**Propósito:** Representação técnica para depuração

##### **Método:** `substitute(self, substitutions: Dict[Symbol, Union[int, float, Symbol]]) -> 'SymbolicStateSpace'`
**Entrada:** `substitutions`: Dicionário com substituições {símbolo: valor}
**Saída:** Sistema com substituições aplicadas
**Propósito:** Substitui símbolos em todas as matrizes do sistema

##### **Método:** `simplify(self) -> 'SymbolicStateSpace'`
**Entrada:** Nenhuma
**Saída:** Sistema simplificado
**Propósito:** Simplifica todas as matrizes do sistema

##### **Método:** `eigenvalues(self) -> list`
**Entrada:** Nenhuma
**Saída:** Lista de autovalores
**Propósito:** Calcula os autovalores da matriz A (polos do sistema)

##### **Método:** `characteristic_polynomial(self, variable: Symbol = None) -> sp.Expr`
**Entrada:** `variable`: Variável do polinômio (padrão 's')
**Saída:** Polinômio característico
**Propósito:** Calcula o polinômio característico det(sI - A)

##### **Método:** `transfer_function(self, variable: Symbol = None) -> Matrix`
**Entrada:** `variable`: Variável da função de transferência (padrão 's')
**Saída:** Matriz de funções de transferência
**Propósito:** Calcula a função de transferência G(s) = C(sI - A)^(-1)B + D

##### **Método:** `is_controllable(self) -> bool`
**Entrada:** Nenhuma
**Saída:** Boolean indicando controlabilidade
**Propósito:** Verifica controlabilidade usando a matriz de controlabilidade

##### **Método:** `is_observable(self) -> bool`
**Entrada:** Nenhuma
**Saída:** Boolean indicando observabilidade
**Propósito:** Verifica observabilidade usando a matriz de observabilidade

##### **Método:** `to_latex(self) -> str`
**Entrada:** Nenhuma
**Saída:** String com código LaTeX
**Propósito:** Converte para representação LaTeX do sistema

##### **Método:** `series(self, other: 'SymbolicStateSpace') -> 'SymbolicStateSpace'`
**Entrada:** `other`: Outro sistema em espaço de estados
**Saída:** Sistema resultante da conexão em série
**Propósito:** Conexão em série com outro sistema

### 🔧 **PROPRIEDADES:**

##### **Propriedade:** `n_states -> int`
**Propósito:** Número de estados do sistema

##### **Propriedade:** `n_inputs -> int`
**Propósito:** Número de entradas do sistema

##### **Propriedade:** `n_outputs -> int`
**Propósito:** Número de saídas do sistema

### 📁 **Arquivo:** `src/controllab/core/history.py`

### 🔧 **CLASSES IMPLEMENTADAS:**

#### **Classe:** `OperationStep`
**Localização:** `src/controllab/core/history.py` linha 10
**Como chamar:**
```python
from controllab.core.history import OperationStep
step = OperationStep("OPERAÇÃO", "Descrição", "antes", "depois", {"info": "adicional"})
```

##### **Método:** `__init__(self, operation: str, description: str, before: Any, after: Any, metadata: Optional[Dict] = None)`
**Entrada:**
- `operation`: Tipo de operação
- `description`: Descrição da operação
- `before`: Estado antes da operação
- `after`: Estado após a operação
- `metadata`: Metadados opcionais
**Saída:** Instância inicializada da classe
**Propósito:** Representa um passo de operação no histórico com timestamp

##### **Método:** `__str__(self)`
**Entrada:** Nenhuma
**Saída:** String com representação do passo
**Propósito:** Representação formatada do passo de operação

---

#### **Classe:** `OperationHistory`
**Localização:** `src/controllab/core/history.py` linha 21
**Como chamar:**
```python
from controllab.core.history import OperationHistory
history = OperationHistory(max_steps=100)
```

##### **Método:** `__init__(self, max_steps: int = 100)`
**Entrada:** `max_steps`: Número máximo de passos a manter
**Saída:** Instância inicializada da classe
**Propósito:** Sistema de histórico para rastreamento pedagógico de operações

##### **Método:** `add_step(self, operation: str, description: str, before: Any, after: Any, metadata: Optional[Dict] = None)`
**Entrada:**
- `operation`: Tipo de operação
- `description`: Descrição da operação
- `before`: Estado antes
- `after`: Estado depois
- `metadata`: Metadados opcionais
**Saída:** Nenhuma (modifica estado interno)
**Propósito:** Adiciona um passo ao histórico com limite automático

##### **Método:** `get_formatted_steps(self, format_type: str = "text") -> str`
**Entrada:** `format_type`: Tipo de formatação ('text', 'latex', 'html')
**Saída:** String com passos formatados
**Propósito:** Retorna os passos formatados no tipo especificado

##### **Método:** `_format_text(self) -> str`
**Entrada:** Nenhuma
**Saída:** String formatada em texto simples
**Propósito:** Formatação em texto simples

##### **Método:** `_format_latex(self) -> str`
**Entrada:** Nenhuma
**Saída:** String formatada em LaTeX
**Propósito:** Formatação em LaTeX

##### **Método:** `_format_html(self) -> str`
**Entrada:** Nenhuma
**Saída:** String formatada em HTML
**Propósito:** Formatação em HTML

##### **Método:** `_to_latex_safe(self, expr) -> str`
**Entrada:** `expr`: Expressão a converter
**Saída:** String em LaTeX
**Propósito:** Converte expressão para LaTeX de forma segura

##### **Método:** `clear(self)`
**Entrada:** Nenhuma
**Saída:** Nenhuma (modifica estado interno)
**Propósito:** Limpa o histórico

##### **Método:** `get_last_step(self) -> Optional[OperationStep]`
**Entrada:** Nenhuma
**Saída:** Último passo ou None
**Propósito:** Retorna o último passo registrado

##### **Método:** `get_steps_by_operation(self, operation: str) -> List[OperationStep]`
**Entrada:** `operation`: Tipo de operação a filtrar
**Saída:** Lista de passos filtrados
**Propósito:** Retorna passos filtrados por tipo de operação

##### **Método:** `get_formatted_history(self) -> List[str]`
**Entrada:** Nenhuma
**Saída:** Lista de strings formatadas
**Propósito:** Retorna histórico formatado como lista de strings
**Contexto Teórico:** Gera relatório pedagógico sequencial de todas as operações matemáticas realizadas, fundamental para ensino passo a passo de sistemas de controle

##### **Método:** `get_latex_history(self) -> List[str]`
**Entrada:** Nenhuma
**Saída:** Lista de strings em LaTeX
**Propósito:** Retorna histórico formatado em LaTeX
**Contexto Teórico:** Produz documentação matemática profissional das derivações, permitindo geração automática de material didático com notação matemática padrão

##### **Método:** `export_to_dict(self) -> List[Dict]`
**Entrada:** Nenhuma
**Saída:** Lista de dicionários
**Propósito:** Exporta histórico para formato de dicionário
**Contexto Teórico:** Facilita integração com sistemas externos e armazenamento de sessões pedagógicas para posterior análise ou reprodução

---

#### **Classe:** `HistoryManager`
**Localização:** `src/controllab/core/history.py` linha 181
**Como chamar:**
```python
from controllab.core.history import HistoryManager
manager = HistoryManager(max_steps=100)
```

##### **Método:** `__init__(self, max_steps: int = 100)`
**Entrada:** `max_steps`: Número máximo de passos
**Saída:** Instância inicializada da classe
**Propósito:** Gerenciador simplificado de histórico para interface numérica
**Contexto Teórico:** Versão otimizada do sistema de histórico para operações numéricas de alta frequência, mantendo rastreabilidade sem impacto na performance

##### **Método:** `add_step(self, operation: str, description: str, before: Any, after: Any, metadata: Optional[Dict] = None)`
**Entrada:** Parâmetros do passo de operação
**Saída:** Nenhuma (modifica estado interno)
**Propósito:** Adiciona um passo ao histórico
**Contexto Teórico:** Registra transformações numéricas preservando contexto simbólico original, essencial para validação de conversões simbólico-numéricas

##### **Método:** `get_full_history(self) -> List[Dict]`
**Entrada:** Nenhuma
**Saída:** Lista de dicionários com histórico completo
**Propósito:** Retorna histórico completo como lista de dicionários
**Contexto Teórico:** Fornece auditoria completa das operações para verificação de precisão numérica e debugging de algoritmos

##### **Método:** `clear_history(self)`
**Entrada:** Nenhuma
**Saída:** Nenhuma (modifica estado interno)
**Propósito:** Limpa o histórico
**Contexto Teórico:** Reset do sistema para nova sessão de análise, mantendo performance em análises longas

##### **Método:** `get_formatted_history(self) -> List[str]`
**Entrada:** Nenhuma
**Saída:** Lista de strings formatadas
**Propósito:** Retorna histórico formatado
**Contexto Teórico:** Geração de relatórios legíveis das operações numéricas para documentação e ensino

---

### 📁 **Arquivo:** `src/controllab/core/symbolic_utils.py`

### 🔧 **FUNÇÕES AUXILIARES:**

#### **Função:** `create_laplace_variable(name: str = 's') -> Symbol`
**Localização:** `src/controllab/core/symbolic_utils.py` linha 10
**Como chamar:**
```python
from controllab.core.symbolic_utils import create_laplace_variable
s = create_laplace_variable('s')
```
**Entrada:** `name`: Nome da variável (padrão 's')
**Saída:** Variável simbólica de Laplace
**Propósito:** Cria uma variável de Laplace para uso em funções de transferência
**Contexto Teórico:** Define variável complexa s = σ + jω fundamental para **análise no domínio de Laplace (Cap 2.2)**, base de toda teoria de sistemas lineares

#### **Função:** `create_z_variable(name: str = 'z') -> Symbol`
**Localização:** `src/controllab/core/symbolic_utils.py` linha 20
**Como chamar:**
```python
from controllab.core.symbolic_utils import create_z_variable
z = create_z_variable('z')
```
**Entrada:** `name`: Nome da variável (padrão 'z')
**Saída:** Variável simbólica Z
**Propósito:** Cria variável Z para sistemas discretos
**Contexto Teórico:** Define variável complexa z fundamental para **Sistemas de Controle Digital (Cap 13)** onde z = e^(sT) relaciona domínios s e z
**Saída:** Variável simbólica da transformada Z
**Propósito:** Cria uma variável de transformada Z para sistemas discretos

#### **Função:** `poly_from_roots(roots: List[Union[int, float, complex, Symbol]], variable: Symbol) -> sp.Expr`
**Localização:** `src/controllab/core/symbolic_utils.py` linha 30
**Como chamar:**
```python
from controllab.core.symbolic_utils import poly_from_roots
import sympy as sp
s = sp.symbols('s')
poly = poly_from_roots([-1, -2, -3], s)
```
**Entrada:**
- `roots`: Lista de raízes
- `variable`: Variável do polinômio
**Saída:** Polinômio com as raízes especificadas
**Propósito:** Cria um polinômio a partir de suas raízes

#### **Função:** `validate_proper_tf(numerator: Union[Poly, sp.Expr], denominator: Union[Poly, sp.Expr]) -> bool`
**Localização:** `src/controllab/core/symbolic_utils.py` linha 40
**Como chamar:**
```python
from controllab.core.symbolic_utils import validate_proper_tf
is_proper = validate_proper_tf(numerator, denominator)
```
**Entrada:**
- `numerator`: Numerador da função de transferência
- `denominator`: Denominador da função de transferência
**Saída:** True se for própria, False caso contrário
**Propósito:** Valida se uma função de transferência é própria (grau num ≤ grau den)

#### **Função:** `cancel_common_factors(numerator: Union[Poly, sp.Expr], denominator: Union[Poly, sp.Expr]) -> Tuple[sp.Expr, sp.Expr]`
**Localização:** `src/controllab/core/symbolic_utils.py` linha 65
**Como chamar:**
```python
from controllab.core.symbolic_utils import cancel_common_factors
num_simp, den_simp = cancel_common_factors(numerator, denominator)
#### **Função:** `cancel_common_factors(numerator, denominator) -> Tuple`
**Localização:** `src/controllab/core/symbolic_utils.py` linha 77
**Como chamar:**
```python
from controllab.core.symbolic_utils import cancel_common_factors
num_simplified, den_simplified = cancel_common_factors(numerator, denominator)
```
**Entrada:**
- `numerator`: Numerador
- `denominator`: Denominador
**Saída:** Tupla com numerador e denominador simplificados
**Propósito:** Cancela fatores comuns entre numerador e denominador
**Contexto Teórico:** Implementa simplificação algébrica fundamental G(s) = N(s)/D(s) → N'(s)/D'(s) cancelando zeros e polos coincidentes, essencial para forma mínima de sistemas

#### **Função:** `extract_poles_zeros(numerator: Union[Poly, sp.Expr], denominator: Union[Poly, sp.Expr]) -> Tuple[List, List]`
**Localização:** `src/controllab/core/symbolic_utils.py` linha 97
**Como chamar:**
```python
from controllab.core.symbolic_utils import extract_poles_zeros
zeros, poles = extract_poles_zeros(numerator, denominator)
```
**Entrada:**
- `numerator`: Numerador da função de transferência
- `denominator`: Denominador da função de transferência
**Saída:** Tupla com lista de zeros e lista de polos
**Propósito:** Extrai polos e zeros de uma função de transferência
**Contexto Teórico:** Calcula singularidades fundamentais: **polos** (raízes de D(s)) determinam estabilidade (Cap 4.2), **zeros** (raízes de N(s)) afetam resposta transitória (Cap 4.8)

#### **Função:** `create_proper_tf(zeros: List, poles: List, gain: float = 1.0, variable: Symbol = None) -> Tuple[sp.Expr, sp.Expr]`
**Localização:** `src/controllab/core/symbolic_utils.py` linha 137
**Como chamar:**
```python
from controllab.core.symbolic_utils import create_proper_tf
num, den = create_proper_tf(zeros=[-1], poles=[-2, -3], gain=2.0)
```
**Entrada:**
- `zeros`: Lista de zeros
- `poles`: Lista de polos
- `gain`: Ganho da função de transferência
- `variable`: Variável da função (padrão 's')
**Saída:** Tupla com numerador e denominador
**Propósito:** Cria uma função de transferência própria a partir de polos, zeros e ganho
**Contexto Teórico:** Constrói G(s) = K∏(s-zᵢ)/∏(s-pⱼ) na **forma fatorada (Cap 2.3)**, representação fundamental que revela diretamente características dinâmicas do sistema

#### **Função:** `expand_partial_fractions(numerator: sp.Expr, denominator: sp.Expr, variable: Symbol = None) -> sp.Expr`
**Localização:** `src/controllab/core/symbolic_utils.py` linha 159
**Como chamar:**
```python
from controllab.core.symbolic_utils import expand_partial_fractions
expansion = expand_partial_fractions(numerator, denominator)
```
**Entrada:**
- `numerator`: Numerador
- `denominator`: Denominador
- `variable`: Variável (padrão 's')
**Saída:** Expansão em frações parciais
**Propósito:** Expande função de transferência em frações parciais
**Contexto Teórico:** Implementa **expansão em frações parciais (Cap 4.10)**: G(s) = A₁/(s-p₁) + A₂/(s-p₂) + ... essencial para transformada inversa de Laplace e análise temporal

#### **Função:** `symbolic_stability_analysis(denominator: sp.Expr, variable: Symbol = None) -> dict`
**Localização:** `src/controllab/core/symbolic_utils.py` linha 175
**Como chamar:**
```python
from controllab.core.symbolic_utils import symbolic_stability_analysis
result = symbolic_stability_analysis(denominator)
```
**Entrada:**
- `denominator`: Polinômio característico
- `variable`: Variável do polinômio
**Saída:** Dicionário com resultado da análise de estabilidade
**Propósito:** Análise de estabilidade simbólica usando critério de Routh-Hurwitz

#### **Função:** `convert_to_latex_formatted(expression: sp.Expr) -> str`
**Localização:** `src/controllab/core/symbolic_utils.py` linha 213
**Como chamar:**
```python
from controllab.core.symbolic_utils import convert_to_latex_formatted
latex_code = convert_to_latex_formatted(expression)
```
**Entrada:** `expression`: Expressão SymPy
**Saída:** String com código LaTeX formatado
**Propósito:** Converte expressão para LaTeX formatado

### 📁 **Arquivo:** `src/controllab/core/transforms.py`

### 🔧 **CLASSES IMPLEMENTADAS:**

#### **Classe:** `LaplaceTransform`
**Localização:** `src/controllab/core/transforms.py` linha 10
**Como chamar:**
```python
from controllab.core.transforms import LaplaceTransform
laplace = LaplaceTransform()
```

##### **Método:** `__init__(self)`
**Entrada:** Nenhuma
**Saída:** Instância inicializada da classe
**Propósito:** Inicializa classe para transformadas de Laplace com histórico

##### **Método:** `transform(self, time_function: sp.Expr, time_var: sp.Symbol = None, s_var: sp.Symbol = None) -> sp.Expr`
**Entrada:**
- `time_function`: Função no domínio do tempo
- `time_var`: Variável temporal (default: t)
- `s_var`: Variável de Laplace (default: s)
**Saída:** Transformada de Laplace
**Propósito:** Aplica transformada de Laplace com regras básicas e fallback

##### **Método:** `inverse_transform(self, s_function: sp.Expr, s_var: sp.Symbol = None, time_var: sp.Symbol = None) -> sp.Expr`
**Entrada:**
- `s_function`: Função no domínio de Laplace
- `s_var`: Variável de Laplace (default: s)
- `time_var`: Variável temporal (default: t)
**Saída:** Função no domínio do tempo
**Propósito:** Aplica transformada inversa de Laplace usando frações parciais

##### **Método:** `_apply_basic_rules(self, time_function: sp.Expr, time_var: sp.Symbol, s_var: sp.Symbol) -> sp.Expr`
**Entrada:** Função no domínio do tempo e variáveis
**Saída:** Transformada aplicando regras básicas
**Propósito:** Aplica regras básicas da transformada de Laplace

##### **Método:** `_partial_fraction_inverse(self, s_function: sp.Expr, s_var: sp.Symbol, time_var: sp.Symbol) -> sp.Expr`
**Entrada:** Função de Laplace e variáveis
**Saída:** Inversa usando frações parciais
**Propósito:** Calcula inversa usando decomposição em frações parciais

##### **Método:** `_inverse_simple_term(self, term: sp.Expr, s_var: sp.Symbol, time_var: sp.Symbol) -> sp.Expr`
**Entrada:** Termo simples e variáveis
**Saída:** Inversa do termo
**Propósito:** Calcula inversa de termos simples conhecidos

---

#### **Classe:** `ZTransform`
**Localização:** `src/controllab/core/transforms.py` linha 207
**Como chamar:**
```python
from controllab.core.transforms import ZTransform
z_transform = ZTransform()
```

##### **Método:** `__init__(self)`
**Entrada:** Nenhuma
**Saída:** Instância inicializada da classe
**Propósito:** Inicializa classe para transformadas Z com histórico

##### **Método:** `transform(self, discrete_sequence: sp.Expr, n_var: sp.Symbol = None, z_var: sp.Symbol = None) -> sp.Expr`
**Entrada:**
- `discrete_sequence`: Sequência discreta x[n]
- `n_var`: Variável discreta (default: n)
- `z_var`: Variável Z (default: z)
**Saída:** Transformada Z
**Propósito:** Aplica transformada Z para sequências discretas

##### **Método:** `inverse_transform(self, z_function: sp.Expr, z_var: sp.Symbol = None, n_var: sp.Symbol = None) -> sp.Expr`
**Entrada:**
- `z_function`: Função no domínio Z
- `z_var`: Variável Z (default: z)
- `n_var`: Variável discreta (default: n)
**Saída:** Sequência no domínio discreto
**Propósito:** Aplica transformada inversa Z

##### **Método:** `_apply_z_rules(self, sequence: sp.Expr, n_var: sp.Symbol, z_var: sp.Symbol) -> sp.Expr`
**Entrada:** Sequência e variáveis
**Saída:** Transformada Z aplicando regras
**Propósito:** Aplica regras básicas da transformada Z

##### **Método:** `_z_inverse_partial_fractions(self, z_function: sp.Expr, z_var: sp.Symbol, n_var: sp.Symbol) -> sp.Expr`
**Entrada:** Função Z e variáveis
**Saída:** Inversa usando frações parciais
**Propósito:** Calcula inversa Z usando frações parciais

##### **Método:** `_z_inverse_simple_term(self, term: sp.Expr, z_var: sp.Symbol, n_var: sp.Symbol) -> sp.Expr`
**Entrada:** Termo simples e variáveis
**Saída:** Inversa do termo
**Propósito:** Calcula inversa Z de termos simples conhecidos

---

#### **Classe:** `FourierTransform`
**Localização:** `src/controllab/core/transforms.py` linha 384
**Como chamar:**
```python
from controllab.core.transforms import FourierTransform
fourier = FourierTransform()
```

##### **Método:** `__init__(self)`
**Entrada:** Nenhuma
**Saída:** Instância inicializada da classe
**Propósito:** Inicializa classe para transformadas de Fourier com histórico

##### **Método:** `transform(self, time_function: sp.Expr, time_var: sp.Symbol = None, freq_var: sp.Symbol = None) -> sp.Expr`
**Entrada:**
- `time_function`: Função no domínio do tempo
- `time_var`: Variável temporal (default: t)
- `freq_var`: Variável de frequência (default: omega)
**Saída:** Transformada de Fourier
**Propósito:** Aplica transformada de Fourier

##### **Método:** `inverse_transform(self, freq_function: sp.Expr, freq_var: sp.Symbol = None, time_var: sp.Symbol = None) -> sp.Expr`
**Entrada:**
- `freq_function`: Função no domínio da frequência
- `freq_var`: Variável de frequência (default: omega)
- `time_var`: Variável temporal (default: t)
**Saída:** Função no domínio do tempo
**Propósito:** Aplica transformada inversa de Fourier

---

### 📁 **Arquivo:** `src/controllab/core/visualization.py`

### 🔧 **CLASSES IMPLEMENTADAS:**

#### **Classe:** `SymbolicPlotter`
**Localização:** `src/controllab/core/visualization.py` linha 11
**Como chamar:**
```python
from controllab.core.visualization import SymbolicPlotter
plotter = SymbolicPlotter()
```

##### **Método:** `__init__(self)`
**Entrada:** Nenhuma
**Saída:** Instância inicializada da classe
**Propósito:** Inicializa classe para geração de gráficos simbólicos

##### **Método:** `generate_bode_expressions(self, transfer_function: SymbolicTransferFunction) -> dict`
**Entrada:** `transfer_function`: Função de transferência
**Saída:** Dicionário com expressões para magnitude e fase
**Propósito:** Gera expressões simbólicas para diagrama de Bode

##### **Método:** `generate_nyquist_expression(self, transfer_function: SymbolicTransferFunction) -> dict`
**Entrada:** `transfer_function`: Função de transferência
**Saída:** Dicionário com expressões para parte real e imaginária
**Propósito:** Gera expressão simbólica para diagrama de Nyquist

##### **Método:** `generate_root_locus_equations(self, open_loop_tf: SymbolicTransferFunction, gain_symbol: sp.Symbol = None) -> dict`
**Entrada:**
- `open_loop_tf`: Função de transferência de malha aberta
- `gain_symbol`: Símbolo do ganho (default: K)
**Saída:** Dicionário com equações e informações do lugar das raízes
**Propósito:** Gera equações para lugar das raízes

### 📁 **Arquivo:** `src/controllab/core/controller_design.py`

### 🔧 **CLASSES IMPLEMENTADAS:**

#### **Classe:** `PIDController`
**Localização:** `src/controllab/core/controller_design.py` linha 10
**Como chamar:**
```python
from controllab.core.controller_design import PIDController
pid = PIDController()
```

##### **Método:** `__init__(self)`
**Entrada:** Nenhuma
**Saída:** Instância inicializada da classe
**Propósito:** Inicializa classe para design de controladores PID
**Contexto Teórico:** Implementa framework para **projeto de controladores PID (Cap 9.2)** - controlador mais usado na indústria pela simplicidade e efetividade

##### **Método:** `design_pid(self, plant: SymbolicTransferFunction, kp: Union[sp.Symbol, float] = None, ki: Union[sp.Symbol, float] = None, kd: Union[sp.Symbol, float] = None) -> SymbolicTransferFunction`
**Entrada:**
- `plant`: Planta a ser controlada
- `kp`: Ganho proporcional (default: símbolo K_p)
- `ki`: Ganho integral (default: símbolo K_i)
- `kd`: Ganho derivativo (default: símbolo K_d)
**Saída:** Controlador PID como SymbolicTransferFunction
**Propósito:** Projeta controlador PID C(s) = Kp + Ki/s + Kd*s
**Contexto Teórico:** Implementa **ação PID (Cap 9.2)**: Kp reduz erro de regime, Ki elimina erro regime permanente, Kd melhora resposta transitória mas amplifica ruído

##### **Método:** `tune_ziegler_nichols(self, plant: SymbolicTransferFunction, critical_gain: Union[sp.Symbol, float], critical_period: Union[sp.Symbol, float]) -> dict`
**Entrada:**
- `plant`: Planta
- `critical_gain`: Ganho crítico
- `critical_period`: Período crítico
**Saída:** Dicionário com parâmetros sintonizados para P, PI, PID
**Propósito:** Sintonia de Ziegler-Nichols aplicando regras clássicas
**Contexto Teórico:** Implementa **método empírico de Ziegler-Nichols**: baseado em teste de estabilidade limítrofe, fornece regras práticas para sintonia sem modelo detalhado da planta

---

#### **Classe:** `LeadLagCompensator`
**Localização:** `src/controllab/core/controller_design.py` linha 72
**Como chamar:**
```python
from controllab.core.controller_design import LeadLagCompensator
compensator = LeadLagCompensator()
```

##### **Método:** `__init__(self)`
**Entrada:** Nenhuma
**Saída:** Instância inicializada da classe
**Propósito:** Inicializa classe para design de compensadores lead-lag
**Contexto Teórico:** Framework para **compensação clássica (Caps 9.3, 11.4-11.5)** - técnica fundamental para moldar resposta em frequência

##### **Método:** `design_lead(self, desired_phase_margin: float, crossover_frequency: Union[sp.Symbol, float], alpha: Union[sp.Symbol, float] = None) -> SymbolicTransferFunction`
**Entrada:**
- `desired_phase_margin`: Margem de fase desejada (graus)
- `crossover_frequency`: Frequência de cruzamento
- `alpha`: Parâmetro do compensador (calculado automaticamente se None)
**Saída:** Compensador lead como SymbolicTransferFunction
**Propósito:** Projeta compensador lead para melhorar resposta transitória
**Contexto Teórico:** Implementa **compensação de avanço (Cap 11.4)**: adiciona avanço de fase máximo φmax = sin⁻¹((α-1)/(α+1)) na frequência ωm = √(zeropole) para melhorar margem de fase

##### **Método:** `design_lag(self, steady_state_error_requirement: float, beta: Union[sp.Symbol, float] = None) -> SymbolicTransferFunction`
**Entrada:**
- `steady_state_error_requirement`: Requisito de erro em regime
- `beta`: Parâmetro do compensador (β > 1)
**Saída:** Compensador lag como SymbolicTransferFunction
**Propósito:** Projeta compensador lag para melhorar erro em regime permanente
**Contexto Teórico:** Implementa **compensação de atraso (Cap 11.5)**: adiciona ganho DC β>1 sem afetar margem de fase, melhora erro regime permanente posicionando polo/zero em baixas frequências

---

#### **Classe:** `StateSpaceController`
**Localização:** `src/controllab/core/controller_design.py` linha 177
**Como chamar:**
```python
from controllab.core.controller_design import StateSpaceController
ss_controller = StateSpaceController()
```

##### **Método:** `__init__(self)`
**Entrada:** Nenhuma
**Saída:** Instância inicializada da classe
**Propósito:** Inicializa classe para design de controladores no espaço de estados
**Contexto Teórico:** Framework para **controle moderno (Cap 12)** - representação em espaço de estados permite controle MIMO e aplicação de métodos otimais

##### **Método:** `pole_placement(self, A: sp.Matrix, B: sp.Matrix, desired_poles: List[sp.Expr]) -> sp.Matrix`
**Entrada:**
- `A`: Matriz de estados
- `B`: Matriz de entrada
- `desired_poles`: Polos desejados
**Saída:** Matriz de ganho K ou None se não controlável
**Propósito:** Alocação de polos por realimentação de estados usando fórmula de Ackermann
**Contexto Teórico:** Implementa **realimentação de estados (Cap 12.2)**: se (A,B) controlável, existe K tal que det(sI-A+BK) tem polos arbitrários via fórmula de Ackermann

##### **Método:** `lqr_design(self, A: sp.Matrix, B: sp.Matrix, Q: sp.Matrix, R: sp.Matrix) -> dict`
**Entrada:**
- `A`: Matriz de estados
- `B`: Matriz de entrada
- `Q`: Matriz de peso dos estados
- `R`: Matriz de peso do controle
**Saída:** Dicionário com estrutura LQR
**Propósito:** Design LQR (Linear Quadratic Regulator) com equação de Riccati
**Contexto Teórico:** Implementa **controle ótimo LQR (Cap 12.5)**: minimiza J = ∫(x'Qx + u'Ru)dt via solução da equação de Riccati PA + A'P - PBR⁻¹B'P + Q = 0

##### **Método:** `_controllability_matrix(self, A: sp.Matrix, B: sp.Matrix) -> sp.Matrix`
**Entrada:** Matrizes A e B
**Saída:** Matriz de controlabilidade
**Propósito:** Calcula matriz de controlabilidade [B, AB, A²B, ..., A^(n-1)B]
**Contexto Teórico:** Implementa **teste de controlabilidade (Cap 12.2)**: sistema (A,B) é controlável se matriz de controlabilidade tem posto completo n

##### **Método:** `_evaluate_polynomial_at_matrix(self, poly: sp.Expr, matrix: sp.Matrix, variable: sp.Symbol) -> sp.Matrix`
**Entrada:** Polinômio, matriz e variável
**Saída:** Resultado da avaliação do polinômio na matriz
**Propósito:** Avalia polinômio em uma matriz (usado na fórmula de Ackermann)
**Contexto Teórico:** Implementa **teorema de Cayley-Hamilton**: toda matriz satisfaz sua própria equação característica, base da fórmula de Ackermann para alocação de polos

---

#### **Classe:** `ObserverDesign`
**Localização:** `src/controllab/core/controller_design.py` linha 344
**Como chamar:**
```python
from controllab.core.controller_design import ObserverDesign
observer = ObserverDesign()
```

##### **Método:** `__init__(self)`
**Entrada:** Nenhuma
**Saída:** Instância inicializada da classe
**Propósito:** Inicializa classe para design de observadores
**Contexto Teórico:** Framework para **observadores de estado (Cap 12.6)** - estimam estados não mensuráveis usando princípio de dualidade controlador-observador

##### **Método:** `luenberger_observer(self, A: sp.Matrix, C: sp.Matrix, desired_poles: List[sp.Expr]) -> sp.Matrix`
**Entrada:**
- `A`: Matriz de estados
- `C`: Matriz de saída
- `desired_poles`: Polos desejados do observador
**Saída:** Matriz de ganho L do observador ou None se não observável
**Propósito:** Design de observador de Luenberger usando dualidade
**Contexto Teórico:** Implementa **observador de Luenberger (Cap 12.6)**: se (A,C) observável, existe L tal que det(sI-A+LC) tem polos arbitrários por dualidade com controlabilidade

##### **Método:** `_observability_matrix(self, A: sp.Matrix, C: sp.Matrix) -> sp.Matrix`
**Entrada:** Matrizes A e C
**Saída:** Matriz de observabilidade
**Propósito:** Calcula matriz de observabilidade [C; CA; CA²; ...; CA^(n-1)]
**Contexto Teórico:** Implementa **teste de observabilidade (Cap 12.6)**: sistema (A,C) é observável se matriz de observabilidade tem posto completo n, dual da controlabilidade

---

### 📁 **Arquivo:** `src/controllab/core/stability_analysis.py`

### 🔧 **CLASSES IMPLEMENTADAS:**

#### **Classe:** `RouthHurwitzAnalyzer`
**Localização:** `src/controllab/core/stability_analysis.py` linha 11
**Como chamar:**
```python
from controllab.core.stability_analysis import RouthHurwitzAnalyzer
routh = RouthHurwitzAnalyzer()
```

##### **Método:** `__init__(self)`
**Entrada:** Nenhuma
**Saída:** Instância inicializada da classe
**Propósito:** Inicializa analisador de Routh-Hurwitz
**Contexto Teórico:** Framework para **critério de Routh-Hurwitz (Cap 6.3)** - método algébrico para determinar estabilidade sem calcular raízes do polinômio característico

##### **Método:** `analyze(self, characteristic_poly: sp.Expr, variable: sp.Symbol) -> dict`
**Entrada:**
- `characteristic_poly`: Polinômio característico
- `variable`: Variável do polinômio
**Saída:** Dicionário com resultado da análise
**Propósito:** Executa análise completa de Routh-Hurwitz
**Contexto Teórico:** Implementa **critério de Routh-Hurwitz (Cap 6.3)**: sistema estável sse todos elementos da primeira coluna da tabela têm mesmo sinal

##### **Método:** `_build_routh_table(self, coeffs: List) -> List[List]`
**Entrada:** Lista de coeficientes
**Saída:** Tabela de Routh como lista de listas
**Propósito:** Constrói tabela de Routh a partir dos coeficientes
**Contexto Teórico:** Implementa **construção da tabela de Routh**: arranjo sistemático dos coeficientes em linhas alternadas, com elementos calculados por determinantes

##### **Método:** `_analyze_stability(self, routh_table: List[List]) -> dict`
**Entrada:** Tabela de Routh
**Saída:** Resultado da análise de estabilidade
**Propósito:** Analisa estabilidade baseada na primeira coluna da tabela
**Contexto Teórico:** Aplica **critério de estabilidade**: número de mudanças de sinal na primeira coluna = número de polos no semiplano direito

---

#### **Classe:** `NyquistAnalyzer`
**Localização:** `src/controllab/core/stability_analysis.py` linha 151
**Como chamar:**
```python
from controllab.core.stability_analysis import NyquistAnalyzer
nyquist = NyquistAnalyzer()
```

##### **Método:** `__init__(self)`
**Entrada:** Nenhuma
**Saída:** Instância inicializada da classe
**Propósito:** Inicializa analisador de Nyquist
**Contexto Teórico:** Framework para **critério de Nyquist (Cap 10.3)** - método gráfico para análise de estabilidade via diagrama polar de malha aberta

##### **Método:** `analyze(self, open_loop_tf: SymbolicTransferFunction, frequency_range: Tuple[float, float] = (-3, 3)) -> dict`
**Entrada:**
- `open_loop_tf`: Função de transferência de malha aberta
- `frequency_range`: Faixa de frequências (log10)
**Saída:** Dicionário com resultado da análise
**Propósito:** Executa análise de Nyquist substituindo s por jω
**Contexto Teórico:** Implementa **critério de Nyquist (Cap 10.3)**: Z = P + N onde Z=polos malha fechada instáveis, P=polos malha aberta instáveis, N=envolvimentos de (-1,0)

##### **Método:** `_count_rhp_poles(self, poles: List[sp.Expr]) -> int`
**Entrada:** Lista de polos
**Saída:** Número de polos no semiplano direito
**Propósito:** Conta polos no semiplano direito para critério de Nyquist
**Contexto Teórico:** Calcula **P no critério de Nyquist**: número de polos de G(s)H(s) com parte real positiva, essencial para aplicação do critério

---

#### **Classe:** `BodeAnalyzer`
**Localização:** `src/controllab/core/stability_analysis.py` linha 216
**Como chamar:**
```python
from controllab.core.stability_analysis import BodeAnalyzer
bode = BodeAnalyzer()
```

##### **Método:** `__init__(self)`
**Entrada:** Nenhuma
**Saída:** Instância inicializada da classe
**Propósito:** Inicializa analisador de Bode
**Contexto Teórico:** Framework para **diagramas de Bode (Cap 10.1-10.2)** - representação logarítmica de magnitude e fase vs frequência para análise de sistemas

##### **Método:** `analyze(self, transfer_function: SymbolicTransferFunction) -> dict`
**Entrada:** `transfer_function`: Função de transferência para análise
**Saída:** Dicionário com magnitude, fase e margens
**Propósito:** Executa análise de Bode calculando magnitude e fase
**Contexto Teórico:** Implementa **análise de Bode (Cap 10.2)**: |G(jω)|dB = 20log|G(jω)|, ∠G(jω), identificando frequências de corte e margens de estabilidade

##### **Método:** `_calculate_margins(self, jw_expr: sp.Expr, omega: sp.Symbol) -> dict`
**Entrada:** Expressão G(jω) e variável de frequência
**Saída:** Expressões para cálculo de margens
**Propósito:** Fornece expressões simbólicas para cálculo de margens
**Contexto Teórico:** Calcula **margens de estabilidade (Cap 10.6)**: MG quando ∠G(jω) = -180°, MF quando |G(jω)| = 1, critérios de projeto

---

#### **Classe:** `RootLocusAnalyzer`
**Localização:** `src/controllab/core/stability_analysis.py` linha 317
**Como chamar:**
```python
from controllab.core.stability_analysis import RootLocusAnalyzer
root_locus = RootLocusAnalyzer()
```

##### **Método:** `__init__(self)`
**Entrada:** Nenhuma
**Saída:** Instância inicializada da classe
**Propósito:** Inicializa analisador do lugar das raízes
**Contexto Teórico:** Framework para **lugar das raízes (Cap 8)** - método gráfico mostrando como polos de malha fechada variam com ganho K

##### **Método:** `analyze(self, open_loop_tf: SymbolicTransferFunction, gain_symbol: sp.Symbol = None) -> dict`
**Entrada:**
- `open_loop_tf`: Função de transferência de malha aberta
- `gain_symbol`: Símbolo para o ganho variável (default: K)
**Saída:** Dicionário com análise do lugar das raízes
**Propósito:** Executa análise completa do lugar das raízes incluindo assintotas
**Contexto Teórico:** Implementa **análise do lugar das raízes (Cap 8.2-8.5)**: aplicação das regras de construção (número de ramos, assintotas, pontos de quebra) para 1+KG(s)H(s)=0
...
(O restante da documentação será gerado na próxima etapa, pois a saída é muito grande)
