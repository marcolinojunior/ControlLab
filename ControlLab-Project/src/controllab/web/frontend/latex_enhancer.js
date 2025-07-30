
/**
 * 🧮 ControlLab LaTeX Frontend Enhancer
 * 
 * Adiciona funcionalidades de renderização LaTeX ao frontend
 * usando MathJax para visualização matemática aprimorada
 */

class LaTeXRenderer {
    constructor() {
        this.mathJaxReady = false;
        this.initializeMathJax();
    }
    
    initializeMathJax() {
        // Configura MathJax se não estiver configurado
        if (typeof window.MathJax === 'undefined') {
            window.MathJax = {
                tex: {
                    inlineMath: [['$', '$'], ['\\(', '\\)']],
                    displayMath: [['$$', '$$'], ['\\[', '\\]']],
                    processEscapes: true,
                    processEnvironments: true
                },
                options: {
                    skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre']
                },
                startup: {
                    ready: () => {
                        MathJax.startup.defaultReady();
                        this.mathJaxReady = true;
                        console.log('✅ MathJax carregado para LaTeX rendering');
                    }
                }
            };
        }
    }
    
    /**
     * Renderiza LaTeX em elemento específico
     * @param {string} elementId - ID do elemento
     * @param {string} latexContent - Conteúdo LaTeX
     */
    renderLatex(elementId, latexContent) {
        const element = document.getElementById(elementId);
        if (!element) return;
        
        // Adiciona delimitadores LaTeX se necessário
        const formattedLatex = this.formatLatexContent(latexContent);
        element.innerHTML = formattedLatex;
        
        // Re-renderiza MathJax
        if (this.mathJaxReady && window.MathJax) {
            MathJax.typesetPromise([element]).catch((err) => {
                console.warn('Erro na renderização LaTeX:', err);
                // Fallback para texto simples
                element.innerHTML = `<code>${latexContent}</code>`;
            });
        }
    }
    
    /**
     * Formata conteúdo LaTeX com delimitadores apropriados
     * @param {string} latex - Conteúdo LaTeX
     * @returns {string} Conteúdo formatado
     */
    formatLatexContent(latex) {
        // Se já tem delimitadores, retorna como está
        if (latex.includes('$$') || latex.includes('\\[')) {
            return latex;
        }
        
        // Adiciona delimitadores para display math
        return `$$${latex}$$`;
    }
    
    /**
     * Renderiza resposta completa com LaTeX
     * @param {Object} response - Resposta do servidor
     */
    renderResponse(response) {
        // Renderiza função de transferência principal
        if (response.latex && response.latex.transfer_function) {
            this.renderLatex('main-transfer-function', response.latex.transfer_function);
        }
        
        // Renderiza polos
        if (response.latex && response.latex.poles) {
            this.renderLatex('system-poles', response.latex.poles);
        }
        
        // Renderiza zeros
        if (response.latex && response.latex.zeros) {
            this.renderLatex('system-zeros', response.latex.zeros);
        }
        
        // Renderiza especificações
        if (response.latex && response.latex.specifications) {
            this.renderSpecifications(response.latex.specifications);
        }
    }
    
    /**
     * Renderiza especificações do sistema
     * @param {Object} specs - Especificações em LaTeX
     */
    renderSpecifications(specs) {
        const specsContainer = document.getElementById('specifications-latex');
        if (!specsContainer) return;
        
        let specsHtml = '<div class="specs-latex-container">';
        
        for (const [key, latex] of Object.entries(specs)) {
            specsHtml += `
                <div class="spec-item">
                    <span class="spec-latex">$${latex}$</span>
                </div>
            `;
        }
        
        specsHtml += '</div>';
        specsContainer.innerHTML = specsHtml;
        
        // Re-renderiza MathJax
        if (this.mathJaxReady && window.MathJax) {
            MathJax.typesetPromise([specsContainer]);
        }
    }
    
    /**
     * Adiciona expressão LaTeX ao histórico visual
     * @param {string} expression - Expressão LaTeX
     * @param {string} description - Descrição da expressão
     */
    addToHistory(expression, description) {
        const historyContainer = document.getElementById('latex-history');
        if (!historyContainer) return;
        
        const historyItem = document.createElement('div');
        historyItem.className = 'latex-history-item';
        historyItem.innerHTML = `
            <div class="latex-expression">$$${expression}$$</div>
            <div class="latex-description">${description}</div>
            <div class="latex-timestamp">${new Date().toLocaleTimeString()}</div>
        `;
        
        historyContainer.prepend(historyItem);
        
        // Re-renderiza MathJax para o novo item
        if (this.mathJaxReady && window.MathJax) {
            MathJax.typesetPromise([historyItem]);
        }
    }
}

// Instância global do renderizador LaTeX
window.latexRenderer = new LaTeXRenderer();

// Extensão da classe ControlLabInterface para suporte LaTeX
if (typeof window.ControlLabInterface !== 'undefined') {
    const originalProcessResponse = window.ControlLabInterface.prototype.processResponse;
    
    window.ControlLabInterface.prototype.processResponse = function(response) {
        // Processa resposta original
        originalProcessResponse.call(this, response);
        
        // Adiciona renderização LaTeX
        if (response.latex) {
            window.latexRenderer.renderResponse(response);
            
            // Adiciona ao histórico se é uma nova análise
            if (response.latex.transfer_function) {
                window.latexRenderer.addToHistory(
                    response.latex.transfer_function,
                    'Função de Transferência Analisada'
                );
            }
        }
    };
}

console.log('✅ LaTeX Frontend Enhancer carregado');
