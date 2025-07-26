document.addEventListener('DOMContentLoaded', () => {
    const input = document.getElementById('diff-eq-input') || document.getElementById('tf-input');
    const submitBtn = document.getElementById('submit-btn');
    const stepsDiv = document.getElementById('steps');
    const plotDiv = document.getElementById('plot');

    const socket = new WebSocket('ws://localhost:8765');

    socket.onopen = () => {
        console.log('Conectado ao servidor WebSocket.');
    };

    socket.onmessage = (event) => {
        const data = JSON.parse(event.data);

        if (data.type === 'laplace_result') {
            stepsDiv.innerHTML = '';
            data.steps.forEach(step => {
                const stepElement = document.createElement('p');
                stepElement.innerHTML = step;
                stepsDiv.appendChild(stepElement);
            });
            MathJax.typeset();
        } else if (data.type === 'analysis_result') {
            stepsDiv.innerHTML = '';
            data.steps.forEach(step => {
                const stepElement = document.createElement('p');
                stepElement.innerHTML = step;
                stepsDiv.appendChild(stepElement);
            });
            MathJax.typeset();
            if (data.plot) {
                Plotly.newPlot(plotDiv, data.plot.data, data.plot.layout);
            }
        } else if (data.type === 'error') {
            stepsDiv.innerHTML = `<p style="color: red;">${data.message}</p>`;
        }
    };

    socket.onclose = () => {
        console.log('Desconectado do servidor WebSocket.');
    };

    submitBtn.addEventListener('click', () => {
        const value = input.value;
        if (value) {
            let message;
            if (input.id === 'diff-eq-input') {
                message = {
                    type: 'analyze_laplace',
                    payload: { diff_eq: value }
                };
            } else {
                message = {
                    type: 'analyze_tf',
                    payload: { tf: value }
                };
            }
            socket.send(JSON.stringify(message));
        }
    });
});
