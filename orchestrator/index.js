const express = require('express');
const WebSocket = require('ws');

const app = express();
const port = 3000;
const pythonWsUrl = 'ws://localhost:8765';

// Middleware to parse JSON bodies
app.use(express.json());

/**
 * @api {post} /api/query Forwards a query to the Python backend
 * @apiName PostQuery
 * @apiGroup Orchestrator
 *
 * @apiBody {String} query The user's input string (e.g., "G(s) = 1/(s+1)").
 *
 * @apiSuccess {Object} response The JSON response from the Python backend.
 */
app.post('/api/query', (req, res) => {
  const userInput = req.body.query;
  if (!userInput) {
    return res.status(400).json({ error: 'Missing "query" in request body' });
  }

  console.log(`[Orchestrator] Received query: "${userInput}". Forwarding to Python backend...`);

  const ws = new WebSocket(pythonWsUrl);

  // Handle WebSocket connection opening
  ws.on('open', () => {
    console.log('[Orchestrator] Connected to Python WebSocket server.');
    const requestPayload = {
      type: 'analyze',
      input: userInput,
      timestamp: Date.now()
    };
    ws.send(JSON.stringify(requestPayload));
    console.log('[Orchestrator] Sent query to Python.');
  });

  // Handle messages received from the Python server
  ws.on('message', (data) => {
    console.log('[Orchestrator] Received response from Python.');
    try {
      const parsedData = JSON.parse(data);
      // Send the response back to the original HTTP client
      res.status(200).json(parsedData);
    } catch (error) {
      console.error('[Orchestrator] Error parsing JSON from Python:', error);
      res.status(500).json({ error: 'Failed to parse response from backend' });
    }
    // Close the WebSocket connection once the job is done
    ws.close();
  });

  // Handle WebSocket errors
  ws.on('error', (error) => {
    console.error(`[Orchestrator] WebSocket error: ${error.message}`);
    // If the connection fails, it's likely the Python server isn't running.
    res.status(502).json({
        error: 'Could not connect to the Python backend.',
        details: 'Ensure the Python WebSocket server is running.'
    });
  });

  // Handle WebSocket connection closing
  ws.on('close', () => {
    console.log('[Orchestrator] Disconnected from Python WebSocket server.');
  });
});

app.listen(port, () => {
  console.log(`[Orchestrator] Server listening at http://localhost:${port}`);
});
