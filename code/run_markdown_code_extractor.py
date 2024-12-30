import json
from jet.code import MarkdownCodeExtractor

MARKDOWN = """
### File Path: `README.md`
```markdown
# Project Overview

This project is a React frontend and Node.js backend application that displays a page split into two sides: a PDF viewer and an AI Chatbot.

## Features

1. **PDF Viewer**: A side of the page dedicated to displaying PDF files.
2. **AI Chatbot**: An interactive chat interface on the other side for live messaging.
3. **APIs**:
   - For updating and receiving a PDF link.
   - For sending and receiving live chat messages.

## Installation

To set up this project, follow these steps:

1. Clone the repository.
2. Run `setup.sh` to install dependencies.
3. Start both frontend and backend servers using `run.sh`.
```

### File Path: `setup.sh`
```bash
#!/bin/bash

# Install Node.js dependencies
cd frontend
npm install

cd ../backend
npm install
```

### File Path: `run.sh`
```bash
#!/bin/bash

# Start the backend server
node backend/server.js &

# Start the frontend development server
npm start frontend
```

### File Path: `frontend/src/App.tsx`
```tsx
import React, { useState } from 'react';
import './App.css';

function App() {
  const [pdfUrl, setPdfUrl] = useState<string>('');
  const [chatMessages, setChatMessages] = useState<string[]>([]);

  const handlePdfUpdate = (url: string) => {
    setPdfUrl(url);
  };

  return (
    <div className="flex h-screen">
      <div className="w-1/2 bg-gray-100 p-4">
        <h2>PDF Viewer</h2>
        <iframe src={pdfUrl} title="PDF" width="100%" height="100%"></iframe>
      </div>
      <div className="w-1/2 bg-white p-4">
        <h2>AI Chatbot</h2>
        <div id="chat-messages" className="max-h-96 overflow-y-scroll border p-2 mb-4">
          {chatMessages.map((message, index) => (
            <div key={index} className="mb-2">{message}</div>
          ))}
        </div>
        <input
          type="text"
          id="chat-input"
          placeholder="Type a message..."
          className="w-full px-4 py-2 mb-4 border rounded"
        />
        <button onClick={() => {
          const input = document.getElementById('chat-input') as HTMLInputElement;
          setChatMessages([...chatMessages, input.value]);
          input.value = '';
        }} className="px-4 py-2 bg-blue-500 text-white rounded">Send</button>
      </div>
    </div>
  );
}

export default App;
```

### File Path: `frontend/src/App.css`
```css
@tailwind base;
@tailwind components;
@tailwind utilities;

body {
  margin: 0;
  font-family: Arial, sans-serif;
}
```

### File Path: `backend/server.js`
```javascript
const express = require('express');
const bodyParser = require('body-parser');
const axios = require('axios');

const app = express();
app.use(bodyParser.json());

let pdfUrl = '';
let chatMessages = [];

app.post('/update-pdf', (req, res) => {
  const { url } = req.body;
  if (url) {
    pdfUrl = url;
    res.send({ message: 'PDF URL updated successfully' });
  } else {
    res.status(400).send({ message: 'Invalid PDF URL' });
  }
});

app.get('/pdf-url', (req, res) => {
  res.json({ pdfUrl });
});

app.post('/chat-message', async (req, res) => {
  const { message } = req.body;
  if (message) {
    chatMessages.push(message);
    try {
      // Assuming there's an AI model that can respond to chat messages
      const response = await axios.get('https://api.ai.com/chat', {
        params: { message },
        headers: { 'Authorization': 'Bearer YOUR_AI_API_KEY' }
      });
      const aiResponse = response.data.response;
      chatMessages.push(aiResponse);
      res.send({ chatMessages });
    } catch (error) {
      res.status(500).send({ error: 'Failed to get AI response' });
    }
  } else {
    res.status(400).send({ message: 'Invalid message' });
  }
});

app.get('/chat-messages', (req, res) => {
  res.json({ chatMessages });
});

const PORT = process.env.PORT || 3001;
app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});
```

### File Path: `backend/package.json`
```json
{
  "name": "backend",
  "version": "1.0.0",
  "description": "",
  "main": "server.js",
  "scripts": {
    "start": "node server.js"
  },
  "dependencies": {
    "axios": "^0.21.1",
    "body-parser": "^1.19.0",
    "express": "^4.17.1"
  }
}
```

### File Path: `frontend/package.json`
```json
{
  "name": "frontend",
  "version": "1.0.0",
  "description": "",
  "main": "index.js",
  "scripts": {
    "start": "react-scripts start"
  },
  "dependencies": {
    "react": "^17.0.2",
    "react-dom": "^17.0.2",
    "react-scripts": "4.0.3",
    "tailwindcss": "^2.2.19"
  }
}
```

### File Path: `frontend/src/index.tsx`
```tsx
import React from 'react';
import ReactDOM from 'react-dom';
import './index.css';
import App from './App';

ReactDOM.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
  document.getElementById('root')
);
```

### File Path: `frontend/src/index.css`
```css
@tailwind base;
@tailwind components;
@tailwind utilities;

#root {
  width: 100vw;
  height: 100vh;
}
```

This codebase provides a basic implementation of the desired features. You can further customize and enhance it as needed.
"""

if __name__ == "__main__":
    extractor = MarkdownCodeExtractor()
    result = extractor.extract_code_blocks(MARKDOWN)
    print(json.dumps(result, indent=2))
