const express = require('express');
const cors = require('cors');
const multer = require('multer');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

const app = express();
const port = process.env.PORT || 5000;

app.use(cors());
app.use(express.json());

// Set up multer for file uploads
const upload = multer({ dest: 'uploads/' });

const scriptPath = path.join(__dirname, 'controllers/skin_predict_api.py');

// Try to find a python executable
const pythonCandidates = [
  process.env.PYTHON_PATH,
  process.platform === 'win32' ? 'python' : 'python3',
  'py'
].filter(Boolean);

const pythonPath = pythonCandidates[0]; // Let's just use 'python' assuming it's in PATH or handled by OS

console.log(`Starting persistent Python process: ${pythonPath} ${scriptPath}`);

const pythonProcess = spawn(pythonPath, [scriptPath], {
  cwd: path.dirname(scriptPath),
  windowsHide: true,
});

let pythonReady = false;
let pendingRequests = [];

pythonProcess.stdout.on('data', (data) => {
  const lines = data.toString().split('\n');
  for (let line of lines) {
    line = line.trim();
    if (!line) continue;
    
    try {
      const parsed = JSON.parse(line);
      
      if (parsed.status === 'ready') {
        pythonReady = true;
        console.log("Python model is loaded and ready.");
        continue;
      }
      
      if (pendingRequests.length > 0) {
        const req = pendingRequests.shift();
        req.resolve(parsed);
      }
    } catch (e) {
      console.log("Python stdout (non-JSON):", line);
    }
  }
});

pythonProcess.stderr.on('data', (data) => {
  console.error(`Python stderr: ${data}`);
});

pythonProcess.on('close', (code) => {
  console.log(`Python process exited with code ${code}`);
  pythonReady = false;
  while(pendingRequests.length > 0) {
    pendingRequests.shift().reject(new Error("Python process exited prematurely."));
  }
});

// A helper function to send a command to Python and wait for a response
function predictWithPython(imagePath) {
  return new Promise((resolve, reject) => {
    if (!pythonReady) {
      return reject(new Error("Python model is not ready yet. Please try again in a few seconds."));
    }
    pendingRequests.push({ resolve, reject });
    pythonProcess.stdin.write(`${imagePath}\n`);
  });
}

app.post('/predict', upload.single('image'), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No image uploaded' });
  }

  const imagePath = path.resolve(req.file.path);
  
  try {
    const result = await predictWithPython(imagePath);
    
    // Clean up the uploaded file
    if (fs.existsSync(imagePath)) {
      fs.unlinkSync(imagePath); 
    }
    
    if (result.error) {
      return res.status(500).json({ error: result.error });
    }
    
    return res.status(200).json({ success: true, result });
  } catch (error) {
    // Clean up the uploaded file
    if (fs.existsSync(imagePath)) {
      fs.unlinkSync(imagePath); 
    }
    return res.status(500).json({ error: error.message });
  }
});

app.get('/', (req, res) => {
  res.send('Hello World!');
});

app.listen(port, () => {
  console.log(`AI Skin Disease Microservice is running on port ${port}`);
});
