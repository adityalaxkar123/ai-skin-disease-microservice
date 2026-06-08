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

app.post('/predict', upload.single('image'), (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No image uploaded' });
  }

  const imagePath = path.resolve(req.file.path);
  const scriptPath = path.join(__dirname, 'controllers/skin_predict_api.py');

  // Try to find a python executable
  const pythonCandidates = [
    process.env.PYTHON_PATH,
    process.platform === 'win32' ? 'python' : 'python3',
    'py'
  ].filter(Boolean);

  const pythonPath = pythonCandidates[0]; // Let's just use 'python' assuming it's in PATH or handled by OS

  const args = [scriptPath, imagePath];
  console.log(`Running: ${pythonPath} ${args.join(' ')}`);

  const child = spawn(pythonPath, args, {
    cwd: path.dirname(scriptPath),
    windowsHide: true,
  });

  let stdout = '';
  let stderr = '';

  child.stdout.on('data', (d) => {
    stdout += d.toString();
  });

  child.stderr.on('data', (d) => {
    stderr += d.toString();
  });

  child.on('error', (error) => {
    fs.unlinkSync(imagePath); // Clean up the uploaded file
    return res.status(500).json({
      error: `Execution failed: ${error.message}`,
    });
  });

  child.on('close', (code) => {
    fs.unlinkSync(imagePath); // Clean up the uploaded file

    if (code !== 0) {
      console.error(`Python script error: ${stderr}`);
      return res.status(500).json({
        error: stderr ? `Python script error: ${stderr}` : `Python script exited with code ${code}`,
      });
    }

    try {
      const parsed = JSON.parse(stdout.trim());
      if (parsed.error) {
        return res.status(500).json({ error: parsed.error });
      }
      return res.status(200).json({ success: true, result: parsed });
    } catch (parseErr) {
      console.error("Failed to parse model output:", parseErr, "Raw:", stdout);
      return res.status(500).json({
        error: "Failed to parse model output.",
        raw: stdout.trim(),
      });
    }
  });
});

app.get('/', (req, res) => {
  res.send('Hello World!');
})

app.listen(port, () => {
  console.log(`AI Skin Disease Microservice is running on port ${port}`);
});
