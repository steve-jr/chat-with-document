# Enterprise Document Chatbot

AI-powered text-base search and chatbot for custom enterprise dataset.

## Features

- Search for detailed content from uploaded documents
- Document department filtering
- Drag-and-drop file upload
- Real-time similarity, accuracy & confidence scoring

## Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate 

# Install dependencies
pip install --no-cache-dir -r requirements.txt

# Set environment variables
cp .env.example .env

# Run the app
python app.py