import os
import uuid
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import logging
from typing import List, Dict, Optional, Tuple
import json
import threading
import timedelta

# Import our RAG components
from document_processor import CompanyDocumentProcessor
from pinecone_vector import PineconeVectorStore
from secured_chatbot import SecureRAGChatbot

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'default_secret_key')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'txt', 'pdf', 'docx', 'doc'}
app.config['MAX_CONTENT_LENGTH'] = 0.5 * 1024 * 1024  # 0.5MB default limit

CORS(app)

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

app_state = {
    'status': 'idle',  # idle, processing, ready, error
    'processing_progress': 0,
    'documents': [],
    'vector_store': None,
    'chatbot': None,
    'message_count': 0,
    'created_at': datetime.now().isoformat(),
}

original_app_state = app_state.copy()

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    global app_state, original_app_state
    # if the status is not idle and created_at is more than 30 minutes ago, reset the app state
    if app_state['status'] != 'idle' and app_state['created_at'] < (datetime.now() - timedelta(minutes=30)).isoformat():
        # Remove uploaded files
        for doc in app_state['documents']:
            try:
                os.remove(doc)
            except:
                pass
        app_state = original_app_state.copy()
        app_state['created_at'] = datetime.now().isoformat()

    return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
def upload_documents():
    """Handle document upload"""

    if 'documents' not in request.files:
        return jsonify({'error': 'No documents provided'}), 400
    
    files = request.files.getlist('documents')
    uploaded_files = []

    MAX_FILE_SIZE = app.config['MAX_CONTENT_LENGTH']
    MAX_TOTAL_SIZE = 2 * 1024 * 1024  # 2MB total limit
    MAX_FILES = 5

    # Validate number of files
    if len(files) > MAX_FILES:
        return jsonify({'error': f'Too many files. Maximum {MAX_FILES} files allowed.'}), 400
    
    # Clean up any existing documents first
    for old_doc in app_state['documents']:
        try:
            if os.path.exists(old_doc):
                os.remove(old_doc)
        except:
            pass
    app_state['documents'] = []
    
    total_size = 0

    for file in files:
        if file and file.filename:
            if not allowed_file(file.filename):
                return jsonify({
                    'error': f'File type not allowed: {file.filename}. Allowed types: {", ".join(app.config["ALLOWED_EXTENSIONS"])}'
                }), 400

            # Check individual file size
            file.seek(0, os.SEEK_END)
            file_size = file.tell()
            file.seek(0)  # Reset file pointer

            if file_size > MAX_FILE_SIZE:
                return jsonify({
                    'error': f'File too large: {file.filename}. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB'
                }), 400
            

            if file_size == 0:
                return jsonify({
                    'error': f'Empty file: {file.filename}'
                }), 400
            
            total_size += file_size
            # Check total size
            if total_size > MAX_TOTAL_SIZE:
                return jsonify({
                    'error': f'Total file size exceeds limit of {MAX_TOTAL_SIZE // (1024*1024)}MB'
                }), 400
            
            
            try:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                uploaded_files.append({
                    'filename': file.filename,
                    'filepath': filepath,
                    'size': file_size
                })
                app_state['documents'].append(filepath)
                logger.info(f"Saved file: {filename} ({file_size} bytes)")
            except Exception as e:
                logger.error(f"Error saving file {filename}: {str(e)}")
                # Clean up any uploaded files on error
                for uploaded in uploaded_files:
                    try:
                        os.remove(uploaded['filepath'])
                    except:
                        pass
                return jsonify({'error': f'Failed to save file: {file.filename}'}), 500
    
    if not uploaded_files:
        return jsonify({'error': 'No valid files uploaded'}), 400
    
    # Start processing in background
    # Update status to processing
    app_state['status'] = 'processing'
    app_state['processing_progress'] = 10
    
    try:
        # Process documents in a separate thread
        threading.Thread(
            target=process_documents,
            daemon=True
        ).start()
    except Exception as e:
            logger.error(f"Error processing documents: {str(e)}")
            app_state['status'] = 'error'
            return jsonify({'error': 'Failed to process documents'}), 500
    
    return jsonify({
        'success': True,
        'uploaded': uploaded_files,
        'total_documents': len(app_state['documents']),
        'total_size': total_size
    })


def process_documents():
    """Process documents in the background"""

    try:
        # Update progress
        app_state['processing_progress'] = 20
        
        # Initialize document processor
        processor = CompanyDocumentProcessor()
        
        # Load and chunk documents
        documents = processor.load_documents(app_state['documents'])
        app_state['processing_progress'] =  40
        
        chunks = processor.chunk_documents(documents)
        app_state['processing_progress'] =  60
        
        # Initialize Pinecone vector store
        vector_store = PineconeVectorStore()
        
        # Create embeddings and index
        vector_store.add_documents(chunks)
        app_state['processing_progress'] = 80
        
        # Initialize chatbot
        chatbot = SecureRAGChatbot(vector_store=vector_store)
        
        # Store in app state
        app_state['vector_store'] = vector_store
        app_state['chatbot'] = chatbot
        
        # Mark as ready
        app_state['status'] = 'ready'
        app_state['processing_progress'] = 100
        logger.info("Successfully processed documents")
        
    except Exception as e:
        logger.error(f"Error in process_documents: {str(e)}")
        app_state['status'] = 'error'
        raise

@app.route('/api/status', methods=['GET'])
def get_status():
    return jsonify({
        'status': app_state['status'],
        'progress': app_state['processing_progress'],
        'document_count': len(app_state['documents']),
        'message_count': app_state['message_count']
    })

@app.route('/api/chat', methods=['POST'])
def chat():
    # Check if chatbot is ready
    if app_state['status'] != 'ready':
        return jsonify({
            'error': 'Chatbot not ready. Please upload documents first.',
            'status': app_state['status']
        }), 400
    
    data = request.json
    message = data.get('message', '').strip()
    
    if not message:
        return jsonify({'error': 'No message provided'}), 400
    
    # Get chatbot
    chatbot = app_state['chatbot']
    if not chatbot:
        return jsonify({'error': 'Chatbot not initialized'}), 500
    
    try:
        # Get response
        response_data = chatbot.get_response(message)
        
        # Update message count
        app_state['message_count'] += 1
        
        # Format response
        return jsonify({
            'success': True,
            'response': response_data['response'],
            'sources': response_data.get('sources', []),
            'confidence': response_data.get('confidence', 0),
            'query_id': response_data.get('query_id', ''),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        return jsonify({'error': 'Failed to generate response'}), 500

@app.route('/api/reset', methods=['POST'])
def reset_session():
    """Reset the app state"""
    if app_state:
        # Remove uploaded files
        for doc in app_state['documents']:
            try:
                os.remove(doc)
            except:
                pass
        
    # Reset app state
    app_state = original_app_state.copy()
    app_state['created_at'] = datetime.now().isoformat()
    
    return jsonify({'success': True })

# Error handlers
@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal error: {str(error)}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    if os.environ.get('ENVIRONMENT') == 'production':
        # Production settings
        port = int(os.environ.get('PORT', 8080))
        app.run(host='0.0.0.0', port=port, debug=False)
    else:
        app.run(debug=True, port=5001)