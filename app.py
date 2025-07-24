import os
import uuid
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
from werkzeug.utils import secure_filename
import logging
from typing import List, Dict, Optional, Tuple
import json
import threading


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
app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024  # 1MB default limit

CORS(app)

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('sessions', exist_ok=True)

# Global variables for session management
user_sessions = {}

class SessionManager:
    """Manages user sessions and their associated vector stores"""
    
    def __init__(self):
        self.sessions = {}
    
    def create_session(self, session_id: str) -> Dict:
        """Create a new session"""
        self.sessions[session_id] = {
            'id': session_id,
            'created_at': datetime.now().isoformat(),
            'status': 'idle',  # idle, processing, ready, error
            'documents': [],
            'vector_store': None,
            'chatbot': None,
            'message_count': 0,
            'processing_progress': 0
        }
        return self.sessions[session_id]
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get session data"""
        return self.sessions.get(session_id)
    
    def update_status(self, session_id: str, status: str, progress: int = 0):
        """Update session status"""
        if session_id in self.sessions:
            self.sessions[session_id]['status'] = status
            self.sessions[session_id]['processing_progress'] = progress
    
    def cleanup_old_sessions(self, max_age_hours: int = 24):
        """Clean up old sessions"""
        # Implementation for cleaning up old sessions
        pass
    
    def get_all_sessions(self) -> List[Dict]:
        """Get all sessions"""
        return list(self.sessions.values())

session_manager = SessionManager()

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    # Generate new session ID if not exists
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())

    if not session.get('session_id'):
        session['session_id'] = str(uuid.uuid4())

    if not session_manager.get_session(session['session_id']):
        session_manager.create_session(session['session_id'])
    
    return render_template('index.html', session_id=session['session_id'])

@app.route('/api/upload', methods=['POST'])
def upload_documents():
    """Handle document upload"""
    session_id = session.get('session_id')
    if not session_id:
        return jsonify({'error': 'No session found'}), 400
    
    user_session = session_manager.get_session(session_id)
    if not user_session:
        return jsonify({'error': 'Invalid session'}), 400
    
    if 'documents' not in request.files:
        return jsonify({'error': 'No documents provided'}), 400
    
    files = request.files.getlist('documents')
    uploaded_files = []

    MAX_FILE_SIZE = app.config['MAX_CONTENT_LENGTH']  # 1MB default
    MAX_TOTAL_SIZE = 10 * 1024 * 1024  # 10MB total limit
    MAX_FILES = 10

    # Validate number of files
    if len(files) > MAX_FILES:
        return jsonify({'error': f'Too many files. Maximum {MAX_FILES} files allowed.'}), 400
    
    # Clean up any existing documents first
    for old_doc in user_session.get('documents', []):
        try:
            if os.path.exists(old_doc):
                os.remove(old_doc)
        except:
            pass
    user_session['documents'] = []
    
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
                # Add session ID to filename to keep sessions separate
                filename = f"{session_id}_{filename}"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                uploaded_files.append({
                    'filename': file.filename,
                    'filepath': filepath,
                    'size': file_size
                })
                user_session['documents'].append(filepath)
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
    session_manager.update_status(session_id, 'processing', 10)
    
    try:
        # Process documents in a separate thread
        processing_thread = threading.Thread(
            target=process_documents_for_session,
            args=(session_id,),
            daemon=True
        )
        processing_thread.start()
    except Exception as e:
            logger.error(f"Error processing documents: {str(e)}")
            session_manager.update_status(session_id, 'error')
            return jsonify({'error': 'Failed to process documents'}), 500
    
    return jsonify({
        'success': True,
        'uploaded': uploaded_files,
        'total_documents': len(user_session['documents']),
        'total_size': total_size
    })

    # # Start processing in background
    # if uploaded_files:
    #     # Update status to processing
    #     session_manager.update_status(session_id, 'processing', 10)
        
    #     # Process documents (simplified - in production, do this async)
    #     try:
    #         process_documents_for_session(session_id)
    #     except Exception as e:
    #         logger.error(f"Error processing documents: {str(e)}")
    #         session_manager.update_status(session_id, 'error')
    #         return jsonify({'error': 'Failed to process documents'}), 500
    
    # return jsonify({
    #     'success': True,
    #     'uploaded': uploaded_files,
    #     'total_documents': len(user_session['documents'])
    # })

def process_documents_for_session(session_id: str):
    """Process documents for a specific session"""
    user_session = session_manager.get_session(session_id)
    if not user_session:
        return
    
    try:
        # Update progress
        session_manager.update_status(session_id, 'processing', 20)
        
        # Initialize document processor
        processor = CompanyDocumentProcessor()
        
        # Load and chunk documents
        documents = processor.load_documents(user_session['documents'])
        session_manager.update_status(session_id, 'processing', 40)
        
        chunks = processor.chunk_documents(documents)
        session_manager.update_status(session_id, 'processing', 60)
        
        # Initialize Pinecone vector store
        vector_store = PineconeVectorStore(
            session_id=session_id
        )
        
        # Create embeddings and index
        vector_store.add_documents(chunks)
        session_manager.update_status(session_id, 'processing', 80)
        
        # Initialize chatbot
        chatbot = SecureRAGChatbot(vector_store=vector_store)
        
        # Store in session
        user_session['vector_store'] = vector_store
        user_session['chatbot'] = chatbot
        
        # Mark as ready
        session_manager.update_status(session_id, 'ready', 100)
        logger.info(f"Successfully processed documents for session {session_id}")
        
    except Exception as e:
        logger.error(f"Error in process_documents_for_session: {str(e)}")
        session_manager.update_status(session_id, 'error')
        raise

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get current session status"""
    session_id = session.get('session_id')
    if not session_id:
        return jsonify({'error': 'No session found'}), 400
    
    user_session = session_manager.get_session(session_id)
    if not user_session:
        return jsonify({'error': 'Invalid session'}), 400
    
    return jsonify({
        'status': user_session['status'],
        'progress': user_session['processing_progress'],
        'document_count': len(user_session['documents']),
        'message_count': user_session['message_count']
    })

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages"""
    session_id = session.get('session_id')
    if not session_id:
        return jsonify({'error': 'No session found'}), 400
    
    user_session = session_manager.get_session(session_id)
    if not user_session:
        return jsonify({'error': 'Invalid session'}), 400
    
    # Check if chatbot is ready
    if user_session['status'] != 'ready':
        return jsonify({
            'error': 'Chatbot not ready. Please upload documents first.',
            'status': user_session['status']
        }), 400
    
    data = request.json
    message = data.get('message', '').strip()
    
    if not message:
        return jsonify({'error': 'No message provided'}), 400
    
    # Get chatbot
    chatbot = user_session['chatbot']
    if not chatbot:
        return jsonify({'error': 'Chatbot not initialized'}), 500
    
    try:
        # Get response
        response_data = chatbot.get_response(message)
        
        # Update message count
        user_session['message_count'] += 1
        
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
    """Reset the current session"""
    session_id = session.get('session_id')
    if session_id:
        user_session = session_manager.get_session(session_id)
        if user_session:
            # Clean up vector store
            if user_session['vector_store']:
                try:
                    user_session['vector_store'].delete_namespace(session_id)
                except:
                    pass
            
            # Remove uploaded files
            for doc in user_session['documents']:
                try:
                    os.remove(doc)
                except:
                    pass
        
        # Create new session
        session['session_id'] = str(uuid.uuid4())
        session_manager.create_session(session['session_id'])
    
    return jsonify({'success': True, 'new_session_id': session['session_id']})

# Error handlers
@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal error: {str(error)}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)