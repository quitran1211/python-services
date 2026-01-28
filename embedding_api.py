from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Load model (this takes time on first run)
logger.info("Loading embedding model...")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
logger.info("Model loaded successfully!")

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "ok"})

@app.route('/embed', methods=['POST'])
def embed():
    """Generate embedding for single text"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({"error": "Text is required"}), 400
        
        embedding = model.encode(text, convert_to_numpy=True)
        
        return jsonify({
            "embedding": embedding.tolist()
        })
    
    except Exception as e:
        logger.error(f"Embed error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/embed-batch', methods=['POST'])
def embed_batch():
    """Generate embeddings for multiple texts"""
    try:
        data = request.get_json()
        texts = data.get('texts', [])
        
        if not texts:
            return jsonify({"error": "Texts array is required"}), 400
        
        embeddings = model.encode(texts, convert_to_numpy=True)
        
        return jsonify({
            "embeddings": embeddings.tolist()
        })
    
    except Exception as e:
        logger.error(f"Batch embed error: {str(e)}")
        return jsonify({"error": str(e)}), 500
@app.route("/rag/search", methods=["POST"])
def rag_search():
    data = request.json
    query = data.get("query")

    if not query:
        return jsonify({"error": "query required"}), 400

    # 1. embedding câu hỏi
    query_embedding = model.encode(query).tolist()

    # 2. trả về embedding (tạm thời)
    return jsonify({
        "embedding": query_embedding
    })
if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 8001))
    app.run(host='0.0.0.0', port=port, debug=False)
