from flask import Flask, request, render_template, jsonify, send_from_directory
import os
from image_search import ImageSearchEngine
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

IMAGES_FOLDER = 'coco_images_resized'
app.config['IMAGES_FOLDER'] = IMAGES_FOLDER

try:
    search_engine = ImageSearchEngine(
        model_name='ViT-B/32',
        pretrained='openai',
        embeddings_path='image_embeddings.pickle',
        image_folder='./coco_images_resized'
    )
except Exception as e:
    print(f"Error initializing search engine: {str(e)}")
    raise

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory(app.config['IMAGES_FOLDER'], filename)

@app.route('/search', methods=['POST'])
def search():
    try:
        query_type = request.form.get('query_type')
        use_pca = request.form.get('use_pca') == 'true'
        pca_components = int(request.form.get('pca_components', 50))
        
        if query_type == 'image':
            if 'image' not in request.files:
                return jsonify({'error': 'No image file provided'}), 400
            
            file = request.files['image']
            if file.filename == '':
                return jsonify({'error': 'No selected file'}), 400
            
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            results = search_engine.image_search(
                image_path=filepath,
                k=5,
                use_pca=use_pca,
                n_components=pca_components
            )
            
            os.remove(filepath)
            
            return jsonify(results)
            
        elif query_type == 'text':
            text_query = request.form.get('text_query', '')
            if not text_query:
                return jsonify({'error': 'No text query provided'}), 400
                
            results = search_engine.text_search(
                text_query=text_query,
                k=5,
                use_pca=use_pca,
                n_components=pca_components
            )
            return jsonify(results)
            
        elif query_type == 'hybrid':
            if 'image' not in request.files:
                return jsonify({'error': 'No image file provided'}), 400
                
            text_query = request.form.get('text_query', '')
            if not text_query:
                return jsonify({'error': 'No text query provided'}), 400
                
            file = request.files['image']
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            hybrid_weight = float(request.form.get('hybrid_weight', 0.5))
            
            results = search_engine.hybrid_search(
                image_path=filepath,
                text_query=text_query,
                lambda_weight=hybrid_weight,
                k=5,
                use_pca=use_pca,
                n_components=pca_components
            )
            
            os.remove(filepath)
            return jsonify(results)
            
        else:
            return jsonify({'error': 'Invalid query type'}), 400
            
    except Exception as e:
        import traceback
        print("Error:", str(e))
        print("Traceback:", traceback.format_exc())
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=3000)
