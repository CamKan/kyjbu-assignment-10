import os
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from open_clip import create_model_and_transforms
import open_clip
import pandas as pd
from sklearn.decomposition import PCA
from typing import List, Dict, Optional, Union, Tuple

class ImageSearchEngine:
    def __init__(self, model_name: str = 'ViT-B/32', pretrained: str = 'openai',
                 embeddings_path: str = 'image_embeddings.pickle',
                 image_folder: str = 'coco_images_resized'):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print(f"Using device: {self.device}")
        
        self.image_folder = image_folder
        
        self.model, _, self.preprocess = create_model_and_transforms(
            model_name, 
            pretrained=pretrained,
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
        
        self.load_embeddings(embeddings_path)
        self.pca = None
        self.pca_fitted = False

    def fit_pca(self, n_components: int):
        if self.pca is None or self.pca.n_components_ != n_components:
            self.pca = PCA(n_components=n_components)
            self.pca.fit(self.embeddings)
            self.pca_fitted = True
    
    def load_embeddings(self, embeddings_path: str):
        self.df = pd.read_pickle(embeddings_path)
        self.embeddings = np.stack(self.df['embedding'].values)
    
    def get_similar_images(self, query_embedding: np.ndarray, k: int = 5,
                          use_pca: bool = False, n_components: int = 50) -> List[Dict]:

        embeddings = self.embeddings.copy()
        
        if use_pca:
            if not self.pca_fitted or self.pca.n_components_ != n_components:
                self.fit_pca(n_components)

            embeddings = self.pca.transform(embeddings)
            query_embedding = self.pca.transform(query_embedding.reshape(1, -1))
        
        similarities = np.dot(embeddings, query_embedding.T).squeeze()
        top_k_idx = np.argsort(similarities)[-k:][::-1]
        
        results = []
        for idx in top_k_idx:
            results.append({
                'file_name': self.df.iloc[idx]['file_name'],
                'path': os.path.join(self.image_folder, self.df.iloc[idx]['file_name']),
                'similarity': float(similarities[idx])
            })
        
        return results
    
    @torch.no_grad()
    def text_search(self, text_query: str, k: int = 5, use_pca: bool = False,
                   n_components: int = 50) -> List[Dict]:

        text = self.tokenizer([text_query]).to(self.device)
        text_features = F.normalize(self.model.encode_text(text))
        query_embedding = text_features.cpu().numpy()
        
        return self.get_similar_images(query_embedding, k, use_pca, n_components)
    
    @torch.no_grad()
    def image_search(self, image_path: str, k: int = 5, use_pca: bool = False,
                    n_components: int = 50) -> List[Dict]:

        try:
            image = Image.open(image_path).convert('RGB')
            image = self.preprocess(image).unsqueeze(0)
            image = image.to(self.device)
            
            image_features = F.normalize(self.model.encode_image(image))
            query_embedding = image_features.cpu().numpy()
            
            return self.get_similar_images(query_embedding, k, use_pca, n_components)
        except Exception as e:
            print(f"Error in image_search: {str(e)}")
            raise
    
    @torch.no_grad()
    def hybrid_search(self, image_path: str, text_query: str, lambda_weight: float = 0.5,
                     k: int = 5, use_pca: bool = False, n_components: int = 50) -> List[Dict]:

        image = self.preprocess(Image.open(image_path).convert('RGB')).unsqueeze(0).to(self.device)
        image_features = F.normalize(self.model.encode_image(image))

        text = self.tokenizer([text_query]).to(self.device)
        text_features = F.normalize(self.model.encode_text(text))

        combined_features = F.normalize(
            lambda_weight * text_features + (1.0 - lambda_weight) * image_features
        )
        query_embedding = combined_features.cpu().numpy()
        
        return self.get_similar_images(query_embedding, k, use_pca, n_components)

if __name__ == "__main__":

    engine = ImageSearchEngine()

    results = engine.text_search("cat cuddles with dog on sofa.")
    print("Text Search Results:")
    for r in results:
        print(f"File: {r['file_name']}, Similarity: {r['similarity']:.4f}")

    results = engine.image_search("house.jpg")
    print("\nImage Search Results:")
    for r in results:
        print(f"File: {r['file_name']}, Similarity: {r['similarity']:.4f}")
