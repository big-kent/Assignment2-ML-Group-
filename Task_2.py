import os
import numpy as np # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
from tensorflow.keras.applications import EfficientNetB0 # type: ignore
from tensorflow.keras.applications.efficientnet import preprocess_input # type: ignore
from sklearn.neighbors import NearestNeighbors # type: ignore
import concurrent.futures
import pickle
import matplotlib.pyplot as plt # type: ignore
import matplotlib.image as mpimg # type: ignore

def is_image_file(filename):
    """Check if a file is an image based on its extension."""
    valid_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif"]
    return any(filename.lower().endswith(ext) for ext in valid_extensions)

def extract_features(img_path, model):
    """Extract features from an image using the EfficientNetB0 model."""
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    features = model.predict(preprocessed_img)
    flattened_features = features.flatten()
    normalized_features = flattened_features / np.linalg.norm(flattened_features)
    return normalized_features

def process_image(img_path, model):
    return img_path, extract_features(img_path, model)

def extract_features_from_dataset(directory, model, cache_path='features_cache_task2.pkl'):
    """Extract features from all images in the dataset."""
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            image_paths, features_list = pickle.load(f)
    else:
        image_paths = []
        features_list = []
        img_paths = []
        for subdir, dirs, files in os.walk(directory):
            for file in files:
                if is_image_file(file):
                    img_paths.append(os.path.join(subdir, file))

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(lambda p: process_image(p, model), img_paths))

        for img_path, features in results:
            image_paths.append(img_path)
            features_list.append(features)

        with open(cache_path, 'wb') as f:
            pickle.dump((image_paths, features_list), f)
    return image_paths, features_list

def recommend_similar_images(features, all_features, all_paths, n=10):
    """Recommend n similar images based on feature similarity."""
    neighbors = NearestNeighbors(n_neighbors=n, metric='euclidean')
    all_features = np.array(all_features)
    neighbors.fit(all_features)
    features = np.array(features).reshape(1, -1)
    distances, indices = neighbors.kneighbors(features)
    recommended_images = [all_paths[idx] for idx in indices.flatten()]
    similarity_scores = distances.flatten()
    return recommended_images, similarity_scores

def display_images(image_paths):
    """Display images in a grid."""
    plt.figure(figsize=(15, 10))
    for i, img_path in enumerate(image_paths):
        img = mpimg.imread(img_path)
        plt.subplot(2, 5, i + 1)  # Adjust the grid size depending on how many images you want to show
        plt.imshow(img)
        plt.axis('off')
        plt.title(os.path.basename(img_path))
    plt.show()