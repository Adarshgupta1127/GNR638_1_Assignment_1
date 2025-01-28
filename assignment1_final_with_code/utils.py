import os
import cv2
import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def get_image_paths_and_labels(data_path):
    """Get paths and labels for all images in the directory."""
    print("Step 1: Gathering image paths and labels...")
    image_paths = []
    labels = []
    categories = os.listdir(data_path)
    for category in categories:
        category_path = os.path.join(data_path, category)
        if os.path.isdir(category_path):
            for img_name in os.listdir(category_path):
                img_path = os.path.join(category_path, img_name)
                image_paths.append(img_path)
                labels.append(category)
    print(f"Found {len(image_paths)} images across {len(categories)} categories.")
    return image_paths, labels, categories

def extract_tiny_image_features(image_paths, size=(16, 16)):
    """Extract tiny image features from each image."""
    print("Step 2: Extracting Tiny Image features...")
    features = []
    valid_image_paths = []
    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Unable to read image {img_path}. Skipping...")
            continue
        img = cv2.resize(img, size)
        img = img.flatten()
        features.append(img)
        valid_image_paths.append(img_path)
    print(f"Extracted features from {len(features)} valid images.")
    return np.array(features), valid_image_paths

def extract_bag_of_sifts(image_paths, labels, vocab=None, vocab_size=200):
    """Extract Bag of SIFT features from each image, ensuring labels remain in sync."""
    sift = cv2.SIFT_create()
    descriptors_list = []
    valid_image_paths = []
    valid_labels = []

    print("Step 2.1: Detecting and computing SIFT descriptors for images...")
    for img_path, label in zip(image_paths, labels):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: Unable to read image {img_path}. Skipping...")
            continue
        kp, des = sift.detectAndCompute(img, None)
        if des is not None:
            descriptors_list.append(des)
            valid_image_paths.append(img_path)
            valid_labels.append(label)

    if vocab is None:
        print("Step 2.2: Clustering SIFT descriptors to create visual vocabulary...")
        descriptors = np.vstack(descriptors_list)
        _, labels, centers = cv2.kmeans(
            descriptors.astype(np.float32),
            vocab_size,
            None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2),
            10,
            cv2.KMEANS_RANDOM_CENTERS
        )
        vocab = centers
        with open('vocab.pkl', 'wb') as f:
            pickle.dump(vocab, f)
        print(f"Visual vocabulary created with {vocab_size} clusters.")

    print("Step 2.3: Creating Bag-of-Words histograms for each image...")
    histograms = []
    for des in descriptors_list:
        histogram = np.zeros(vocab_size)
        for descriptor in des:
            distances = np.linalg.norm(vocab - descriptor, axis=1)
            cluster_id = np.argmin(distances)
            if cluster_id < vocab_size:
                histogram[cluster_id] += 1
        histograms.append(histogram)

    print("Bag-of-Words histograms created.")
    return np.array(histograms), valid_labels, vocab

def train_and_evaluate_classifiers(features_train, labels_train, features_test, labels_test, classifier_type='nearest_neighbor'):
    """Train and evaluate classifiers."""
    print(f"Step 3: Training and evaluating {classifier_type} classifier...")
    le = LabelEncoder()
    encoded_labels_train = le.fit_transform(labels_train)
    encoded_labels_test = le.transform(labels_test)

    if classifier_type == 'nearest_neighbor':
        classifier = KNeighborsClassifier(n_neighbors=3)
    elif classifier_type == 'support_vector_machine':
        classifier = SVC(kernel='linear')
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")

    classifier.fit(features_train, encoded_labels_train)
    predicted_labels = classifier.predict(features_test)
    predicted_labels = le.inverse_transform(predicted_labels)

    accuracy_report = classification_report(labels_test, predicted_labels)
    return predicted_labels, classifier, accuracy_report

def visualize_tsne(features, labels, categories, output_path='tsne_visualization.png'):
    """Visualize the distribution of image features using T-SNE."""
    print("Step 4: Visualizing feature distribution using T-SNE...")
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(features)

    plt.figure(figsize=(10, 8))
    for i, category in enumerate(categories):
        category_indices = [j for j, label in enumerate(labels) if label == category]
        plt.scatter(tsne_results[category_indices, 0], tsne_results[category_indices, 1], label=category)

    plt.legend()
    plt.title('T-SNE Visualization of Image Features')
    plt.savefig(output_path)
    print(f"T-SNE visualization saved as {output_path}.")

def plot_confusion_matrix(cm, categories, output_path='confusion_matrix.png', title='Confusion Matrix', cmap=plt.cm.Blues):
    """Plot the confusion matrix."""
    print("Step 5: Plotting confusion matrix...")
    plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(categories))
    plt.xticks(tick_marks, categories, rotation=45)
    plt.yticks(tick_marks, categories)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(output_path)
    print(f"Confusion matrix saved as {output_path}.")