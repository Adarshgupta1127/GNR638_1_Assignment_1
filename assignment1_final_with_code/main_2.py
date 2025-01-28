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

def extract_bag_of_sifts(image_paths, labels, vocab_size=100):
    """Extract Bag of SIFT features from each image, ensuring labels remain in sync."""
    sift = cv2.SIFT_create()
    descriptors_list = []
    valid_image_paths = []
    valid_labels = []

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
    print(f"Visual vocabulary created with {vocab_size} clusters.")

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
    print(f"Accuracy report for {classifier_type}:")
    print(accuracy_report)

    return predicted_labels, classifier, accuracy_report

def visualize(features, labels, categories, output_path='tsne_visualization.png'):
    """Visualize the distribution of image features using T-SNE."""
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
    # make the labels visible properly not overlaping
    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(categories))
    plt.xticks(tick_marks, categories, rotation=45, ha='right')
    plt.yticks(tick_marks, categories)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(output_path)
    print(f"Confusion matrix saved as {output_path}.")

def main(data_path):
    
    print("loading the data")
    image_paths, labels, categories = get_image_paths_and_labels(data_path)

    vocab_sizes = [50, 100, 150, 200, 250]
    accuracies = []
    
    for vocab_size in vocab_sizes:
        print(f"\nProcessing with vocabulary size: {vocab_size}")

        features, valid_labels, vocab = extract_bag_of_sifts(image_paths, labels, vocab_size=vocab_size)

        features_train, features_test, labels_train, labels_test = train_test_split(features, valid_labels, test_size=0.3, random_state=42)
        print("predicting using the classifier")
        nn_predictions, nn_classifier, nn_accuracy_report = train_and_evaluate_classifiers(features_train, labels_train, features_test, labels_test, 'nearest_neighbor')
        svm_predictions, svm_classifier, svm_accuracy_report = train_and_evaluate_classifiers(features_train, labels_train, features_test, labels_test, 'support_vector_machine')

        nn_cm = confusion_matrix(labels_test, nn_predictions, labels=categories)
        svm_cm = confusion_matrix(labels_test, svm_predictions, labels=categories)

        plot_confusion_matrix(nn_cm, categories, output_path=f'nn_confusion_matrix_{vocab_size}.png', title=f'Nearest Neighbor Confusion Matrix ({vocab_size})')
        plot_confusion_matrix(svm_cm, categories, output_path=f'svm_confusion_matrix_{vocab_size}.png', title=f'SVM Confusion Matrix ({vocab_size})')

        visualize(features, valid_labels, categories, output_path=f'tsne_visualization_{vocab_size}.png')

        accuracies.append({
            'vocab_size': vocab_size,
            'nn_accuracy': nn_accuracy_report,
            'svm_accuracy': svm_accuracy_report
        })

    vocab_sizes = np.array(vocab_sizes)
    nn_accuracies = [float(acc['nn_accuracy'].split()[-2]) for acc in accuracies]
    svm_accuracies = [float(acc['svm_accuracy'].split()[-2]) for acc in accuracies]

    plt.figure(figsize=(8, 6))
    plt.plot(vocab_sizes, nn_accuracies, label='Nearest Neighbor')
    plt.plot(vocab_sizes, svm_accuracies, label='SVM')
    plt.xlabel('Vocabulary Size')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs Vocabulary Size')
    plt.legend()
    plt.savefig('accuracy_vs_vocab_size.png')
    print("Accuracy vs Vocabulary Size plot saved.")

    print("\nImage classification process completed.")

# Example usage
main('./data/Images')
