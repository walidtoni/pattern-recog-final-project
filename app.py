import os
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import pickle
from skimage.feature import hog, local_binary_pattern
from io import BytesIO
import base64
import matplotlib
matplotlib.use('Agg')  # Required for saving plots in Flask
import matplotlib.pyplot as plt

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# Load only the model (assuming it has built-in preprocessing)
with open('models/model.p', 'rb') as f:
    model = pickle.load(f)

# Default classes (modify based on your model)
CLASS_NAMES = {
    0: "PET",
    1: "HDPE", 
    2: "PVC",
    3: "LDPE",
    4: "PP",
    5: "PS",
    6: "Other"
}

def plot_image_processing(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (128, 128))
    eq = cv2.equalizeHist(resized)
    blur = cv2.GaussianBlur(eq, (5, 5), 0)

    # HOG visualization
    _, hog_img = hog(blur, orientations=9, pixels_per_cell=(8, 8),
                     cells_per_block=(2, 2), visualize=True)

    # LBP visualization
    lbp = local_binary_pattern(blur, P=8, R=1, method='uniform')

    plt.figure(figsize=(10, 15))  # Adjust aspect ratio

    # 2 columns x 3 rows
    plt.subplot(3, 2, 1)
    plt.title('Original Image')
    plt.imshow(img_rgb)
    plt.axis('off')

    plt.subplot(3, 2, 2)
    plt.title('Grayscale')
    plt.imshow(gray, cmap='gray')
    plt.axis('off')

    plt.subplot(3, 2, 3)
    plt.title('Equalized')
    plt.imshow(eq, cmap='gray')
    plt.axis('off')

    plt.subplot(3, 2, 4)
    plt.title('Blurred')
    plt.imshow(blur, cmap='gray')
    plt.axis('off')

    plt.subplot(3, 2, 5)
    plt.title('HOG Features')
    plt.imshow(hog_img, cmap='gray')
    plt.axis('off')

    plt.subplot(3, 2, 6)
    plt.title('LBP Features')
    plt.imshow(lbp, cmap='gray')
    plt.axis('off')

    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def extract_hog_features(image):
    # Must match training parameters
    return hog(image, orientations=9, pixels_per_cell=(8, 8),
             cells_per_block=(2, 2), channel_axis=None)

def extract_color_hist(image, bins=(8, 8, 8)):  # Must match training
    resized = cv2.resize(image, (128, 128))
    hist = cv2.calcHist([resized], [0, 1, 2], None, bins,
                       [0, 256, 0, 256, 0, 256])
    return cv2.normalize(hist, hist).flatten()

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (128, 128))  # Must match training size
    eq = cv2.equalizeHist(resized)
    return cv2.GaussianBlur(eq, (5, 5), 0)

def extract_features(image_path, mode='all_combined'):
    gray_image = preprocess_image(image_path)
    color_image = cv2.imread(image_path)
    
    features = []
    if 'hog' in mode or mode == 'all_combined':
        features.extend(extract_hog_features(gray_image))
    if 'color' in mode or mode == 'all_combined':
        features.extend(extract_color_hist(color_image))
    if 'lbp' in mode or mode == 'all_combined':
        features.extend(extract_lbp_features(gray_image))
    
    return np.array(features)

def extract_lbp_features(gray_image, P=8, R=1, method='uniform'):
    lbp = local_binary_pattern(gray_image, P, R, method)
    (hist, _) = np.histogram(lbp.ravel(),
                             bins=np.arange(0, P + 3),
                             range=(0, P + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist

def predict_image(image_path):
    try:
        # Load and preprocess
        gray_image = preprocess_image(image_path)
        color_image = cv2.imread(image_path)
        
        # Extract all features
        hog_feat = extract_hog_features(gray_image)
        color_feat = extract_color_hist(color_image)
        lbp_feat = extract_lbp_features(gray_image)
        
        # Combine features
        features = np.hstack([hog_feat, color_feat, lbp_feat])
        print(f"Feature sizes - HOG: {len(hog_feat)}, Color: {len(color_feat)}, LBP: {len(lbp_feat)}")
        
        # Validate dimension
        if len(features) != 8622:
            raise ValueError(f"Feature dimension mismatch. Expected 8622, got {len(features)}")
        
        # Predict
        features = features.reshape(1, -1)
        pred = model.predict(features)[0]
        return CLASS_NAMES.get(pred, "Unknown")
    
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return "Error"

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename != '':
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            prediction = predict_image(filepath)
            plot_url = plot_image_processing(filepath)  # ✅ Add this line
            
            return render_template('index.html', 
                                   filename=filename,
                                   prediction=prediction,
                                   plot_url=plot_url)  # ✅ Pass this to template

    return render_template('index.html')


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)