import os
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import pickle
from skimage.feature import hog, local_binary_pattern
import matplotlib
matplotlib.use('Agg')  # Required for saving plots in Flask
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# Load model components
def load_model_components():
    with open('models/best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('models/label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    return model, scaler, le

model, scaler, le = load_model_components()

# Helper functions from your original code
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (128, 128))
    eq = cv2.equalizeHist(resized)
    blur = cv2.GaussianBlur(eq, (5, 5), 0)
    return blur

def extract_hog_features(image):
    features, _ = hog(image, orientations=9, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), visualize=True, channel_axis=None)
    return features

def extract_color_hist(image, bins=(8, 8, 8)):
    resized = cv2.resize(image, (128, 128))
    hist = cv2.calcHist([resized], [0, 1, 2], None, bins,
                        [0, 256, 0, 256, 0, 256])
    return cv2.normalize(hist, hist).flatten()

def extract_lbp_features(gray_image, P=8, R=1, method='uniform'):
    lbp = local_binary_pattern(gray_image, P, R, method)
    (hist, _) = np.histogram(lbp.ravel(),
                             bins=np.arange(0, P + 3),
                             range=(0, P + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist

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

def predict_image(image_path):
    features = extract_features(image_path).reshape(1, -1)
    features_scaled = scaler.transform(features)
    pred_encoded = model.predict(features_scaled)[0]
    label = le.inverse_transform([pred_encoded])[0]
    return label

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
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 3, 1)
    plt.title('Original Image')
    plt.imshow(img_rgb)
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.title('Grayscale')
    plt.imshow(gray, cmap='gray')
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.title('Equalized')
    plt.imshow(eq, cmap='gray')
    plt.axis('off')
    
    plt.subplot(2, 3, 4)
    plt.title('Blurred')
    plt.imshow(blur, cmap='gray')
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    plt.title('HOG Features')
    plt.imshow(hog_img, cmap='gray')
    plt.axis('off')
    
    plt.subplot(2, 3, 6)
    plt.title('LBP Features')
    plt.imshow(lbp, cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    
    # Save plot to bytes
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Make prediction
            prediction = predict_image(filepath)
            
            # Generate processing visualization
            plot_url = plot_image_processing(filepath)
            
            return render_template('index.html', 
                                filename=filename,
                                prediction=prediction,
                                plot_url=plot_url)
    
    return render_template('index.html')

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)