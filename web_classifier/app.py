# app.py
import os
import csv
import glob
from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Generate a random secret key

# Configuration
BASE_IMAGE_PATH = 'images'
ROI_SUBFOLDER = 'roi'
USER_DB_FILE = 'data/users.csv'
ANNOTATIONS_DIR = 'data/annotations'

# Ensure the data and annotations directories exist
os.makedirs('data', exist_ok=True)
os.makedirs(ANNOTATIONS_DIR, exist_ok=True)

# Create user database if it doesn't exist
if not os.path.exists(USER_DB_FILE):
    with open(USER_DB_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['username', 'password_hash'])
        # Add a default user
        writer.writerow(['admin', generate_password_hash('admin')])

def get_available_folders():
    """Get a list of available folders in the base image path"""
    folders = []
    for item in os.listdir(BASE_IMAGE_PATH):
        folder_path = os.path.join(BASE_IMAGE_PATH, item, ROI_SUBFOLDER)
        if os.path.isdir(folder_path) and any(f.endswith('.jpg') for f in os.listdir(folder_path)):
            folders.append(item)
    return sorted(folders)

def get_image_list(folder):
    """Get a list of all jpg images in the specified folder"""
    image_path = os.path.join(BASE_IMAGE_PATH, folder, ROI_SUBFOLDER)
    return sorted([os.path.basename(f) for f in glob.glob(os.path.join(image_path, '*.jpg'))])

def get_results_file(username, folder):
    """Get the results file name for the specified user and folder"""
    return os.path.join(ANNOTATIONS_DIR, f'annotations_{username}_{folder}.csv')

def get_annotated_images(username, folder):
    """Get a set of images that have already been annotated by the user"""
    annotated = set()
    results_file = get_results_file(username, folder)
    
    if os.path.exists(results_file):
        with open(results_file, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                annotated.add(row['filename'])
    return annotated

def get_next_image(username, folder):
    """Get the next image that hasn't been annotated by the user"""
    all_images = get_image_list(folder)
    annotated = get_annotated_images(username, folder)
    
    for img in all_images:
        if img not in annotated:
            return img
    
    return None  # All images have been annotated

def get_annotation_stats(username, folder):
    """Get statistics about annotations for the specified user and folder"""
    all_images = get_image_list(folder)
    annotated = get_annotated_images(username, folder)
    
    total = len(all_images)
    completed = len(annotated)
    remaining = total - completed
    progress = (completed / total * 100) if total > 0 else 0
    
    return {
        'total': total,
        'completed': completed,
        'remaining': remaining,
        'progress': progress
    }

def authenticate_user(username, password):
    """Check if the username and password match"""
    with open(USER_DB_FILE, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['username'] == username and check_password_hash(row['password_hash'], password):
                return True
    return False

def register_user(username, password):
    """Register a new user"""
    # Check if username already exists
    with open(USER_DB_FILE, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['username'] == username:
                return False
    
    # Add new user
    with open(USER_DB_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([username, generate_password_hash(password)])
    return True

@app.route('/images/<path:folder_path>/<path:filename>')
def serve_image(folder_path, filename):
    """Serve image files directly from the images directory"""
    # Construct the full path to where the image is stored
    image_dir = os.path.join(BASE_IMAGE_PATH, folder_path, ROI_SUBFOLDER)
    return send_from_directory(image_dir, filename)

@app.route('/')
def index():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    username = session['username']
    
    # Get available folders
    folders = get_available_folders()
    
    # If no folders are available
    if not folders:
        return render_template('no_folders.html')
    
    # If no folder is selected, show folder selection page
    if 'folder' not in session:
        # Get folder stats for display
        folder_stats = {}
        for folder in folders:
            folder_stats[folder] = get_annotation_stats(username, folder)
        
        return render_template('select_folder.html', folders=folders, folder_stats=folder_stats)
    
    # If folder is invalid (might have been removed), redirect to folder selection
    if session['folder'] not in folders:
        session.pop('folder', None)
        return redirect(url_for('index'))
    
    # Get the next image to annotate
    folder = session['folder']
    next_image = get_next_image(username, folder)
    
    # Get statistics
    stats = get_annotation_stats(username, folder)
    
    if next_image:
        # Use the direct image route instead of static files
        image_url = url_for('serve_image', folder_path=folder, filename=next_image)
        
        return render_template('annotate.html', 
                               image=next_image, 
                               folder=folder,
                               image_url=image_url,
                               progress=stats['progress'],
                               annotated=stats['completed'],
                               total=stats['total'])
    else:
        return render_template('complete.html', folder=folder, stats=stats)

@app.route('/select_folder/<folder>')
def select_folder(folder):
    if 'username' not in session:
        return redirect(url_for('login'))
    
    folders = get_available_folders()
    if folder in folders:
        session['folder'] = folder
    
    return redirect(url_for('index'))

@app.route('/submit', methods=['POST'])
def submit():
    if 'username' not in session or 'folder' not in session:
        return redirect(url_for('login'))
    
    username = session['username']
    folder = session['folder']
    filename = request.form.get('filename')
    courtship_score = request.form.get('courtship_score')
    confidence_score = request.form.get('confidence_score')
    
    # Check if this image has already been annotated by this user
    annotated_images = get_annotated_images(username, folder)
    if filename in annotated_images:
        # Skip saving and just go to the next image
        return redirect(url_for('index'))
    
    # Get the results file for this user and folder
    results_file = get_results_file(username, folder)
    
    # Create the file with header if it doesn't exist
    if not os.path.exists(results_file):
        with open(results_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['scorer', 'filename', 'courtship_score', 'confidence_score', 'folder'])
    
    # Save the annotation
    with open(results_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([username, filename, courtship_score, confidence_score, folder])
    
    return redirect(url_for('index'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if authenticate_user(username, password):
            session['username'] = username
            # Clear folder selection on login
            if 'folder' in session:
                session.pop('folder', None)
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error='Invalid credentials')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if register_user(username, password):
            session['username'] = username
            return redirect(url_for('index'))
        else:
            return render_template('register.html', error='Username already exists')
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    session.pop('folder', None)
    return redirect(url_for('login'))

@app.route('/change_folder')
def change_folder():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    if 'folder' in session:
        session.pop('folder', None)
    
    return redirect(url_for('index'))

if __name__ == '__main__':
    # Run the app
    app.run(host='0.0.0.0', port=5000, debug=True)