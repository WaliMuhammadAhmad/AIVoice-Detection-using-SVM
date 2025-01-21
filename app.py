import os
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from predict import analyze_audio

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed file extensions
ALLOWED_EXTENSIONS = {'wav'}

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def process():
    try:
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash("No file part in the request.")
            return redirect(url_for('index'))

        file = request.files['file']

        # If the user does not select a file, return to the index page
        if file.filename == '':
            flash("No file selected.")
            return redirect(url_for('index'))

        # Check if the file is allowed
        if file and allowed_file(file.filename):
            # Secure the filename and save it to the upload folder
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Analyze the audio file
            result = analyze_audio(file_path)

            # Render the index template with the result
            return render_template('index.html', result=result)

        else:
            flash("Invalid file type. Only .wav files are allowed.")
            return redirect(url_for('index'))

    except Exception as e:
        # Log the error and return to the index page with an error message
        print(f"Error processing file: {e}")
        flash("An error occurred while processing the file.")
        return redirect(url_for('index'))

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

if __name__ == '__main__':
    app.run(debug=True)