from flask import Flask,render_template,request,session,Blueprint
import cv2
import cvzone
import math
import os 
from ultralytics import YOLO
import time
import threading
app = Flask(__name__)
model = YOLO("best.pt")
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
# db = SQLAlchemy(app)

# class User(db.Model):
#     id = db.Column(db.Integer,primary_key = True)

# def play_alert_sound():
#     # Play a simple alert sound using pygame
#     pygame.mixer.music.load('alarm.mp3')  # Replace 'alert_sound.wav' with your sound file
#     pygame.mixer.music.play()
#     time.sleep(5)  # Adjust duration as needed
#     pygame.mixer.music.stop()

# def async_play_alert_sound():
#     # Run the play_alert_sound function in a separate thread
#     alert_thread = threading.Thread(target=play_alert_sound)
#     alert_thread.start()

@app.route('/')
def project():
    return render_template('home.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/GetStarted', methods=['POST'])
def getStarted():
    return render_template('predict.html')

@app.route('/aboutUs')
def aboutUs():
    return render_template('aboutUs.html')

@app.route('/contactUs')
def contactUs():
    return render_template('contactUs.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/imagePredict')
def imagePredict():
    return render_template('imagePredict.html')

@app.route('/videoPredict')
def videoPredict():
    return render_template('videoPredict.html')

@app.route('/livePredict')
def livePredict():
    return render_template('livePredict.html')

@app.route('/imageUpload', methods=["GET", "POST"])
def imgPred():
    if request.method == 'POST':
        f = request.files['file']
        if f.filename == '':
            return render_template('fileCheck.html')

        basepath = os.path.dirname(__file__)
        upload_path = os.path.join(basepath, "static/uploads", f.filename)
        result_path = os.path.join(basepath, "static/results", f.filename)

        f.save(upload_path)

        results = model(upload_path)
        annotated = results[0].plot()
        cv2.imwrite(result_path, annotated)

        return render_template('imagePredict.html', result_image=f"/static/results/{f.filename}")


@app.route('/videoUpload', methods=["GET", "POST"])
def vidPred():
    if request.method == 'POST':
        f = request.files['file']
        if f.filename == '':
            return render_template('fileCheck.html')

        basepath = os.path.dirname(__file__)
        upload_path = os.path.join(basepath, "static/uploads", f.filename)
        result_path = os.path.join(basepath, "static/results", f.filename)

        f.save(upload_path)

        cap = cv2.VideoCapture(upload_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(result_path, fourcc, fps, (width, height))

        while True:
            success, frame = cap.read()
            if not success:
                break
            results = model(frame)
            annotated = results[0].plot()
            out.write(annotated)

        cap.release()
        out.release()

        return render_template('videoPredict.html', result_video=f"/static/results/{f.filename}")


if __name__ == "__main__":
    app.run(debug=True)
