# Usage:
# 1. Install Python dependencies: cv2, flask (wish that pip install works like a charm)
# 2. Run "python3 main.py".
# 3. Navigate the browser to the local webpage.
from flask import Flask, render_template, Response, request
from flask_cors import CORS
from game import NarutoGame
from time import sleep
import json

app = Flask(__name__)
CORS(app)
app.config['SECRET_KEY'] = 'secret!'
message_to_send = "[]"
prev_message = "[]"

game_start = False


@app.route('/')
def index():
    return render_template('index.html')


def gen(game):
    global message_to_send
    while True:
        frame = game.get_frame()
        message_to_send = json.dumps(game.player_stats)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed')
def video_feed():
    print("video feed")
    return Response(gen(NarutoGame()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/stream")
def stream():
    def event_stream():
        global prev_message
        print("text stream")
        while True:
            if message_to_send:
                sleep(1)
                if message_to_send != prev_message:
                    print("Sent player stats")
                    yield "data:{}\n\n".format(message_to_send)
                prev_message = message_to_send
    return Response(event_stream(), mimetype="text/event-stream")


@app.route('/voice', methods=['POST', 'GET'])
def login():
    if request.method == 'POST':
        choice = request.form["var1"]
        return '''something'''
    elif request.method == 'GET':
        return '''something'''


if __name__ == '__main__':
    app.run(debug=True)
