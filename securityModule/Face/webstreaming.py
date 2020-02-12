# USAGE
# python webstreaming.py --ip 0.0.0.0 --port 8000
from twilio.rest import Client
# import the necessary packages
from sercurity_module import Sercurity
# from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template, jsonify, make_response, request, url_for, redirect
import threading
import argparse
import datetime
# import imutils
import cv2
import time
import numpy as np
import time

import firebase_admin
from firebase_admin import credentials, firestore
import json

cred = credentials.Certificate('/Users/master/Documents/PersonalDev/uoftHacks/uoftHacks2020/supervisor-f2f29-firebase-adminsdk-l2twy-ae836f2735.json')

default_app = firebase_admin.initialize_app(cred)

#Global variable
GScreen = False

db = firestore.client()

# def toBinary(n):
#     return ''.join(str(1 & int(n) >> i) for i in range(64)[::-1])

# a = toBinary(0)

# hardcoded
userID = 'NH17KayNX5dm0nlnPhklw3gzN7i2'

#Global 
GScreen = {"Camera1": 'off', "Camera2": 'off'}

gFOnChange = True
Gfeatures = {}

doc_ref = db.collection(u'users').document(userID)
doc_ref.set(Gfeatures)


def send_msg(number, data):
	account_sid = os.environ["TWILIO_ACCOUNT_SID"]
	auth_token = os.environ["TWILIO_AUTH_TOKEN"]
	client = Client(account_sid, auth_token)
	# This my person phone number
	client.messages.create(to=number, from_="12563803381", body=data)


def send_log(detected1):
	t = time.localtime()
	current_time = time.strftime("%H:%M:%S", t)
	doc_ref = db.collection(u'Camera').document(u'camera1')
	doc_ref.update({ "Log":{"time": current_time, "name":detected1}})
	

# Use Case of the function 
# send_log("Tien")
# send_log("goku")
# send_log("Disco")

# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful for multiple browsers/tabs
# are viewing tthe stream)
outputFrame = None
lock = threading.Lock()

# initialize a flask object
app = Flask(__name__)

# initialize the video stream and allow the camera sensor to
# warmup


@app.route("/")
def index():
	# return the rendered template
	return render_template("landingpage.html")

@app.route("/settings")
def settings():
	# return the rendered template
	return render_template("settings.html")

@app.route("/landingpage")
def landingpage():
	# return the rendered template
	return render_template("index.html")

@app.route("/profile")
def profile():
	# return the rendered template
	return render_template("profile.html")

@app.route("/registration")
def registration():
	# return the rendered template
	return render_template("registration.html")

def detect_motion(frameCount, datasets_path, vs):
	# grab global references to the video stream, output frame, and
	# lock variables
	global gFOnChange, Gfeatures, outputFrame, lock, GScreen
	sr = Sercurity(datasets_path)
	sr.load_known_face()
	total = 0
	# loop over frames from the video stream
	while True:
		if gFOnChange:
    			sr.load_config(Gfeatures)
			gFOnChange = False
			Gfeatures = {}


		if GScreen['Camera1'] == 'on' or GScreen['Camera2'] == 'on':
			# read the next frame from the video stream, resize it,
			# convert the frame to grayscale, and blur it
			new_frame = None
			frames = []
			# camera1
			if GScreen['Camera1'] == 'on' and GScreen['Camera2'] == 'off':
				_, new_frame = vs[0].read()
				new_frame = cv2.resize(new_frame, (450, 450))
			else:
				_, new_frame = vs[1].read()
				new_frame = cv2.resize(new_frame, (450, 450))
			if GScreen['Camera1'] == 'on' and GScreen['Camera2'] == 'on':
				for v in vs:
					_, frame = v.read()
					frames.append(cv2.resize(frame, (450, 450)))
				new_frame = np.concatenate((frames[0], frames[1]), axis=1)

			if total % 350 == 0:
				tlables, v_scores = sr.detect_labels(new_frame)
				dangers, danger_scores = sr.analyzer(tlables, v_scores)
				# print(dangers)
				if dangers != None:
					print(dangers[0].tostring(), "detected!!!!!!!, Confidence score =", danger_scores[0])
					send_log(dangers[0].tostring())
					send_msg(6473348273, dangers[0].tostring())


			mo, frame_marked = sr.detect_and_show(new_frame, total, frameCount)
			if mo: 
				#[(),()]
				#locations = sr.face_detection(frame)
				face_locations, names = sr.recongnize(new_frame)	
				frame_marked = sr.display_result(frame_marked, face_locations, names)


			# acquire the lock, set the output frame, and release the
			# lock
			# print(frame)
			total += 1
			with lock:
				outputFrame = frame_marked.copy()
		
def generate():
	# grab global references to the output frame and lock variables
	global outputFrame, lock

	# loop over frames from the output stream
	while True:
		# wait until the lock is acquired
		with lock:
			# check if the output frame is available, otherwise skip
			# the iteration of the loop
			if outputFrame is None:
				continue

			# encode the frame in JPEG format
			(flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

			# ensure the frame was successfully encoded
			if not flag:
				continue

		# yield the output frame in the byte format
		yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
			bytearray(encodedImage) + b'\r\n')



@app.route("/video_feed")
def video_feed():
	# return the response generated along with the specific media
	# type (mime type)
	return Response(generate(),mimetype = "multipart/x-mixed-replace; boundary=frame")

@app.route("/_view_log", methods=['POST'])
def view_log():
	doc_ref = db.collection(u'Camera').document(u'camera1')
	doc = doc_ref.get()
	# print(u'Document data: {}'.format(doc.to_dict()))
	result = doc.get('Log')
	# print(u'Document data: {}'.format(result))
	return jsonify(result)

@app.route("/_view_logii", methods=['POST'])
def view_logii():
	doc_ref = db.collection(u'Camera').document(u'camera2')
	doc = doc_ref.get()
	
	result = doc.get('Log')
	
	return jsonify(result)

@app.route("/_view_logiii", methods=['POST'])
def view_logiii():
	doc_ref = db.collection(u'Camera').document(u'camera3')
	doc = doc_ref.get()
	result = doc.get('Log')
	return jsonify(result)

@app.route("/_update_Toddler", methods=['POST'])
def update_Toddler():
  clicked=request.form['data']
	doc_ref = db.collection(u'users').document(userID)
	doc_ref.update({'Toddler': clicked})
	global gFOnChange
	gFOnChange = True
	Gfeatures['Toddler'] = clicked
	return jsonify("Success")


@app.route("/_update_Sharp", methods=['POST'])
def view_logiii():
	doc_ref = db.collection(u'Camera').document(u'camera3')
	doc = doc_ref.get()
	print(u'Document data: {}'.format(doc.to_dict()))
	result = doc.get('Log')
	print(u'Document data: {}'.format(result))
	return jsonify(result)
	


@app.route("/_update_Weapon", methods=['POST'])
def update_Weapon():
	clicked=request.form['data']
	doc_ref = db.collection(u'users').document(userID)
	doc_ref.update({'Weapon': clicked})
	gFOnChange = True
	Gfeatures['Weapon'] = clicked
	return jsonify("Success")


@app.route("/_update_Shoes", methods=['POST'])
def update_Shoes():
	clicked=request.form['data']
	doc_ref = db.collection(u'users').document(userID)
	doc_ref.update({'Shoes': clicked})
	global gFOnChange
	gFOnChange = True
	Gfeatures['Shoes'] = clicked
	return jsonify("Success")

@app.route("/_update_Dirt", methods=['POST'])
def update_Dirt():
	clicked=request.form['data']
	doc_ref = db.collection(u'users').document(userID)
	doc_ref.update({'Dirt': clicked})
	global gFOnChange
	gFOnChange = True
	Gfeatures['Dirt'] = clicked
	return jsonify("Success")

@app.route("/_update_Animal", methods=['POST'])
def update_Animal():
	clicked=request.form['data']
	doc_ref = db.collection(u'users').document(userID)
	doc_ref.update({'Animal': clicked})
	gFOnChange = True
	Gfeatures['Animal'] = clicked
	return jsonify("Success")

@app.route("/_update_Fruit", methods=['POST'])
def update_Fruit():
	clicked=request.form['data']
	doc_ref = db.collection(u'users').document(userID)
	doc_ref.update({'Fruit': clicked})
	gFOnChange = True
	Gfeatures['Fruit'] = clicked
	return jsonify("Success")

@app.route("/_update_Fire", methods=['POST'])
def update_Fire():
	clicked=request.form['data']
	doc_ref = db.collection(u'users').document(userID)
	doc_ref.update({'Fire': clicked})
	gFOnChange = True
	Gfeatures['Fire'] = clicked
	return jsonify("Success")
	
@app.route("/_update_SMStwilio", methods=['POST'])
def update_SMStwilio():
	clicked=request.form['data']
	doc_ref = db.collection(u'users').document(userID)
	doc_ref.update({'SMStwilio': clicked})
	global gFOnChange
	gFOnChange = True
	Gfeatures['SMStwilio'] = clicked
	return jsonify("Success")


@app.route("/_update_Screen1", methods=['POST'])
def update_Screen1():
	clicked=request.form['data']
	
	if (clicked == 'on'):
		GScreen['Camera1'] = 'on'

	else:
		GScreen['Camera1'] = 'off'
	
	return jsonify("Success")

@app.route("/_get_Screen", methods=['POST'])
def get_Screen():
	clicked=request.form['data']
	# update the global variable 
	GScreen = clicked

	return jsonify("Success")

@app.route("/_update_Screen2", methods=['POST'])
def update_Screen2():
	clicked=request.form['data']
	if (clicked == 'on'):
		GScreen['Camera2'] = 'on'
	else:
		GScreen['Camera2'] = 'off'
	
	return jsonify("Success")

# check to see if this is the main thread of execution
if __name__ == '__main__':
	# construct the argument parser and parse command line arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--ip", type=str, required=True,
		help="ip address of the device")
	ap.add_argument("-o", "--port", type=int, required=True,
		help="ephemeral port number of the server (1024 to 65535)")
	ap.add_argument("-f", "--frame-count", type=int, default=32,
		help="# of frames used to construct the background model")
	args = vars(ap.parse_args())
	#vs1 = VideoStream(src=4).start()
	#time.sleep(2.0)

	#vs2 = VideoStream(src=0).start()
	#time.sleep(2.0)

	vs1 = cv2.VideoCapture(0)
	vs2 = cv2.VideoCapture(1)

	# start a thread that will perform motion detection
	t = threading.Thread(target=detect_motion, args=(
		args["frame_count"],'./datasets', [vs1,vs2]))
	t.daemon = True
	t.start()

	#t = threading.Thread(target=detect_motion, args=(
	#	args["frame_count"],'./datasets', vs2))
	#t.daemon = True
	#t.start()
	# start the flask app
	app.run(host=args["ip"], port=args["port"], debug=True,
		threaded=True, use_reloader=False)
	# app.run(host="138.51.9.170", port=args["port"], debug=True,
	# 	threaded=True, use_reloader=False)

# release the video stream pointer
	vs1.release()
	vs2.release()