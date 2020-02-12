import face_recognition
import cv2
import numpy as np
import torch
import torchtext
import os
import sys
import string
import random
import imutils
import datetime
from google.cloud import vision

class Sercurity:
	def __init__(self, datasets_path, accumWeight=0.5):
		self.glove = torchtext.vocab.GloVe(name="6B", dim=50)
		self.datasets_path = datasets_path
		self.known_face_encodings = []
		self.known_face_names = []
	   	# store the accumulated weight factor
		self.accumWeight = accumWeight
		# initialize the background model
		self.bg = None
		self.dangers = ['gun', 'knife', 'coke']

	def print_closest_words(self, vec, n):
		dists = torch.norm(self.glove.vectors - vec, dim=1)
		lst = sorted(enumerate(dists.numpy()), key=lambda x: x[1])
		result = []
		for idx, difference in lst[1:n+1]: 	
			result.append(self.glove.itos[idx])
		return result
	
	def load_config(self, dist):
		for key, item in dist.items():
			print(key, item)
			if item == 'on':
				if 'SMS' in key:
				# set up SMS
					pass
				else:
					self.dangers.append(key)
			else:
				if 'SMS' in key:
					# disable SMS
					pass
				else:
					self.dangers.remove(key)

			

	def update(self, image):
		# if the background model is None, initialize it
		if self.bg is None:
			self.bg = image.copy().astype("float")
			return

		# update the background model by accumulating the weighted
		# average
		cv2.accumulateWeighted(image, self.bg, self.accumWeight)

	def load_known_face(self):
		for filename in os.listdir(self.datasets_path):
			if 'jpg' in filename or 'png' in filename:
				face_path = os.path.join(self.datasets_path, filename)
				# Load facechicken from datasets
				face = face_recognition.load_image_file(face_path)
				face_encoding = face_recognition.face_encodings(face)[0]
				self.known_face_encodings.append(face_encoding)
				face_name = filename.replace('_', '.').split('.')[0]
				self.known_face_names.append(face_name)
				print(face_name)

	# inputs:
	# locations will be a list of tuple which contains the location of each face[(),()]
	# img will be a 3 dimentional img matrix
	# return:
	# list of face face_encodings
	def extract_face(self,locations, img):
		return locations.face_encoding(img, locations) 

	def randomString(self, stringLength=10):
		letters = string.ascii_lowercase
		return ''.join(random.choice(letters) for i in range(stringLength))

	def face_detection(self, image):
		client = vision.ImageAnnotatorClient()
		imageByte = vision.types.Image(content=cv2.imencode('.jpg', image)[1].tostring())

		response = client.label_detection(image=imageByte)
		labels = response.label_annotations
		print('Labels:')
		for label in labels:
			print(label.description)
			
		return faceBounds
	def set_danger_label(self, list_label):
		self.dange_labels = list_label

	def detect_labels(self, image):
		"""Detects labels in the file."""
		client = vision.ImageAnnotatorClient()
		imageByte = vision.types.Image(content=cv2.imencode('.jpg', image)[1].tostring())
		tlabels = []
		v_scores = []
		objects = client.object_localization(image=imageByte).localized_object_annotations
		print('Number of objects found: {}'.format(len(objects)))
		for object_ in objects:
			tlabels.append(object_.name)
			v_scores.append(object_.score)
			print('\n{} (confidence: {})'.format(object_.name, object_.score))

		return tlabels, v_scores       
	
	def analyzer(self, labels, v_scores):
		
		if np.shape(v_scores)[0] != 0:
			n_scores = np.zeros(np.shape(v_scores))
			n_label = np.empty(np.shape(v_scores), dtype="S15")
			for i, label in enumerate(labels):

				##### word calculus #####
				label = label.split(' ')
				label_t = self.glove[label[0].lower()]
				for token in label[1:]:
					label_t = label_t + self.glove[token]
				##### word calculus #####

				# print(label)
				for danger in self.dangers:
					danger_t = self.glove[danger.lower()].unsqueeze(0)
					max_sim = 0
					# danger_sets = self.print_closest_words(danger_t, 2)
					# max_sim = torch.cosine_similarity(label_t.unsqueeze(0), danger_t)
					# n_label[i] = danger
					# n_scores[i] = max_sim 
					# for danger_sub in danger_sets:
						# danger_sub_t = self.glove[danger_sub.lower()].unsqueeze(0)
					similarity = torch.cosine_similarity(label_t.unsqueeze(0), danger_t)
						# print(similarity)
					if similarity > max_sim:
						max_sim = similarity
						n_label[i] = danger
						n_scores[i] = similarity.item()
			labels = np.array(labels)
			n_scores = np.array(n_scores)
			v_scores = np.array(v_scores)
			norm = np.add(n_scores, v_scores) / 2
			# print(norm[norm>0.55])
			n_label = np.array(n_label)
			dangers_need_report = n_label[norm>0.78]
			norm = norm[norm>0.78]
			return dangers_need_report, norm
		return None, None

	def add_new_face_to_datasets(self, face_img, face_encoding):
		confirm = input('Unkonw face detected. Do you want add it to dataset?(Y/N)')
		if confirm == 'Y' or confirm == 'y':
			new_face_name = input('Name:')
			new_face_path = os.path.join(self.datasets_path, new_face_name+'_'+randomString(10)+'.jpg')
 
			cv2.imwrite(new_face_path, face_img) 
			self.known_face_encodings.append(face_encoding)
			self.known_face_names.append(new_face_name)
			print('New image saved!')

	def crop_face(self, frame, face_location):
		top = face_location[0]*4
		right = face_location[1]*4
		bottom = face_location[2]*4
		left = face_location[3]*4

		face = frame[top:bottom, left:right]
		return face

	def shrink_frame(self, frame):
		small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
		return small_frame[:, :, ::-1]

	def recongnize(self, frame):
		rgb_small_frame = self.shrink_frame(frame)
		face_locations = face_recognition.face_locations(rgb_small_frame)
		face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

		name_result = []
		for i, face_encoding in enumerate(face_encodings):
			matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
			name = 'Unknown' 
			face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
			best_match_index = np.argmin(face_distances)
			if matches[best_match_index]:
				name = self.known_face_names[best_match_index]
			name_result.append(name)
		return face_locations, name_result

	def display_result(self, frame, locations, names):
		# Display the results
		for (top, right, bottom, left), name in zip(locations, names):
			# Scale back up face locations since the frame we detected in was scaled to 1/4 size
			top *= 4
			right *= 4
			bottom *= 4
			left *= 4

			# Draw a box around the face
			cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

			# Draw a label with a name below the face
			cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
			font = cv2.FONT_HERSHEY_DUPLEX
			cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
		return frame
 
	def motion_detect(self, image, tVal=45):

		   # compute the absolute difference between the background model                           
		   # and the image passed in, then threshold the delta image                                
		   delta = cv2.absdiff(self.bg.astype("uint8"), image)
		   thresh = cv2.threshold(delta, tVal, 255, cv2.THRESH_BINARY)[1]                           
		   
		   # perform a series of erosions and dilations to remove small                             
		   # blobs
		   thresh = cv2.erode(thresh, None, iterations=2)
		   thresh = cv2.dilate(thresh, None, iterations=2)                                          
		   
		   # find contours in the thresholded image and initialize the                              
		   # minimum and maximum bounding box regions for motion
		   cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,                                
				   cv2.CHAIN_APPROX_SIMPLE)
		   cnts = imutils.grab_contours(cnts)                                                       
		   (minX, minY) = (np.inf, np.inf)
		   (maxX, maxY) = (-np.inf, -np.inf)                                                        
		   
		   # if no contours were found, return None                                                 
		   if len(cnts) == 0:
				   return None                                                                      
		   
		   # otherwise, loop over the contours                                                      
		   for c in cnts:
				   # compute the bounding box of the contour and use it to                          
				   # update the minimum and maximum bounding box regions                            
				   (x, y, w, h) = cv2.boundingRect(c)
				   (minX, minY) = (min(minX, x), min(minY, y)) 
				   (maxX, maxY) = (max(maxX, x + w), max(maxY, y + h))                              
		   
		   # otherwise, return a tuple of the thresholded image along                               
		   # with bounding box
		   return (thresh, (minX, minY, maxX, maxY))

	def detect_and_show(self, frame, total, frameCount):
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		if np.shape(gray) == np.shape(self.bg):
			gray = cv2.GaussianBlur(gray, (7, 7), 0)

			mo = False
			# grab the current timestamp and draw it on the frame
			timestamp = datetime.datetime.now()
			cv2.putText(frame, timestamp.strftime(
				"%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
			# if the total number of frames has reached a sufficient
			# number to construct a reasonable background model, then
			# continue to process the frame
			if total > frameCount:
				# detect motion in the image
				motion = self.motion_detect(gray)
				# cehck to see if motion was found in the frame
				if motion is not None:
				# unpack the tuple and draw the box surrounding the
				# "motion area" on the output frame
					(thresh, (minX, minY, maxX, maxY)) = motion
					cv2.rectangle(frame, (minX, minY), (maxX, maxY),
						(0, 0, 255), 2)
					mo = True
			# update the background model and increment the total number
			# of frames read thus far
			self.update(gray)
			return mo, frame
		else:
			self.bg = None
			self.update(gray)
			return False, frame


