import face_recognition
import cv2
import numpy as np
import os
import sys
import string
import random



# Get datasets dirctory
rootdir = sys.argv[1]

# Create arrays of known face encodings and their names
known_face_encodings = []
known_face_names = []

for filename in os.listdir(rootdir):
    if 'jpg' in filename or 'png' in filename:
        face_path = os.path.join(rootdir, filename)
        # Load face from datasets
        face = face_recognition.load_image_file(face_path)
        face_encoding = face_recognition.face_encodings(face)[0]
        known_face_encodings.append(face_encoding)
        face_name = filename.replace('_', '.').split('.')[0]
        known_face_names.append(face_name)
        print(face_name)


# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)


# Initialize some variables
#face_locations = []
#face_encodings = []
#face_names = []
process_this_frame = True

def randomString(stringLength=10):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))

# inputs:
# locations will be a list of tuple which contains the location of each face[(),()]
# img will be a 3 dimentional img matrix
# return:
# list of face face_encodings
def extract_face(locations, img):
    return face_recongition.face_encoding(rgb_small_frame, locations) 

def motion_detection(pre_frame, current_frame):
    pass
while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        print(face_encoding)
        face_names = []
        for i, face_encoding in enumerate(face_encodings):
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            # Unkonw face detected
            # more feature can be addded
            else:
                print(face_locations)
                confirm = input('Unkonw face detected. Do you want add it to dataset?(Y/N)')
                if confirm == 'Y' or confirm == 'y':
                    new_face_name = input('Name:')
                    new_face_path = os.path.join(rootdir, new_face_name+'_'+randomString(10)+'.jpg')

                    top = face_locations[i][0]*4
                    right = face_locations[i][1]*4
                    bottom = face_locations[i][2]*4
                    left = face_locations[i][3]*4

                    face = frame[top:bottom, left:right]
                    cv2.imwrite(new_face_path, face) 
                    known_face_encodings.append(face_encoding)
                    known_face_names.append(new_face_name)
                    print('New image saved!')

            face_names.append(name)

    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
