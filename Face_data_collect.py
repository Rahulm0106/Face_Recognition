## 1. Read and show video stream, capture images
## 2. Detect Faces and show bounding box (haarcascade)
## 3. Flatten the largest face image and save in a numpy array
## 4. Repeat the above for multiple people to generate training data

#Import Libraries
import cv2
import numpy as np

# Init Camera
cap = cv2.VideoCapture(0)

# Face Detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

skip = 0

face_data = []
dataset_path = ''

file_name = input("Enter the name of the person : ")
while True:
    ret, frame = cap.read()
    
    # Check if image is detected or not
    if ret == False:
        continue
        
    # Convert image to grayscale
    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(frame,1.3,5)
    faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)

    # Store every 10th face
    if(skip%10==0):
        pass

    # Pick the last face (because it is the largest face according to area(x[2]*x[3]))
    for face in faces[-1:]:
        x,y,w,h = face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        
        # Extract (Crop out required face area) : region of interest
        offset = 10
        face_section = frame[y-offset:y+h+offset,x-offset:]
        face_section = cv2.resize(face_section, (100,100))

        skip += 1
        if skip%10==0:
            face_data.append(face_section)
            print(len(face_data))

    cv2.imshow('Frame', frame)
    #cv2.imshow("Face Section", face_section)
    
    # Wait for user input - q, then loop will stop
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break

# Convert our face list array into a numpy array
face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)

np.save(dataset_path+file_name+'.npy',face_data)
print("Data Successfully saved at "+dataset_path+file_name+'.npy')

cap.release()
cv2.destroyAllWindows()