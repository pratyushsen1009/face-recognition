import cv2
import face_recognition
import warnings
warnings.filterwarnings("ignore")

original_image = cv2.imread('images/test/modi-trump.jpg') 
modi_image = face_recognition.load_image_file('images/train/modi.jpg')
#assuming sample image is containing a single face
modi_face_encoding = face_recognition.face_encodings(modi_image)[0] 
trump_image = face_recognition.load_image_file('images/train/trump.jpg')
trump_face_encoding = face_recognition.face_encodings(trump_image)[0]
known_face_encodings = [modi_face_encoding,trump_face_encoding]
known_face_names = ['Narendra Modi','Donald Trump']

image_to_recognize = face_recognition.load_image_file('images/test/modi-trump.jpg')
all_face_locations = face_recognition.face_locations(image_to_recognize,model = 'hog')
all_face_encodings = face_recognition.face_encodings(image_to_recognize,all_face_locations)
for current_face_location,current_face_encoding in zip(all_face_locations,all_face_encodings):
    top_pos,right_pos,bottom_pos,left_pos = current_face_location
    all_matches = face_recognition.compare_faces(known_face_encodings,current_face_encoding)
    name_of_person = 'Unknown'
    if True in all_matches:
        first_match_index = all_matches.index(True)
        name_of_person = known_face_names[first_match_index]
    cv2.rectangle(original_image,(left_pos,top_pos),(right_pos,bottom_pos),(255,0,0),2)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(original_image,name_of_person,(left_pos,bottom_pos),font,0.5,(255,255,255),1)
    
cv2.imshow('Identified Faces',original_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
    