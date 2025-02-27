import cv2
import face_recognition
import warnings
warnings.filterwarnings("ignore")

def printImage(image,title):
    cv2.imshow(title,image)
    #cv2.waitKey(0)  # wait for a key press
    #cv2.destroyAllWindows()  # explicitly close the window
    
webcam_video_stream = cv2.VideoCapture(0)

modi_image = face_recognition.load_image_file('images/train/modi.jpg')
modi_face_encodings = face_recognition.face_encodings(modi_image)[0] 
trump_image = face_recognition.load_image_file('images/train/trump.jpg')
trump_face_encodings = face_recognition.face_encodings(trump_image)[0]
pratyush_image = face_recognition.load_image_file('images/train/pratyush.jpg')
pratyush_face_encodings = face_recognition.face_encodings(pratyush_image)[0] 
known_face_encodings = [modi_face_encodings,trump_face_encodings,pratyush_face_encodings]
known_face_names = ['Narendra Modi','Donald Trump','Pratyush Sen']
all_face_locations = []
all_face_encodings = []
all_face_names = []

while True:
    ret,current_frame = webcam_video_stream.read()
    #resize the image to 1/4th it's original size
    current_frame_small = cv2.resize(current_frame,(0,0),fx=0.25,fy=0.25)
    all_face_locations = face_recognition.face_locations(current_frame_small,number_of_times_to_upsample=2,model = 'hog')    
    all_face_encodings = face_recognition.face_encodings(current_frame_small,all_face_locations)
    for current_face_location,current_face_encoding in zip(all_face_locations,all_face_encodings):
        top_pos,right_pos,bottom_pos,left_pos = current_face_location
        all_matches = face_recognition.compare_faces(known_face_encodings,current_face_encoding)
        name_of_person = 'Unknown'
        if True in all_matches:
            first_match_index = all_matches.index(True)
            name_of_person = known_face_names[first_match_index]
        top_pos*=4
        right_pos*=4
        bottom_pos*=4
        left_pos*=4
        cv2.rectangle(current_frame,(left_pos,top_pos),(right_pos,bottom_pos),(255,0,0),2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(current_frame,name_of_person,(left_pos,bottom_pos),font,0.5,(255,255,255),1)
        printImage(current_frame,"Webcam video")
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam_video_stream.release()
cv2.destroyAllWindows()
        

    
    