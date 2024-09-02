import cv2
import face_recognition
import warnings
warnings.filterwarnings("ignore")

def printImage(image,title):
    cv2.imshow(title,image)
    #cv2.waitKey(0)  # wait for a key press
    #cv2.destroyAllWindows()  # explicitly close the window
    
all_face_locations = []
webcam_video_stream = cv2.VideoCapture(0)
while True:
    ret,current_frame = webcam_video_stream.read()
    #resize the image to 1/4th it's original size
    current_frame_small = cv2.resize(current_frame,(0,0),fx=0.25,fy=0.25)
    all_face_locations = face_recognition.face_locations(current_frame_small,number_of_times_to_upsample=2,model = 'hog')
    for index,current_face_location in enumerate(all_face_locations):
        top_pos,right_pos,bottom_pos,left_pos =  current_face_location
        top_pos*=4
        right_pos*=4
        bottom_pos*=4
        left_pos*=4
        print('Found face at {} at top:{},right:{},bottom:{},left:{}'.format(index+1,top_pos,right_pos,bottom_pos,left_pos))
        cv2.rectangle(current_frame,(left_pos,top_pos),(right_pos,bottom_pos),(0,0,255),2)
    printImage(current_frame,"Webcam video")
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam_video_stream.release()
cv2.destroyAllWindows()
        

    
    