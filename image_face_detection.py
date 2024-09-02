import cv2
import face_recognition
import warnings
warnings.filterwarnings("ignore")

def printImage(image,title):
    cv2.imshow(title,image)
    cv2.waitKey(0)  # wait for a key press
    cv2.destroyAllWindows()  # explicitly close the window
    
image_to_detect = cv2.imread('/media/pratyush/Work/udemy/modi-trump.jpg')
all_face_locations = face_recognition.face_locations(image_to_detect,model = 'hog')
print('There are {} no of faces in this image'.format(len(all_face_locations)))

for index,current_face_location in enumerate(all_face_locations):
    top_pos,right_pos,bottom_pos,left_pos =  current_face_location
    print('Found face at {} at top:{},right:{},bottom:{},left:{}'.format(index+1,top_pos,right_pos,bottom_pos,left_pos))
    current_face_image = image_to_detect[top_pos:bottom_pos,left_pos:right_pos]
    printImage(current_face_image, "Face No.: "+str(index+1))

    
