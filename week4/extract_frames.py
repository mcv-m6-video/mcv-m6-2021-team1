import cv2
import os 

# Opens the Video file
print(os.listdir('./videos_stab'))
cap= cv2.VideoCapture('./videos_stab/SPOILER_wiki.avi')
i=0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    cv2.imwrite('./videos_stab/wiki/frame_'+str(i)+'.jpg',frame)
    i+=1

cap.release()
cv2.destroyAllWindows()
