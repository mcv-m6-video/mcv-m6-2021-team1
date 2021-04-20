import cv2
import utils
import imageio

tracking = []

#change tu compress list
for camera in range(1,6):
    print(camera)
    path_track = f'C:\\Users\\Carmen\\CVMaster\\M6\\aic19-track1-mtmc-train\\train\\S01\\c00{camera}\\gt\\gt.txt'
    dic = utils.parse_aicity_rects(path_track)
    tracking.append(dic)

# print(tracking[1]['f_2067'])
# for track in tracking[1]['f_2067']:
#     track['bbox']
#     cv2.imshow(f'C:\\Users\\Carmen\\CVMaster\\M6\\aic19-track1-mtmc-train\\train\\S01\\c00{camera}\\vdo.avi')

camera = 1
cap = cv2.VideoCapture(f'C:\\Users\\Carmen\\CVMaster\\M6\\aic19-track1-mtmc-train\\train\\S01\\c00{camera}\\vdo.avi')

# cap.set(2, 2)
# ret, frame = cap.read()
# print(frame)
# # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# cv2.imshow('frame', frame)
# cv.waitKey(0)
fr_n = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    if fr_n > 350 and fr_n < 400:
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame', frame)
        cv2.waitKey(0)

        for track in tracking[1]['f_'+str(fr_n)]:
            bbox = track['bbox']
            cv2.imshow('bbox', frame[bbox[1]:bbox[3], bbox[0]:bbox[2]])
            cv2.waitKey(0)
    fr_n += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

cap.release()
cv2.destroyAllWindows()