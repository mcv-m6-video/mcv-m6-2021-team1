import cv2
import utils
import imageio
from matplotlib import pyplot as plt

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

        for track in tracking[0]['f_'+str(fr_n)]:
            bbox = track['bbox']
            cv2.imshow('bbox_origen', frame[bbox[1]:bbox[3], bbox[0]:bbox[2]])
            cv2.waitKey(0)
            neighb = 2

            cap_aux = cv2.VideoCapture(f'C:\\Users\\Carmen\\CVMaster\\M6\\aic19-track1-mtmc-train\\train\\S01\\c00{neighb+1}\\vdo.avi')
            for fr_check in range(fr_n-10, fr_n+10):
                cap_aux.set(1, fr_check)
                _, fr_aux = cap_aux.read()
                for track_aux in tracking[neighb]['f_'+str(fr_check)]:
                    bb_aux = track_aux['bbox']
                    cv2.imshow('compare', fr_aux[bb_aux[1]:bb_aux[3], bb_aux[0]:bb_aux[2]])
                    cv2.waitKey(0)
                    
                    H1 = cv2.calcHist([frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]],[1],None,[256],[0,256])
                    plt.plot(H1,color = 'b')
                    H2 = cv2.calcHist([fr_aux[bb_aux[1]:bb_aux[3], bb_aux[0]:bb_aux[2]]],[2],None,[256],[0,256])
                    plt.plot(H2,color = 'r')
                    plt.title(cv2.compareHist(H1, H2, cv2.HISTCMP_INTERSECT))
                    plt.xlim([0,256])
                    plt.show()
                   
            cap_aux.release()



    fr_n += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

cap.release()
cv2.destroyAllWindows()