import cv2

# change your path; 30 is fps; (2304,1296) is screen size
videoWriter = cv2.VideoWriter('RGB.mp4', cv2.VideoWriter_fourcc(*'MJPG'), 10, (960,540))

for i in range(1, 158):
    # load pictures from your path
    img = cv2.imread('./pic/' + str(i) + 'rgb' + '.jpg')
    img = cv2.resize(img, (960, 540))
    videoWriter.write(img)

videoWriter.release()

