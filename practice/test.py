import cv2
 
def main():
    path = "rtsp://admin:Chinsoft@sx3.7766.org:754/doc/page/config.asp"
    cap = cv2.VideoCapture(path)
    c = 0 
    while True:
        re, frame = cap.read()
        if(re==False): break
        if(c%300==0):
            #cv2.imwrite("./res/{}.png".format(c), frame)
            print("count is ",c)
        c += 1
    camera.release()
 
if __name__ == "__main__":
    main()