import cv2 as cv

face_cascade = cv.CascadeClassifier('haar_face.xml')
eye_cascade = cv.CascadeClassifier('haar_eye.xml')

cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("HATA: Kamera kamera acilamadi.")
    exit()
print("cikmak icin 'c' ye basin")

bekleme_sayaci = 0
uyari_yazildi = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("Video akisi bitti veya kesildi.")
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,scaleFactor = 1.1, minNeighbors = 13)  
    
    for (x, y, w, h) in faces: 
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]  

        roi_gray_upper = roi_gray[0:int(h/2), 0:w]

        eyes = eye_cascade.detectMultiScale(roi_gray_upper, scaleFactor=1.1, minNeighbors=25)

        if len(eyes) > 0:
            bekleme_sayaci = 0
            for (ex, ey, ew, eh) in eyes:
                cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)
        else:
            bekleme_sayaci += 1
            if bekleme_sayaci > 80 and not uyari_yazildi:
                print("UYARI: Lutfen gozlugunuzu cikarin!")
                uyari_yazildi = True
       
    cv.imshow('Face and Eye Detection', frame)

    if cv.waitKey(1) & 0xFF==ord('c'):
        break

cap.release()
cv.destroyAllWindows()
