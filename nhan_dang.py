import cv2
import numpy as np
import datetime

video = cv2.VideoCapture('http://192.168.43.1:8080/video')
nhan_dien_khuon_mat = cv2.face.LBPHFaceRecognizer_create()
nhan_dien_khuon_mat.read('hoc/hoc.yml')
duong_dan = "haarcascade_frontalface_default.xml"
khuon_mat = cv2.CascadeClassifier(duong_dan)

font = cv2.FONT_HERSHEY_SIMPLEX
ten = {0: "Unknown", 1: "Hiếu", 2: "Nghĩa", 3: "Hương", 4: "Trường"}

ghi_log_file = open("log.txt", "a", encoding="utf-8")  # Mở file log với mã UTF-8 để ghi lịch sử

while True:
    ret, khung_hinh = video.read()

    khung_hinh_thay_doi_kich_thuoc = cv2.resize(khung_hinh, None, fx=0.2, fy=0.3)
    img = cv2.flip(khung_hinh_thay_doi_kich_thuoc, -1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Phát hiện khuôn mặt trong khung hình
    khuon_mats = khuon_mat.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in khuon_mats:
        id, do_tin_cay = nhan_dien_khuon_mat.predict(gray[y:y + h, x:x + w])

        if do_tin_cay < 100:
            nguoi_nhan_dien = ten[id]
            do_tin_cay_str = f"Do tin cay: {round(100 - do_tin_cay)}%"
        else:
            nguoi_nhan_dien = "Unknown"
            do_tin_cay_str = "Unknown"

        cv2.putText(img, nguoi_nhan_dien, (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, do_tin_cay_str, (x + 5, y + h - 5), font, 1, (255, 255, 255), 2)

        # Lưu ảnh của người được nhận diện và ghi log
        if nguoi_nhan_dien != "Unknown":
            anh_khuon_mat = img[y:y + h, x:x + w]
            ten_file = f"{nguoi_nhan_dien}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
            cv2.imwrite(ten_file, anh_khuon_mat)

            log_entry = f"{datetime.datetime.now()} - Người: {nguoi_nhan_dien}, Do tin cay: {round(100 - do_tin_cay)}%\n"
            ghi_log_file.write(log_entry)

    cv2.imshow("Nhan Dien Khuon Mat", img)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

ghi_log_file.close()
video.release()
cv2.destroyAllWindows()
