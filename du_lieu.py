import cv2
import os
import time


def chụp_và_lưu_khuôn_mặt(id_khuôn_mặt):
    video = cv2.VideoCapture('http://192.168.43.1:8080/video')
    bộ_phân_loại_khuôn_mặt = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    print(f"\nBắt đầu chụp ảnh khuôn mặt cho ID {id_khuôn_mặt}...")

    folder_path = f"du_lieu/Nguoi_thu_{id_khuôn_mặt}"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    count = 0
    max_images = 5

    while True:
        ret, khung_hình = video.read()

        if not ret:
            break

        # Thay đổi kích thước hiển thị của khung hình
        khung_hình = cv2.resize(khung_hình, None, fx=0.2, fy=0.3)  # Thay đổi kích thước (640, 480) thành kích thước mong muốn

        ảnh = cv2.flip(khung_hình, -1)
        ảnh_xám = cv2.cvtColor(ảnh, cv2.COLOR_BGR2GRAY)

        khuôn_mặt = bộ_phân_loại_khuôn_mặt.detectMultiScale(ảnh_xám, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in khuôn_mặt:
            cv2.rectangle(ảnh, (x, y), (x + w, y + h), (255, 0, 0), 2)
            count += 1

            if count <= max_images:
                cv2.imwrite(f"{folder_path}/User_{id_khuôn_mặt}_{count}.jpg", ảnh_xám[y:y + h, x:x + w])

        cv2.imshow("Chụp khuôn mặt", ảnh)

        if count >= max_images:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print(f"\nĐã chụp xong ảnh khuôn mặt cho ID {id_khuôn_mặt}.")
    video.release()
    cv2.destroyAllWindows()


def chụp_nhiều_khuôn_mặt():
    while True:
        try:
            số_người_dùng = int(input('\nNhập số lượng người dùng cần chụp (số nguyên) <Enter> ==> '))
            break
        except ValueError:
            print("Vui lòng nhập một số nguyên.")

    for i in range(số_người_dùng):
        id_khuôn_mặt = i + 1
        chụp_và_lưu_khuôn_mặt(id_khuôn_mặt)
        time.sleep(10)  # Tạm dừng 10 giây giữa mỗi ID


chụp_nhiều_khuôn_mặt()
