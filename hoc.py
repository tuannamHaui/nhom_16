# import cv2
# import os
# import numpy as np
#
#
# # Hàm timvahoc xử lý tệp ảnh trong thư mục
# def timvahoc(p):
#     f = []
#     ids = []
#
#     for root, dirs, files in os.walk(p):
#         for file in files:
#             if file.endswith(".jpg") or file.endswith(".png"):
#                 img_path = os.path.join(root, file)
#                 img = cv2.imread(img_path)
#
#                 if img is not None:
#                     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#                     f.append(gray)
#                     ids.append(int(os.path.split(img_path)[-1].split("_")[1].split(".")[0]))
#
#     return f, ids
#
# # Đường dẫn đến thư mục chứa ảnh khuôn mặt của bạn
# p = 'du_lieu'
#
# print("\n Học dữ liệu...")
# faces, ids = timvahoc(p)
#
# recognizer = cv2.face_LBPHFaceRecognizer.create()
# recognizer.train(faces, np.array(ids))
#
# recognizer.save('hoc/hoc.yml')
#
# print("\n {0} khuôn mặt được tìm thấy và học. Thoát.".format(len(np.unique(ids))))
import cv2
import os
import numpy as np

def timvahoc(p):
    f = []
    ids = []
    processed_images = {}

    for root, dirs, files in os.walk(p):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png"):
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path)

                if img is not None:
                    img_bytes = img.tobytes()  # Thay thế tostring() bằng tobytes()

                    img_hash = hash(img_bytes)

                    if img_hash not in processed_images:
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        f.append(gray)
                        ids.append(int(os.path.split(img_path)[-1].split("_")[1].split(".")[0]))
                        processed_images[img_hash] = True

    return f, ids

# Đường dẫn đến thư mục chứa ảnh khuôn mặt của bạn
p = 'du_lieu'

print("\n Học dữ liệu...")
faces, ids = timvahoc(p)

recognizer = cv2.face_LBPHFaceRecognizer.create()
recognizer.train(faces, np.array(ids))

recognizer.save('hoc/hoc.yml')

print("\n {0} khuôn mặt được tìm thấy và học. Thoát.".format(len(np.unique(ids))))
