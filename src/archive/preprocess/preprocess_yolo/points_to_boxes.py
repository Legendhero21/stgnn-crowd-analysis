import os, cv2

DATA_DIR = r"D:/stgnn_project/data/yolo_from_mat"
img_path = os.path.join(DATA_DIR, "images", "train", "img001001.jpg")
label_path = os.path.join(DATA_DIR, "labels", "train", "img001001.txt")

img = cv2.imread(img_path)
h, w = img.shape[:2]

with open(label_path) as f:
    for line in f:
        cls, x, y, bw, bh = map(float, line.split())
        xc, yc = x * w, y * h
        bw, bh = bw * w, bh * h
        x1 = int(xc - bw / 2)
        y1 = int(yc - bh / 2)
        x2 = int(xc + bw / 2)
        y2 = int(yc + bh / 2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow("preview", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
