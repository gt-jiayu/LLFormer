import cv2
import numpy as np

class Retinex:
    def __init__(self, img_path):
        self.img = cv2.imread(img_path)

    def ssr(self, size):
        res = self.img

        L_blur = cv2.GaussianBlur(res, (size, size), 3)
        log_R = np.log(res + 0.001) - np.log(L_blur + 0.001)

        return log_R

    def msr(self, scales):
        number = len(scales)
        res = self.img

        h, w = self.img.shape[:2]
        dst_R = np.zeros((h, w), dtype=np.float32)
        log_R = np.zeros((h, w), dtype=np.float32)

        for i in range(number):
            L_blur = cv2.GaussianBlur(res, (scales[i], scales[i]), 0)
            log_R += np.log(res + 0.001) - np.log(L_blur + 0.001)

        log_R = log_R / number
        cv2.normalize(log_R, dst_R, 0, 255, cv2.NORM_MINMAX)
        dst_R = cv2.convertScaleAbs(dst_R)
        dst = cv2.add(self.img, dst_R)

        return dst

def main():
    print('here')
    rex = Retinex('datasets/LOLdataset/eval15/high/55.png')
    cv2.imshow('src', rex.img)
    cv2.waitKey(0)
    cv2.imshow('result', rex.ssr(3))
    cv2.waitKey(0)

if __name__ == "__main__":
    main()
