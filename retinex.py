import cv2
import numpy as np
from PIL import Image


class Retinex:
    def __init__(self, img_path):
        self.img = cv2.cvtColor(np.array(Image.open(img_path).convert('RGB')), cv2.COLOR_RGB2BGR)

    def ssr(self):
        res = self.img
        # 高斯模糊，高斯核设置为0，高斯核的标准差设置为80、自动计算核的大小
        L_blur = cv2.GaussianBlur(res, (0, 0), 80)
        # Retinex公式：log_R = log_S - log(Gauss(S))；加上0.001，防止log计算时有0值出错
        log_R = np.log(res + 0.001) - np.log(L_blur + 0.001)
        # exp运算得到R
        R = np.exp(log_R)

        return R

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
    rex = Retinex('datasets/LOLdataset/eval15/high/111.png')
    cv2.imshow('src', rex.img)
    cv2.waitKey(1000)
    cv2.imshow('result', rex.ssr())
    cv2.waitKey(0)

if __name__ == "__main__":
    main()
