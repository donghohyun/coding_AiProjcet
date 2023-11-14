
import cv2
import numpy as np

class ImageProcessor:

    def img_processed(self, img_ori):
        # 원본이미지를 이진화 작업 후 외각선에서 꼭짓점을 구한다.
        gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # 웹캠으로 사진을 찍을때 이미지가 뒤집혀 보여 이를 해결하기 위해 이미지를 반전하여 보여준다.
        contours = sorted(contours, key=lambda x: cv2.arcLength(x, True), reverse=True)
        # 주어진 배경에서 촬영 시 배경을 제외한 제일 큰 외각선을 선택한다(교구 전체 모습)
        largest_contour = contours[1]

        # 교구의 꼭짓점에서 각 꼭짓점의 위치가 임의로 저장되지 않도록 원하는 방향에 맞춰 저장한다.
        epsilon = 0.04 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)

        edge_point = []
        for point in approx:
            x, y = point[0]
            edge_point.append((x, y))

        left_top_screen = (float('inf'), float('inf'))
        right_top_screen = (float('-inf'), float('inf'))
        left_bottom_screen = (float('inf'), float('-inf'))
        right_bottom_screen = (float('-inf'), float('-inf'))

        for x, y in edge_point:
            if x + y < left_top_screen[0] + left_top_screen[1]:
                left_top_screen = (x, y)
            if x - y > right_top_screen[0] - right_top_screen[1]:
                right_top_screen = (x, y)
            if x - y < left_bottom_screen[0] - left_bottom_screen[1]:
                left_bottom_screen = (x, y)
            if x + y > right_bottom_screen[0] + right_bottom_screen[1]:
                right_bottom_screen = (x, y)
        # 구해진 꼭짓점 좌표를 기준으로 각 꼭지점 좌표를 화면 전체로 퍼트려 이미지를 평면화 시킨다.
        src_points = np.float32([left_top_screen, right_top_screen, left_bottom_screen, right_bottom_screen])
        width, height = 800, 600
        dst_points = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
        perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        warped_image = cv2.warpPerspective(binary, perspective_matrix, (width, height))
        # 이미지 인식률을 높이기 위해 이미지를  RGB값에서 특정값을 기준으로 이진화 한다.
        RGB_img = cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB)
        R_img1, G_img1, B_img1 = cv2.split(RGB_img)
        
        N = 150 # 해당 값을 기준으로 큰것은 255로 작은것은 0으로 반환하여 이미지 이진화

        for h in range(RGB_img.shape[0]):
            for w in range(RGB_img.shape[1]):
                if(np.int32(R_img1[h, w]) > N):
                    R_img1[h, w] = G_img1[h, w] = B_img1[h, w] = 255
                else:
                    R_img1[h, w] = G_img1[h, w] = B_img1[h, w] = 0

        RGB_img[:, :, 0] = R_img1
        RGB_img[:, :, 1] = G_img1
        RGB_img[:, :, 2] = B_img1

        return RGB_img

    def img_crop_processed(self, img, x_val, y_val, w_val, h_val):
        img_crop = img[y_val:y_val+h_val, x_val:x_val+w_val]
        # 모델에 이미지를 넣기 위해 이미지 사이즈 재조정
        img_crop_resize = cv2.resize(img_crop, (480, 480))
        gray_block = cv2.cvtColor(img_crop_resize, cv2.COLOR_BGR2GRAY)
        # 잡티를 제거하기 위해 약간의 블러를 설정한다.
        img_blurred = cv2.GaussianBlur(gray_block, ksize=(5, 5), sigmaX=0)
        # 블럭의 색이 다르더라도 인식가능하게끔 외각선을 구한다.
        img_blur_thresh = cv2.adaptiveThreshold(
                                img_blurred,
                                maxValue=1,
                                adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                thresholdType=cv2.THRESH_BINARY_INV,
                                blockSize=19,
                                C=9
                            )
        return img_blur_thresh

