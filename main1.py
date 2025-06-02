import cv2

# ------------------- 第一部分：处理原始图像并绘制轮廓 -------------------
# 读取原始图像
img = cv2.imread('rock1111.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# 寻找轮廓
contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

i = 0
# 在原图上绘制找到的轮廓
for contour in contours:
    if i == 0:
        i = 1
        continue
    approx = cv2.approxPolyDP(
        contour, 0.01 * cv2.arcLength(contour, True), True)
    cv2.drawContours(img, [contour], 0, (0, 0, 255), 2)
    M = cv2.moments(contour)
    if M['m00'] != 0.0:
        x = int(M['m10'] / M['m00'])
        y = int(M['m01'] / M['m00'])
    elif len(approx) == 4:
        cv2.putText(img, '', (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    elif len(approx) == 6:
        cv2.putText(img, '', (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    else:
        pass  # 根据需要添加其他轮廓的处理

# 显示第一阶段处理结果
cv2.imshow('shapes', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# ------------------- 第二部分：使用上一步的结果直接进行后续处理 -------------------
# 不再读取rock1111.png，而是直接使用img作为输入图像
original_image = img.copy()

gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 200)
contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.destroyAllWindows()

def get_contour_areas(contours):
    all_areas = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        all_areas.append(area)
    return all_areas

# 按面积从大到小排序轮廓
sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

# 取面积最大的和第五大的轮廓（假设列表有足够多的轮廓）
largest_item = sorted_contours[0]
largest_item2 = sorted_contours[4]

# 在图像上绘制选定的两个轮廓
cv2.drawContours(original_image, largest_item, -1, (255, 0, 0), 10)
cv2.drawContours(original_image, largest_item2, -1, (255, 0, 0), 10)

# 显示结果
cv2.imshow('Largest Object', original_image)
cv2.waitKey(0)
cv2.destroyAllWindows()