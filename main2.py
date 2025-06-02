import cv2
import numpy as np

# ------------------- 第一部分：从原始图片提取并绘制初始轮廓 -------------------
# 读取原始图像
img = cv2.imread('rock1111.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# 寻找初始轮廓
contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

i = 0
# 在原图上绘制初始轮廓
for contour in contours:
    if i == 0:
        i = 1
        continue
    approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
    cv2.drawContours(img, [contour], 0, (0, 0, 255), 2)
    M = cv2.moments(contour)
    if M['m00'] != 0.0:
        x = int(M['m10'] / M['m00'])
        y = int(M['m01'] / M['m00'])
    # 根据需要可添加对特定边数(如四边形、六边形)的处理
    # 这里暂时为空

# ------------------- 第二部分：直接在处理过的图像上寻找最大和次大轮廓 -------------------
# 使用第一部分已有的 img，不再另存为 rock1111.png
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 200)

# 寻找最终的轮廓（外部轮廓）
contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# 按轮廓面积从大到小排序
sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

# 确保有足够的轮廓来获取最大和次大轮廓
if len(sorted_contours) >= 2:
    largest_contour = sorted_contours[0]  # 最大轮廓
    second_largest_contour = sorted_contours[1]  # 次大轮廓

    # 计算两个轮廓的周长
    largest_perimeter = cv2.arcLength(largest_contour, True)
    second_largest_perimeter = cv2.arcLength(second_largest_contour, True)

    # 计算周长差异
    perimeter_difference = abs(largest_perimeter - second_largest_perimeter)

    # 基于周长差异评估破碎机状态
    if perimeter_difference > 20:
        status = "существует аномальное отклонение"
        status_color = (0, 0, 255)  # 红色表示异常
    else:
        status = "состояние нормальное"
        status_color = (0, 255, 0)  # 绿色表示正常
 #if存在异常偏差，else状态正常
    # 输出评估结果
    print(f"максимальная длина контура: {largest_perimeter:.2f}")
    print(f"вторичная максимальная длина контура: {second_largest_perimeter:.2f}")
    print(f"разница в длине контуров: {perimeter_difference:.2f}")
    print(f"оценка состояния дробилки: {status}")
    #最大轮廓周长，次大轮廓周长，周长差异，破碎机状态
    # 在图像上绘制最大轮廓（蓝色）
    cv2.drawContours(img, [largest_contour], -1, (255, 0, 0), 2)
    # 在图像上绘制次大轮廓（绿色）
    cv2.drawContours(img, [second_largest_contour], -1, (0, 255, 0), 2)

    # 在图像上显示状态评估结果
    cv2.putText(img, status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
else:
    print("警告：未检测到足够的轮廓进行分析")

# 显示最终结果
cv2.imshow('破碎机状态评估结果', img)
cv2.waitKey(0)
cv2.destroyAllWindows()