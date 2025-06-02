import sys
import cv2
import numpy as np
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout,
                             QHBoxLayout, QFileDialog, QWidget, QSplitter, QFrame,
                             QStatusBar, QMessageBox)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt


class CrusherMonitoringApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.image_path = None
        self.processed_image = None
        self.original_image = None
        self.contour_image_path = None  # 保存带有红色轮廓的图片路径

    def initUI(self):
        # 设置窗口基本属性
        self.setWindowTitle('破碎机状态监控系统')
        self.setGeometry(100, 100, 1200, 800)

        # 创建中央部件和布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # 创建上部控制区域
        control_layout = QHBoxLayout()

        # 上传按钮
        self.upload_btn = QPushButton('上传图片')
        self.upload_btn.clicked.connect(self.upload_image)
        self.upload_btn.setMinimumHeight(40)

        # 分析按钮
        self.analyze_btn = QPushButton('分析破碎机状态')
        self.analyze_btn.clicked.connect(self.analyze_image)
        self.analyze_btn.setMinimumHeight(40)
        self.analyze_btn.setEnabled(False)  # 初始状态禁用

        # 将按钮添加到控制布局
        control_layout.addWidget(self.upload_btn)
        control_layout.addWidget(self.analyze_btn)

        # 添加控制布局到主布局
        main_layout.addLayout(control_layout)

        # 创建分割器用于图像显示区域
        splitter = QSplitter(Qt.Horizontal)

        # 左侧原始图像显示区域
        left_frame = QFrame()
        left_frame.setFrameShape(QFrame.StyledPanel)
        left_layout = QVBoxLayout(left_frame)
        self.original_image_label = QLabel('原始图像')
        self.original_image_label.setAlignment(Qt.AlignCenter)
        self.original_image_display = QLabel()
        self.original_image_display.setAlignment(Qt.AlignCenter)
        self.original_image_display.setMinimumSize(500, 500)
        self.original_image_display.setStyleSheet("border: 1px solid #CCCCCC; background-color: #F5F5F5;")
        left_layout.addWidget(self.original_image_label)
        left_layout.addWidget(self.original_image_display)

        # 右侧处理后图像显示区域
        right_frame = QFrame()
        right_frame.setFrameShape(QFrame.StyledPanel)
        right_layout = QVBoxLayout(right_frame)
        self.processed_image_label = QLabel('处理后图像')
        self.processed_image_label.setAlignment(Qt.AlignCenter)
        self.processed_image_display = QLabel()
        self.processed_image_display.setAlignment(Qt.AlignCenter)
        self.processed_image_display.setMinimumSize(500, 500)
        self.processed_image_display.setStyleSheet("border: 1px solid #CCCCCC; background-color: #F5F5F5;")
        right_layout.addWidget(self.processed_image_label)
        right_layout.addWidget(self.processed_image_display)

        # 添加左右框架到分割器
        splitter.addWidget(left_frame)
        splitter.addWidget(right_frame)

        # 添加分割器到主布局
        main_layout.addWidget(splitter)

        # 创建状态栏用于显示分析结果
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage('请上传图片开始分析')

        # 创建结果展示区域
        results_layout = QVBoxLayout()
        self.results_label = QLabel('分析结果将在此显示')
        self.results_label.setAlignment(Qt.AlignCenter)
        self.results_label.setStyleSheet("background-color: #EFEFEF; padding: 10px; border-radius: 5px;")
        results_layout.addWidget(self.results_label)

        # 添加结果布局到主布局
        main_layout.addLayout(results_layout)

    def upload_image(self):
        """打开文件对话框选择图片"""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "图片文件 (*.png *.jpg *.jpeg *.bmp);;所有文件 (*)", options=options
        )

        if file_path:
            self.image_path = file_path
            self.original_image = cv2.imread(file_path)

            if self.original_image is None:
                QMessageBox.critical(self, "错误", "无法加载图片，请检查文件是否有效")
                return

            # 更新状态栏
            self.status_bar.showMessage(f'已加载图片: {file_path}')

            # 在左侧显示原始图片
            self.display_image(self.original_image, self.original_image_display)

            # 启用分析按钮
            self.analyze_btn.setEnabled(True)
            
            # 清除之前的轮廓图片路径
            self.contour_image_path = None

    def analyze_image(self):
        """分析图像并显示结果"""
        if self.original_image is None:
            QMessageBox.warning(self, "警告", "请先上传图片")
            return

        # 复制原始图像用于处理
        img = self.original_image.copy()

        # 转换为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # 寻找初始轮廓
        contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # 过滤掉太小的轮廓
        filtered_contours = []
        for contour in contours:
            if cv2.contourArea(contour) > 100:  # 面积阈值，可以根据需要调整
                filtered_contours.append(contour)

        # 在图像上绘制红色轮廓
        red_contour_img = self.original_image.copy()
        cv2.drawContours(red_contour_img, filtered_contours, -1, (0, 0, 255), 2)  # 红色轮廓

        # 保存带有红色轮廓的图片
        if self.image_path:
            # 获取原始图片的目录和文件名
            dir_name = os.path.dirname(self.image_path)
            file_name = os.path.basename(self.image_path)
            name, ext = os.path.splitext(file_name)
            
            # 创建带有轮廓的图片的文件路径
            self.contour_image_path = os.path.join(dir_name, f"{name}_contour{ext}")
            
            # 保存带有轮廓的图片
            cv2.imwrite(self.contour_image_path, red_contour_img)
            
            # 显示保存成功的消息
            self.status_bar.showMessage(f'已保存红色轮廓图片: {self.contour_image_path}')
        
        # 使用保存的轮廓图片进行后续分析
        if self.contour_image_path and os.path.exists(self.contour_image_path):
            # 读取带有红色轮廓的图片
            contour_img = cv2.imread(self.contour_image_path)
            
            # 转换为HSV颜色空间以便更容易分离颜色
            hsv = cv2.cvtColor(contour_img, cv2.COLOR_BGR2HSV)
            
            # 定义红色的HSV范围
            lower_red = np.array([0, 100, 100])
            upper_red = np.array([10, 255, 255])
            red_mask1 = cv2.inRange(hsv, lower_red, upper_red)
            
            lower_red = np.array([160, 100, 100])
            upper_red = np.array([180, 255, 255])
            red_mask2 = cv2.inRange(hsv, lower_red, upper_red)
            
            # 合并两个红色掩码
            red_mask = red_mask1 + red_mask2
            
            # 寻找红色区域的轮廓
            red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 在原始图像上寻找蓝色轮廓
            # 使用边缘检测
            gray = cv2.cvtColor(contour_img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 200)

            # 寻找外部轮廓
            contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            # 按轮廓面积排序
            sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

            result_text = ""
            
            # 寻找蓝色轮廓
            # 定义蓝色的HSV范围
            hsv = cv2.cvtColor(contour_img, cv2.COLOR_BGR2HSV)
            lower_blue = np.array([100, 50, 50])
            upper_blue = np.array([130, 255, 255])
            blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
            
            # 寻找蓝色区域的轮廓
            blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 过滤掉太小的蓝色轮廓
            filtered_blue_contours = []
            for contour in blue_contours:
                if cv2.contourArea(contour) > 50:  # 面积阈值，可以根据需要调整
                    filtered_blue_contours.append(contour)
            
            # 确保有足够的轮廓进行分析
            if len(sorted_contours) >= 2:
                largest_contour = sorted_contours[0]  # 最大轮廓
                second_largest_contour = sorted_contours[1]  # 次大轮廓

                # 计算轮廓周长
                largest_perimeter = cv2.arcLength(largest_contour, True)
                second_largest_perimeter = cv2.arcLength(second_largest_contour, True)

                # 计算周长差异
                perimeter_difference = abs(largest_perimeter - second_largest_perimeter)

                # 基于周长差异评估破碎机状态
                if perimeter_difference > 20:
                    status = "存在异常偏差"
                    status_color = (0, 0, 255)  # 红色表示异常
                else:
                    status = "状态正常"
                    status_color = (0, 255, 0)  # 绿色表示正常

                # 在图像上绘制最大轮廓（蓝色）
                cv2.drawContours(contour_img, [largest_contour], -1, (255, 0, 0), 2)
                # 在图像上绘制次大轮廓（绿色）
                cv2.drawContours(contour_img, [second_largest_contour], -1, (0, 255, 0), 2)
                
                # 在图像上绘制检测到的蓝色轮廓（黄色）
                if filtered_blue_contours:
                    cv2.drawContours(contour_img, filtered_blue_contours, -1, (0, 255, 255), 2)

                # 在图像上显示状态评估结果
                cv2.putText(contour_img, status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

                # 准备结果文本
                result_text = f"""
                最大轮廓周长: {largest_perimeter:.2f}
                次大轮廓周长: {second_largest_perimeter:.2f}
                轮廓周长差异: {perimeter_difference:.2f}
                检测到的蓝色轮廓数量: {len(filtered_blue_contours)}
                破碎机状态评估: {status}
                """

                # 更新状态栏
                self.status_bar.showMessage(f'破碎机状态: {status}, 蓝色轮廓数量: {len(filtered_blue_contours)}')
            else:
                result_text = "警告：未检测到足够的轮廓进行分析"
                self.status_bar.showMessage('分析失败：未检测到足够的轮廓')

            # 更新结果标签
            self.results_label.setText(result_text)

            # 保存处理后的图像（包含红色和蓝色轮廓）
            self.processed_image = contour_img
            
            # 保存最终处理后的图像（包含所有轮廓）
            if self.image_path:
                dir_name = os.path.dirname(self.image_path)
                file_name = os.path.basename(self.image_path)
                name, ext = os.path.splitext(file_name)
                final_image_path = os.path.join(dir_name, f"{name}_final{ext}")
                cv2.imwrite(final_image_path, contour_img)
                self.status_bar.showMessage(f'已保存最终处理图片: {final_image_path}')

            # 在右侧显示处理后的图像
            self.display_image(self.processed_image, self.processed_image_display)
        else:
            QMessageBox.warning(self, "警告", "保存或读取轮廓图片失败")

    def display_image(self, cv_img, label):
        """在指定的QLabel上显示OpenCV图像"""
        # 确保图像存在
        if cv_img is None:
            return

        # 调整大小以适应标签
        h, w = cv_img.shape[:2]
        label_w, label_h = label.width(), label.height()

        # 计算缩放比例以适应标签
        scale = min(label_w / w, label_h / h)
        new_w, new_h = int(w * scale), int(h * scale)

        # 调整图像大小
        resized_img = cv2.resize(cv_img, (new_w, new_h))

        # 转换颜色通道
        if len(cv_img.shape) == 3:  # 彩色图像
            rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_img.shape
            bytes_per_line = ch * w
            qt_img = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        else:  # 灰度图像
            h, w = resized_img.shape
            bytes_per_line = w
            qt_img = QImage(resized_img.data, w, h, bytes_per_line, QImage.Format_Grayscale8)

        # 将QImage转换为QPixmap并在标签上显示
        pixmap = QPixmap.fromImage(qt_img)
        label.setPixmap(pixmap)
        label.setAlignment(Qt.AlignCenter)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CrusherMonitoringApp()
    window.show()
    sys.exit(app.exec_())