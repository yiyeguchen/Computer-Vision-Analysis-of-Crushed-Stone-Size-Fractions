import cv2
import numpy as np
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout,
                             QHBoxLayout, QFileDialog, QWidget, QSplitter, QFrame,
                             QStatusBar, QMessageBox)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt
import sys


def cv_imread(file_path):
    try:
        img_np = np.fromfile(file_path, dtype=np.uint8)
        return cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Ошибка чтения изображения: {file_path}, Ошибка: {str(e)}")
        return None


def cv_imwrite(file_path, img):
    try:
        dir_name = os.path.dirname(file_path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)

        ext = os.path.splitext(file_path)[1]
        result, encoded_img = cv2.imencode(ext, img)

        if result:
            with open(file_path, mode='w+b') as f:
                encoded_img.tofile(f)
            return True
        else:
            print(f"Ошибка кодирования изображения: {file_path}")
            return False
    except Exception as e:
        print(f"Ошибка сохранения изображения: {file_path}, Ошибка: {str(e)}")
        return False


class ContourDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.image_path = None
        self.original_image = None
        self.contour_image_path = None
        self.processed_results = {}

    def create_styled_button(self, text, callback):
        btn = QPushButton(text)
        btn.clicked.connect(callback)
        btn.setMinimumHeight(40)

        btn_font = QFont("Microsoft YaHei", 10, QFont.Medium)
        btn.setFont(btn_font)

        btn.setStyleSheet("""
            QPushButton {
                background-color: #4A86E8;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #3D76C9;
            }
            QPushButton:pressed {
                background-color: #2C5AA0;
            }
            QPushButton:disabled {
                background-color: #CCCCCC;
                color: #888888;
            }
        """)

        return btn

    def create_styled_label(self, text, is_title=False, is_result=False):
        label = QLabel(text)
        label.setAlignment(Qt.AlignCenter)

        if is_title:
            title_font = QFont("Microsoft YaHei", 12, QFont.Bold)
            label.setFont(title_font)
            label.setStyleSheet("color: #2C5AA0; margin-bottom: 8px;")
        elif is_result:
            result_font = QFont("Microsoft YaHei", 10)
            label.setFont(result_font)
            label.setStyleSheet("background-color: #EFEFEF; padding: 10px; border-radius: 5px; color: #333333;")
        else:
            normal_font = QFont("Microsoft YaHei", 10)
            label.setFont(normal_font)

        return label

    def initUI(self):
        app = QApplication.instance()
        font = QFont("Microsoft YaHei", 10)
        app.setFont(font)

        self.setWindowTitle('Система обнаружения и анализа контуров')
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet("background-color: #F8F8F8;")

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(10)

        control_layout = QHBoxLayout()
        control_layout.setSpacing(10)

        self.upload_btn = self.create_styled_button('Загрузить изображение', self.upload_image)
        self.process_btn = self.create_styled_button('Обработать изображение', self.process_and_save_contours)
        self.process_btn.setEnabled(False)
        self.precise_btn = self.create_styled_button('Показать контуры', self.extract_precise_contours)
        self.precise_btn.setEnabled(False)

        control_layout.addWidget(self.upload_btn)
        control_layout.addWidget(self.process_btn)
        control_layout.addWidget(self.precise_btn)

        main_layout.addLayout(control_layout)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(2)
        splitter.setStyleSheet("QSplitter::handle { background-color: #CCCCCC; }")

        left_frame = QFrame()
        left_frame.setFrameShape(QFrame.StyledPanel)
        left_frame.setStyleSheet("QFrame { background-color: white; border-radius: 5px; }")
        left_layout = QVBoxLayout(left_frame)

        self.original_image_label = self.create_styled_label('Исходное изображение', is_title=True)
        self.original_image_display = QLabel()
        self.original_image_display.setAlignment(Qt.AlignCenter)
        self.original_image_display.setMinimumSize(500, 500)
        self.original_image_display.setStyleSheet("""
            border: 1px solid #CCCCCC;
            background-color: #F5F5F5;
            border-radius: 4px;
            padding: 2px;
        """)

        left_layout.addWidget(self.original_image_label)
        left_layout.addWidget(self.original_image_display)

        right_frame = QFrame()
        right_frame.setFrameShape(QFrame.StyledPanel)
        right_frame.setStyleSheet("QFrame { background-color: white; border-radius: 5px; }")
        right_layout = QVBoxLayout(right_frame)

        self.processed_image_label = self.create_styled_label('Обработанное изображение', is_title=True)
        self.processed_image_display = QLabel()
        self.processed_image_display.setAlignment(Qt.AlignCenter)
        self.processed_image_display.setMinimumSize(500, 500)
        self.processed_image_display.setStyleSheet("""
            border: 1px solid #CCCCCC;
            background-color: #F5F5F5;
            border-radius: 4px;
            padding: 2px;
        """)

        right_layout.addWidget(self.processed_image_label)
        right_layout.addWidget(self.processed_image_display)

        splitter.addWidget(left_frame)
        splitter.addWidget(right_frame)

        main_layout.addWidget(splitter)

        results_layout = QVBoxLayout()
        self.results_label = self.create_styled_label('Результаты анализа будут отображены здесь', is_result=True)
        self.results_label.setAlignment(Qt.AlignLeft)
        self.results_label.setWordWrap(True)
        self.results_label.setTextFormat(Qt.RichText)
        self.results_label.setMinimumHeight(100)
        results_layout.addWidget(self.results_label)

        main_layout.addLayout(results_layout)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        status_font = QFont("Microsoft YaHei", 9)
        self.status_bar.setFont(status_font)
        self.status_bar.setStyleSheet("color: #555555;")
        self.status_bar.showMessage('Пожалуйста, загрузите изображение для начала анализа')

    def upload_image(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выбрать изображение", "", "Файлы изображений (*.png *.jpg *.jpeg *.bmp);;Все файлы (*)",
            options=options
        )

        if file_path:
            try:
                print(f"Попытка загрузки изображения: {file_path}")
                self.status_bar.showMessage(f'Загрузка изображения: {file_path}')

                if not os.path.exists(file_path):
                    QMessageBox.critical(self, "Ошибка", f"Файл не существует: {file_path}")
                    return

                file_path = os.path.abspath(file_path)
                self.image_path = file_path

                self.original_image = cv_imread(file_path)

                if self.original_image is None:
                    QMessageBox.critical(self, "Ошибка",
                                         f"Не удалось загрузить изображение: {file_path}\nУбедитесь, что формат файла правильный и он не поврежден")
                    return

                self.status_bar.showMessage(f'Изображение загружено: {file_path}')

                self.display_image(self.original_image, self.original_image_display)

                self.process_btn.setEnabled(True)

                self.contour_image_path = None

                self.precise_btn.setEnabled(False)
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Произошла ошибка при загрузке изображения: {str(e)}")
                print(f"Ошибка загрузки изображения: {str(e)}")

    def filter_contours(self, contours, min_area=200):
        filtered_by_area = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

        filtered_contours = []
        for cnt in filtered_by_area:
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)

            if area == 0:
                continue

            complexity = (perimeter ** 2) / (4 * np.pi * area) if area != 0 else 0

            if complexity > 4.0:
                print(f"Отфильтрован контур высокой сложности - Площадь: {area:.2f}, Сложность: {complexity:.2f}")
                continue

            epsilon = 0.02 * perimeter
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            if len(approx) > 12:
                print(
                    f"Отфильтрован контур со слишком большим количеством вершин - Площадь: {area:.2f}, Кол-во вершин: {len(approx)}")
                continue

            filtered_contours.append(cnt)

        return filtered_contours

    def process_and_save_contours(self):
        if self.original_image is None:
            QMessageBox.warning(self, "Предупреждение", "Пожалуйста, сначала загрузите изображение")
            return

        try:
            img = self.original_image.copy()

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

            kernel = np.ones((5, 5), np.uint8)
            threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
            threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
            threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            filtered_contours = self.filter_contours(contours, min_area=200)

            red_contour_img = self.original_image.copy()
            cv2.drawContours(red_contour_img, filtered_contours, -1, (0, 0, 255), 2)

            if self.image_path:
                try:
                    dir_name = os.path.dirname(self.image_path)
                    file_name = os.path.basename(self.image_path)
                    name, ext = os.path.splitext(file_name)

                    self.contour_image_path = os.path.join(dir_name, f"{name}_red_contour{ext}")

                    print(f"Попытка сохранения изображения контуров в: {self.contour_image_path}")

                    success = cv_imwrite(self.contour_image_path, red_contour_img)

                    if not success:
                        raise Exception("Не удалось сохранить изображение")

                    if not os.path.exists(self.contour_image_path):
                        raise Exception("Изображение не найдено после сохранения")

                    self.status_bar.showMessage(
                        f'Изображение с красными контурами сохранено: {self.contour_image_path}')

                    self.display_image(red_contour_img, self.processed_image_display)

                    self.precise_btn.setEnabled(True)

                    self.processed_results['red_contour_image'] = red_contour_img
                    self.processed_results['contours'] = filtered_contours
                except Exception as e:
                    error_msg = f"Произошла ошибка при сохранении изображения контуров: {str(e)}"
                    QMessageBox.warning(self, "Предупреждение", error_msg)
                    print(error_msg)
            else:
                QMessageBox.warning(self, "Предупреждение", "Не удалось сохранить изображение контуров")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Произошла ошибка при обработке изображения: {str(e)}")
            print(f"Ошибка обработки изображения: {str(e)}")

    def find_blue_contours(self):
        if not self.contour_image_path or not os.path.exists(self.contour_image_path):
            QMessageBox.warning(self, "Предупреждение",
                                f"Не удалось найти изображение контуров: {self.contour_image_path}\nПожалуйста, сначала создайте и сохраните изображение с красными контурами")
            return

        try:
            print(f"Попытка загрузки изображения контуров: {self.contour_image_path}")
            contour_img = cv_imread(self.contour_image_path)

            if contour_img is None:
                QMessageBox.critical(self, "Ошибка",
                                     f"Не удалось загрузить изображение контуров: {self.contour_image_path}")
                return

            result_img = contour_img.copy()

            hsv = cv2.cvtColor(contour_img, cv2.COLOR_BGR2HSV)

            lower_red1 = np.array([0, 120, 120])
            upper_red1 = np.array([8, 255, 255])
            red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)

            lower_red2 = np.array([165, 120, 120])
            upper_red2 = np.array([180, 255, 255])
            red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

            red_mask = red_mask1 + red_mask2

            lower_blue = np.array([105, 80, 80])
            upper_blue = np.array([125, 255, 255])
            blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

            kernel = np.ones((5, 5), np.uint8)
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
            blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
            blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)

            red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            print(f"Найдено красных контуров: {len(red_contours)}")

            blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            print(f"Найдено синих контуров: {len(blue_contours)}")

            filtered_red_contours = self.filter_contours(red_contours, min_area=200)
            filtered_blue_contours = self.filter_contours(blue_contours, min_area=150)

            print(f"Количество красных контуров после фильтрации: {len(filtered_red_contours)}")
            print(f"Количество синих контуров после фильтрации: {len(filtered_blue_contours)}")

            cv2.drawContours(result_img, filtered_blue_contours, -1, (255, 0, 0), 2)

            gray = cv2.cvtColor(contour_img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 200)

            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

            all_contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            all_contours = self.filter_contours(all_contours, min_area=200)
            print(f"Найдено всех контуров (после фильтрации): {len(all_contours)}")

            sorted_contours = sorted(all_contours, key=cv2.contourArea, reverse=True)

            result_text = ""
            if len(sorted_contours) >= 2:
                largest_contour = sorted_contours[0]
                second_largest_contour = sorted_contours[1]

                largest_perimeter = cv2.arcLength(largest_contour, True)
                second_largest_perimeter = cv2.arcLength(second_largest_contour, True)

                perimeter_difference = abs(largest_perimeter - second_largest_perimeter)

                cv2.drawContours(result_img, [largest_contour], -1, (255, 0, 0), 2)
                cv2.drawContours(result_img, [second_largest_contour], -1, (0, 255, 255), 2)

                if perimeter_difference > 20:
                    status = "Обнаружено аномальное отклонение"
                    status_color = (0, 0, 255)
                else:
                    status = "Нормальное состояние"
                    status_color = (0, 255, 0)

                blue_count = len(filtered_blue_contours)

                status_text_display = "Status: Anomaly" if status == "Обнаружено аномальное отклонение" else "Status: Normal"

                result_text = f"Состояние: {status}\n"
                result_text += f"Количество синих контуров: {blue_count}\n"
                result_text += f"Периметр наибольшего контура: {largest_perimeter:.2f}\n"
                result_text += f"Периметр второго по величине контура: {second_largest_perimeter:.2f}\n"
                result_text += f"Разница периметров: {perimeter_difference:.2f}"

            else:
                result_text = "Не найдено достаточного количества контуров для анализа"

            self.display_image(result_img, self.processed_image_display)
            self.results_label.setText(result_text)

            self.status_bar.showMessage("Анализ синих контуров завершен")

            if self.image_path:
                try:
                    dir_name = os.path.dirname(self.image_path)
                    file_name = os.path.basename(self.image_path)
                    name, ext = os.path.splitext(file_name)

                    result_image_path = os.path.join(dir_name, f"{name}_analysis_result{ext}")
                    print(f"Попытка сохранения результирующего изображения анализа в: {result_image_path}")

                    success = cv_imwrite(result_image_path, result_img)

                    if not success:
                        raise Exception("Не удалось сохранить результирующее изображение")

                    self.status_bar.showMessage(f"Результаты анализа сохранены: {result_image_path}")
                except Exception as e:
                    error_msg = f"Произошла ошибка при сохранении результирующего изображения: {str(e)}"
                    QMessageBox.warning(self, "Предупреждение", error_msg)
                    print(error_msg)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Произошла ошибка при анализе контуров: {str(e)}")
            print(f"Ошибка анализа контуров: {str(e)}")

    def analyze_contours(self, contours):
        if not contours:
            return {
                "count": 0,
                "avg_area": 0,
                "avg_perimeter": 0,
                "complexity": 0,
                "min_area": 0,
                "max_area": 0,
                "area_std": 0
            }

        areas = [cv2.contourArea(cnt) for cnt in contours]
        perimeters = [cv2.arcLength(cnt, True) for cnt in contours]

        complexities = []
        for i in range(len(contours)):
            if perimeters[i] > 0:
                complexity = (perimeters[i] ** 2) / (4 * np.pi * areas[i]) if areas[i] > 0 else 0
                complexities.append(complexity)
            else:
                complexities.append(0)

        features = {
            "count": len(contours),
            "avg_area": np.mean(areas) if areas else 0,
            "avg_perimeter": np.mean(perimeters) if perimeters else 0,
            "complexity": np.mean(complexities) if complexities else 0,
            "min_area": np.min(areas) if areas else 0,
            "max_area": np.max(areas) if areas else 0,
            "area_std": np.std(areas) if areas else 0
        }

        return features

    def evaluate_crusher_status(self, contour_features):
        status = "Нормальное"
        status_details = {}

        red_features = contour_features.get("red", {"count": 0})
        if red_features["count"] > 0:
            if red_features.get("complexity", 0) > 2.0:
                status = "Аномальное"
                status_details["Сложность красного контура"] = f"Высокая ({red_features['complexity']:.2f} > 2.0)"
            else:
                status_details["Сложность красного контура"] = f"Нормальная ({red_features.get('complexity', 0):.2f})"

            cv_red = red_features.get("area_std", 0) / red_features.get("avg_area", 1) if red_features.get("avg_area",
                                                                                                           0) > 0 else 0
            if cv_red > 0.5:
                status = "Аномальное"
                status_details["Вариация площади красного контура"] = f"Большая (CV = {cv_red:.2f} > 0.5)"
            else:
                status_details["Вариация площади красного контура"] = f"Нормальная (CV = {cv_red:.2f})"

        blue_features = contour_features.get("blue", {"count": 0})
        if blue_features["count"] > 0:
            blue_red_ratio = blue_features["count"] / red_features["count"] if red_features.get("count", 0) > 0 else 0
            if blue_red_ratio > 3.0:
                status = "Аномальное"
                status_details["Соотношение синих/красных контуров"] = f"Аномальное ({blue_red_ratio:.2f} > 3.0)"
            else:
                status_details["Соотношение синих/красных контуров"] = f"Нормальное ({blue_red_ratio:.2f})"

            if blue_features.get("avg_area", 0) < 100:
                status_details["Средняя площадь синего контура"] = f"Малая ({blue_features['avg_area']:.2f} < 100)"
            else:
                status_details[
                    "Средняя площадь синего контура"] = f"Нормальная ({blue_features.get('avg_area', 0):.2f})"

        green_features = contour_features.get("green", {"count": 0})
        if green_features["count"] > 0:
            if green_features.get("avg_area", 0) > 1000:
                status_details[
                    "Средняя площадь зеленого контура"] = f"Большая ({green_features['avg_area']:.2f} > 1000)"
            else:
                status_details[
                    "Средняя площадь зеленого контура"] = f"Нормальная ({green_features.get('avg_area', 0):.2f})"

        total_contours = sum(f.get("count", 0) for f in contour_features.values())
        if total_contours > 0:
            for color, features in contour_features.items():
                if features.get("count", 0) > 0:
                    color_ratio = features["count"] / total_contours
                    key_name = ""
                    if color == "red":
                        key_name = "Красных"
                    elif color == "blue":
                        key_name = "Синих"
                    elif color == "green":
                        key_name = "Зеленых"
                    else:
                        key_name = color.capitalize()
                    status_details[f"Пропорция {key_name} контуров"] = f"{color_ratio:.2%}"

        return status, status_details

    def display_image(self, cv_img, label):
        if cv_img is None:
            label.clear()
            label.setText("Нет изображения")
            return

        try:
            h, w = cv_img.shape[:2]
            label_w = label.width()
            label_h = label.height()

            if h == 0 or label_w <= 0 or label_h <= 0:
                label.clear()
                label.setText("Некорректный размер")
                return

            aspect_ratio = w / h
            if label_w / label_h > aspect_ratio:
                new_h = label_h
                new_w = int(new_h * aspect_ratio)
            else:
                new_w = label_w
                new_h = int(new_w / aspect_ratio)

            if new_w <= 0 or new_h <= 0:
                label.clear()
                label.setText("Ошибка масштабирования")
                return

            resized_img = cv2.resize(cv_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

            qt_image = None
            if len(resized_img.shape) == 3:
                rgb_image = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
                h_r, w_r, ch = rgb_image.shape
                bytes_per_line = ch * w_r
                qt_image = QImage(rgb_image.data, w_r, h_r, bytes_per_line, QImage.Format_RGB888)
            elif len(resized_img.shape) == 2:
                h_r, w_r = resized_img.shape
                bytes_per_line = w_r
                qt_image = QImage(resized_img.data, w_r, h_r, bytes_per_line, QImage.Format_Grayscale8)
            else:
                print("Неподдерживаемый формат изображения для отображения")
                label.clear()
                label.setText("Ошибка формата")
                return

            if qt_image is None or qt_image.isNull():
                print("Ошибка создания QImage")
                label.clear()
                label.setText("Ошибка QImage")
                return

            pixmap = QPixmap.fromImage(qt_image)

            label.setPixmap(pixmap)
        except Exception as e:
            print(f"Ошибка в display_image: {e}")
            label.clear()
            label.setText("Ошибка отображения")

    def extract_precise_contours(self):
        if not self.image_path:
            QMessageBox.warning(self, "Предупреждение", "Пожалуйста, сначала загрузите изображение")
            return

        try:
            original_img = cv_imread(self.image_path)
            if original_img is None:
                QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить исходное изображение: {self.image_path}")
                return

            result_img = original_img.copy()
            height, width = original_img.shape[:2]

            clean_result_img = np.zeros((height, width, 3), dtype=np.uint8)

            print("Применение K-means для сегментации изображения...")
            Z = original_img.reshape((-1, 3))
            Z = np.float32(Z)

            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            K = 8

            _, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

            centers = np.uint8(centers)
            segmented_img = centers[labels.flatten()].reshape(original_img.shape)

            gray = cv2.cvtColor(segmented_img, cv2.COLOR_BGR2GRAY)
            _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

            kernel = np.ones((5, 5), np.uint8)
            threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
            threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
            threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            filtered_contours = self.filter_contours(contours, min_area=200)

            print(f"Количество отфильтрованных контуров: {len(filtered_contours)}")

            sorted_contours = sorted(filtered_contours, key=cv2.contourArea, reverse=True)

            red_contour = None
            blue_contour = None

            if len(sorted_contours) > 0:
                red_contour = sorted_contours[0]

                if len(sorted_contours) > 1:
                    blue_contour = sorted_contours[1]

            result_text = ""

            if red_contour is not None:
                red_perimeter = cv2.arcLength(red_contour, True)
                cv2.drawContours(result_img, [red_contour], -1, (0, 0, 255), 2)
                cv2.drawContours(clean_result_img, [red_contour], -1, (0, 0, 255), 2)
                result_text += f"Длина основного красного контура: {red_perimeter:.2f}\n\n"
            else:
                result_text += "Красный контур не обнаружен\n\n"

            if blue_contour is not None:
                blue_perimeter = cv2.arcLength(blue_contour, True)
                cv2.drawContours(result_img, [blue_contour], -1, (255, 0, 0), 2)
                cv2.drawContours(clean_result_img, [blue_contour], -1, (255, 0, 0), 2)
                result_text += f"Длина основного синего контура: {blue_perimeter:.2f}\n\n"
            else:
                result_text += "Синий контур не обнаружен\n\n"

            # Определение состояния дробилки
            status = "Дробильная установка в норме"

            if red_contour is not None and blue_contour is not None:
                red_perimeter = cv2.arcLength(red_contour, True)
                blue_perimeter = cv2.arcLength(blue_contour, True)
                perimeter_difference = abs(red_perimeter - blue_perimeter)

                # Простое условие для определения состояния
                if perimeter_difference > 20:
                    status = "Дробильная установка нуждается в ремонте"
            else:
                status = "Невозможно определить состояние: недостаточно контуров"

            result_text += f"Результат анализа: {status}"

            self.display_image(result_img, self.processed_image_display)
            self.results_label.setText(result_text)

            self.status_bar.showMessage("Анализ контуров завершен")

            if self.image_path:
                try:
                    dir_name = os.path.dirname(self.image_path)
                    file_name = os.path.basename(self.image_path)
                    name, ext = os.path.splitext(file_name)

                    results_dir = os.path.join(dir_name, "processing_results_ru")
                    os.makedirs(results_dir, exist_ok=True)

                    result_image_path = os.path.join(results_dir, f"{name}_precise_contours{ext}")
                    print(f"Попытка сохранения изображения контуров в: {result_image_path}")
                    success1 = cv_imwrite(result_image_path, result_img)

                    clean_result_path = os.path.join(results_dir, f"{name}_precise_clean{ext}")
                    print(f"Попытка сохранения чистого изображения контуров в: {clean_result_path}")
                    success2 = cv_imwrite(clean_result_path, clean_result_img)

                    if not success1 or not success2:
                        raise Exception("Не удалось сохранить изображения контуров")

                    with open(os.path.join(results_dir, f"{name}_precise_analysis.txt"), 'w', encoding='utf-8') as f:
                        f.write(result_text)

                    self.status_bar.showMessage(f"Результаты анализа контуров сохранены: {result_image_path}")
                except Exception as e:
                    error_msg = f"Произошла ошибка при сохранении результатов анализа контуров: {str(e)}"
                    QMessageBox.warning(self, "Предупреждение", error_msg)
                    print(error_msg)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Произошла ошибка при анализе контуров: {str(e)}")
            print(f"Ошибка обработки изображения: {str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ContourDetectionApp()
    window.show()
    sys.exit(app.exec_())