import cv2
import numpy as np
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout,
                             QHBoxLayout, QFileDialog, QWidget, QSplitter, QFrame,
                             QStatusBar, QMessageBox)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import sys

# 添加处理包含非ASCII字符路径的函数
def cv_imread(file_path):
    """Функция imread для обработки путей с не-ASCII символами"""
    try:
        # 使用numpy从文件中读取二进制数据
        img_np = np.fromfile(file_path, dtype=np.uint8)
        # 使用imdecode解码图像数据
        return cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Ошибка чтения изображения: {file_path}, Ошибка: {str(e)}")
        return None

def cv_imwrite(file_path, img):
    """Функция imwrite для обработки путей с не-ASCII символами"""
    try:
        # 确保目录存在
        dir_name = os.path.dirname(file_path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)

        # 使用imencode编码图像数据
        ext = os.path.splitext(file_path)[1]
        result, encoded_img = cv2.imencode(ext, img)

        if result:
            # 将编码后的数据写入文件
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
        self.processed_results = {} # Используется для хранения результатов обработки, обеспечивая прослеживаемость

    def initUI(self):
        # Установка основных свойств окна
        self.setWindowTitle('Система обнаружения и анализа контуров')
        self.setGeometry(100, 100, 1200, 800)

        # Создание центрального виджета и компоновки
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Создание верхней области управления
        control_layout = QHBoxLayout()

        # Кнопка загрузки
        self.upload_btn = QPushButton('Загрузить изображение')
        self.upload_btn.clicked.connect(self.upload_image)
        self.upload_btn.setMinimumHeight(40)

        # Кнопка обработки
        self.process_btn = QPushButton('Обработать и сохранить контурами')
        self.process_btn.clicked.connect(self.process_and_save_contours)
        self.process_btn.setMinimumHeight(40)
        self.process_btn.setEnabled(False)

        # Кнопка точного извлечения контуров
        self.precise_btn = QPushButton('Точное извлечение контуров')
        self.precise_btn.clicked.connect(self.extract_precise_contours)
        self.precise_btn.setMinimumHeight(40)
        self.precise_btn.setEnabled(False)

        # Кнопка поиска синих контуров
        self.find_contours_btn = QPushButton('Найти синие контуров')
        self.find_contours_btn.clicked.connect(self.find_blue_contours)
        self.find_contours_btn.setMinimumHeight(40)
        self.find_contours_btn.setEnabled(False)

        # Добавление кнопок в компоновку управления
        control_layout.addWidget(self.upload_btn)
        control_layout.addWidget(self.process_btn)
        control_layout.addWidget(self.precise_btn)
        control_layout.addWidget(self.find_contours_btn)

        # Добавление компоновки управления в основную компоновку
        main_layout.addLayout(control_layout)

        # Создание разделителя для области отображения изображений
        splitter = QSplitter(Qt.Horizontal)

        # Левая область для отображения исходного изображения
        left_frame = QFrame()
        left_frame.setFrameShape(QFrame.StyledPanel)
        left_layout = QVBoxLayout(left_frame)
        self.original_image_label = QLabel('Исходное изображение')
        self.original_image_label.setAlignment(Qt.AlignCenter)
        self.original_image_display = QLabel()
        self.original_image_display.setAlignment(Qt.AlignCenter)
        self.original_image_display.setMinimumSize(500, 500)
        self.original_image_display.setStyleSheet("border: 1px solid #CCCCCC; background-color: #F5F5F5;")
        left_layout.addWidget(self.original_image_label)
        left_layout.addWidget(self.original_image_display)

        # Правая область для отображения обработанного изображения
        right_frame = QFrame()
        right_frame.setFrameShape(QFrame.StyledPanel)
        right_layout = QVBoxLayout(right_frame)
        self.processed_image_label = QLabel('Обработанное изображение')
        self.processed_image_label.setAlignment(Qt.AlignCenter)
        self.processed_image_display = QLabel()
        self.processed_image_display.setAlignment(Qt.AlignCenter)
        self.processed_image_display.setMinimumSize(500, 500)
        self.processed_image_display.setStyleSheet("border: 1px solid #CCCCCC; background-color: #F5F5F5;")
        right_layout.addWidget(self.processed_image_label)
        right_layout.addWidget(self.processed_image_display)

        # Добавление левой и правой рамки в разделитель
        splitter.addWidget(left_frame)
        splitter.addWidget(right_frame)

        # Добавление разделителя в основную компоновку
        main_layout.addWidget(splitter)

        # Создание строки состояния для отображения результатов
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage('Пожалуйста, загрузите изображение для начала анализа')

        # Создание области отображения результатов
        results_layout = QVBoxLayout()
        self.results_label = QLabel('Результаты анализа будут отображены здесь')
        self.results_label.setAlignment(Qt.AlignCenter)
        self.results_label.setStyleSheet("background-color: #EFEFEF; padding: 10px; border-radius: 5px;")
        results_layout.addWidget(self.results_label)

        # Добавление компоновки результатов в основную компоновку
        main_layout.addLayout(results_layout)

    def upload_image(self):
        """Открыть диалоговое окно для выбора изображения"""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выбрать изображение", "", "Файлы изображений (*.png *.jpg *.jpeg *.bmp);;Все файлы (*)", options=options
        )

        if file_path:
            try:
                # Печать пути к файлу для отладки
                print(f"Попытка загрузки изображения: {file_path}")
                self.status_bar.showMessage(f'Загрузка изображения: {file_path}')

                # Проверка существования файла
                if not os.path.exists(file_path):
                    QMessageBox.critical(self, "Ошибка", f"Файл не существует: {file_path}")
                    return

                # Убедиться, что используется абсолютный путь
                file_path = os.path.abspath(file_path)
                self.image_path = file_path

                # Попытка загрузки изображения
                self.original_image = cv_imread(file_path)

                if self.original_image is None:
                    QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить изображение: {file_path}\nУбедитесь, что формат файла правильный и он не поврежден")
                    return

                # Обновление строки состояния
                self.status_bar.showMessage(f'Изображение загружено: {file_path}')

                # Отображение исходного изображения слева
                self.display_image(self.original_image, self.original_image_display)

                # Активация кнопки обработки
                self.process_btn.setEnabled(True)

                # Очистка пути к предыдущему изображению контуров
                self.contour_image_path = None

                # Деактивация кнопок поиска синих контуров и точного извлечения контуров
                self.find_contours_btn.setEnabled(False)
                self.precise_btn.setEnabled(False)
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Произошла ошибка при загрузке изображения: {str(e)}")
                print(f"Ошибка загрузки изображения: {str(e)}")

    def filter_contours(self, contours, min_area=200):
        """Оптимизированная функция фильтрации контуров, специально для фильтрации форм вопросительного знака"""
        # Фильтрация слишком маленьких контуров
        filtered_by_area = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

        # Фильтрация контуров в форме вопросительного знака (вопросительные знаки обычно имеют высокую сложность и малую площадь)
        filtered_contours = []
        for cnt in filtered_by_area:
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)

            # Если площадь равна 0, пропустить этот контур
            if area == 0:
                continue

            # Вычисление сложности контура (форм-фактор)
            complexity = (perimeter ** 2) / (4 * np.pi * area) if area != 0 else 0

            # Вопросительные знаки обычно имеют более сложную форму (более высокий форм-фактор) и меньшую площадь
            if complexity > 4.0:  # Повысить порог сложности
                print(f"Отфильтрован контур высокой сложности - Площадь: {area:.2f}, Сложность: {complexity:.2f}")
                continue

            # Использование аппроксимации многоугольником для обнаружения особенностей вопросительного знака
            epsilon = 0.02 * perimeter
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            # Вопросительные знаки обычно имеют больше вершин
            if len(approx) > 12:  # Установлено на основе эмпирических данных
                print(f"Отфильтрован контур со слишком большим количеством вершин - Площадь: {area:.2f}, Кол-во вершин: {len(approx)}")
                continue

            # Добавление прошедших проверку контуров в результат
            filtered_contours.append(cnt)

        return filtered_contours

    def process_and_save_contours(self):
        """Обработать изображение и сохранить изображение с красными контурами"""
        if self.original_image is None:
            QMessageBox.warning(self, "Предупреждение", "Пожалуйста, сначала загрузите изображение")
            return

        try:
            # Копирование исходного изображения для обработки
            img = self.original_image.copy()

            # Преобразование в оттенки серого
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

            # Использование улучшенных морфологических операций для очистки изображения
            kernel = np.ones((5, 5), np.uint8)  # Увеличить размер ядра
            threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
            threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
            # Дополнительные морфологические операции для удаления форм вопросительного знака
            threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)

            # Поиск начальных контуров
            contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Применение оптимизированной фильтрации контуров
            filtered_contours = self.filter_contours(contours, min_area=200) # Повысить порог площади

            # Рисование красных контуров на изображении
            red_contour_img = self.original_image.copy()
            cv2.drawContours(red_contour_img, filtered_contours, -1, (0, 0, 255), 2) # Красные контуры

            # Сохранение изображения с красными контурами
            if self.image_path:
                try:
                    # Получение каталога и имени файла исходного изображения
                    dir_name = os.path.dirname(self.image_path)
                    file_name = os.path.basename(self.image_path)
                    name, ext = os.path.splitext(file_name)

                    # Создание пути к файлу изображения с контурами
                    self.contour_image_path = os.path.join(dir_name, f"{name}_red_contour{ext}")

                    print(f"Попытка сохранения изображения контуров в: {self.contour_image_path}")

                    # Сохранение изображения с контурами
                    success = cv_imwrite(self.contour_image_path, red_contour_img)

                    if not success:
                        raise Exception("Не удалось сохранить изображение")

                    # Проверка успешного сохранения файла
                    if not os.path.exists(self.contour_image_path):
                        raise Exception("Изображение не найдено после сохранения")

                    # Отображение сообщения об успешном сохранении
                    self.status_bar.showMessage(f'Изображение с красными контурами сохранено: {self.contour_image_path}')

                    # Отображение изображения с красными контурами
                    self.display_image(red_contour_img, self.processed_image_display)

                    # Активация кнопок поиска синих контуров и точного извлечения контуров
                    self.find_contours_btn.setEnabled(True)
                    self.precise_btn.setEnabled(True)

                    # Сохранение результатов обработки для прослеживаемости
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
        """Поиск синих контуров на изображении с красными контурами"""
        if not self.contour_image_path or not os.path.exists(self.contour_image_path):
            QMessageBox.warning(self, "Предупреждение",
                                f"Не удалось найти изображение контуров: {self.contour_image_path}\nПожалуйста, сначала создайте и сохраните изображение с красными контурами")
            return

        try:
            # Чтение изображения с красными контурами
            print(f"Попытка загрузки изображения контуров: {self.contour_image_path}")
            contour_img = cv_imread(self.contour_image_path)

            if contour_img is None:
                QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить изображение контуров: {self.contour_image_path}")
                return

            # Копирование изображения для отображения результатов
            result_img = contour_img.copy()

            # Преобразование в цветовое пространство HSV для более легкого разделения цветов
            hsv = cv2.cvtColor(contour_img, cv2.COLOR_BGR2HSV)

            # Более точный диапазон HSV для красного цвета
            lower_red1 = np.array([0, 120, 120])  # Повысить нижние пределы насыщенности и яркости
            upper_red1 = np.array([8, 255, 255])  # Сузить диапазон оттенков
            red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)

            lower_red2 = np.array([165, 120, 120]) # Повысить нижние пределы насыщенности и яркости
            upper_red2 = np.array([180, 255, 255])
            red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

            # Объединение двух красных масок
            red_mask = red_mask1 + red_mask2

            # Более точный диапазон HSV для синего цвета
            lower_blue = np.array([105, 80, 80]) # Повысить нижние пределы
            upper_blue = np.array([125, 255, 255]) # Сузить диапазон
            blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

            # Использование улучшенных морфологических операций для очистки масок
            kernel = np.ones((5, 5), np.uint8) # Использовать большее ядро
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
            blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
            blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)

            # Поиск контуров красных областей
            red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            print(f"Найдено красных контуров: {len(red_contours)}")

            # Поиск контуров синих областей
            blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            print(f"Найдено синих контуров: {len(blue_contours)}")

            # Применение оптимизированной фильтрации контуров
            filtered_red_contours = self.filter_contours(red_contours, min_area=200)
            filtered_blue_contours = self.filter_contours(blue_contours, min_area=150)

            print(f"Количество красных контуров после фильтрации: {len(filtered_red_contours)}")
            print(f"Количество синих контуров после фильтрации: {len(filtered_blue_contours)}")

            # Рисование синих контуров на изображении (стандартным синим цветом)
            cv2.drawContours(result_img, filtered_blue_contours, -1, (255, 0, 0), 2)

            # Найти все контуры (включая красные и синие)
            gray = cv2.cvtColor(contour_img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 200)

            # Использование морфологических операций для очистки изображения краев
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

            all_contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # Применение оптимизированной фильтрации контуров
            all_contours = self.filter_contours(all_contours, min_area=200)
            print(f"Найдено всех контуров (после фильтрации): {len(all_contours)}")

            # Сортировка контуров по площади
            sorted_contours = sorted(all_contours, key=cv2.contourArea, reverse=True)

            # Убедиться, что контуров достаточно для анализа
            result_text = ""
            if len(sorted_contours) >= 2:
                largest_contour = sorted_contours[0]  # Самый большой контур
                second_largest_contour = sorted_contours[1]  # Второй по величине контур

                # Вычисление периметра контуров
                largest_perimeter = cv2.arcLength(largest_contour, True)
                second_largest_perimeter = cv2.arcLength(second_largest_contour, True)

                # Вычисление разницы периметров
                perimeter_difference = abs(largest_perimeter - second_largest_perimeter)

                # Отметка двух самых больших контуров на изображении (разными цветами)
                cv2.drawContours(result_img, [largest_contour], -1, (255, 0, 0), 2)  # Синий
                cv2.drawContours(result_img, [second_largest_contour], -1, (0, 255, 255), 2)  # Желтый

                # Оценка состояния на основе разницы периметров и количества синих контуров
                if perimeter_difference > 20:
                    status = "Обнаружено аномальное отклонение"
                    status_color = (0, 0, 255) # Красный для аномалии
                else:
                    status = "Нормальное состояние"
                    status_color = (0, 255, 0) # Зеленый для нормального состояния

                blue_count = len(filtered_blue_contours)

                # Убедиться, что текст состояния содержит только ASCII символы (для cv2.putText)
                # В данном случае переведенный текст уже в кириллице, которая не ASCII.
                # cv2.putText может не поддерживать кириллицу без специальных шрифтов.
                # Для простоты, будем использовать английские эквиваленты для отображения на изображении.
                status_text_display = "Status: Anomaly" if status == "Обнаружено аномальное отклонение" else "Status: Normal"
                # status_text = "".join(char for char in status if ord(char) < 128) # Это удалит кириллицу

                result_text = f"Состояние: {status}\n"
                result_text += f"Количество синих контуров: {blue_count}\n"
                result_text += f"Периметр наибольшего контура: {largest_perimeter:.2f}\n"
                result_text += f"Периметр второго по величине контура: {second_largest_perimeter:.2f}\n"
                result_text += f"Разница периметров: {perimeter_difference:.2f}"

                # ############################################################
                # ## УДАЛИТЬ ИЛИ ЗАКОММЕНТИРОВАТЬ СЛЕДУЮЩУЮ СТРОКУ, ЧОБЫ УБРАТЬ ТЕКСТ СТАТУСА ##
                # ############################################################
                # # Добавление текстовой метки на изображение (используя английский текст для совместимости)
                # cv2.putText(result_img, status_text_display, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
                # ############################################################

            else:
                result_text = "Не найдено достаточного количества контуров для анализа"

            # Отображение результатов
            self.display_image(result_img, self.processed_image_display)
            self.results_label.setText(result_text)

            # Обновление строки состояния
            self.status_bar.showMessage("Анализ синих контуров завершен")

            # Сохранение результирующего изображения
            if self.image_path:
                try:
                    dir_name = os.path.dirname(self.image_path)
                    file_name = os.path.basename(self.image_path)
                    name, ext = os.path.splitext(file_name)

                    result_image_path = os.path.join(dir_name, f"{name}_analysis_result{ext}")
                    print(f"Попытка сохранения результирующего изображения анализа в: {result_image_path}")

                    # Сохранение результирующего изображения
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
        """Анализ характеристик контуров"""
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

        # Извлечение характеристик контуров
        areas = [cv2.contourArea(cnt) for cnt in contours]
        perimeters = [cv2.arcLength(cnt, True) for cnt in contours]

        # Вычисление сложности контура (форм-фактор)
        complexities = []
        for i in range(len(contours)):
            if perimeters[i] > 0:
                # Форм-фактор круга равен 1, чем более неправильная форма, тем он больше
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
        """Оценка состояния дробилки на основе характеристик многоцветных контуров"""
        status = "Нормальное"
        status_details = {}

        # Оценка на основе характеристик красных контуров
        red_features = contour_features.get("red", {"count": 0}) # Используем .get для избежания KeyError
        if red_features["count"] > 0:
            # Проверка сложности красных контуров
            if red_features.get("complexity", 0) > 2.0: # Порог можно настроить
                status = "Аномальное"
                status_details["Сложность красного контура"] = f"Высокая ({red_features['complexity']:.2f} > 2.0)"
            else:
                status_details["Сложность красного контура"] = f"Нормальная ({red_features.get('complexity', 0):.2f})"

            # Проверка коэффициента вариации площади красных контуров
            cv_red = red_features.get("area_std", 0) / red_features.get("avg_area", 1) if red_features.get("avg_area", 0) > 0 else 0 # Избегаем деления на ноль
            if cv_red > 0.5: # Порог можно настроить
                status = "Аномальное"
                status_details["Вариация площади красного контура"] = f"Большая (CV = {cv_red:.2f} > 0.5)"
            else:
                status_details["Вариация площади красного контура"] = f"Нормальная (CV = {cv_red:.2f})"

        # Оценка на основе характеристик синих контуров
        blue_features = contour_features.get("blue", {"count": 0})
        if blue_features["count"] > 0:
            # Проверка соотношения синих контуров
            blue_red_ratio = blue_features["count"] / red_features["count"] if red_features.get("count", 0) > 0 else 0
            if blue_red_ratio > 3.0: # Порог можно настроить
                status = "Аномальное"
                status_details["Соотношение синих/красных контуров"] = f"Аномальное ({blue_red_ratio:.2f} > 3.0)"
            else:
                status_details["Соотношение синих/красных контуров"] = f"Нормальное ({blue_red_ratio:.2f})"

            # Проверка средней площади синих контуров
            if blue_features.get("avg_area", 0) < 100: # Порог можно настроить
                status_details["Средняя площадь синего контура"] = f"Малая ({blue_features['avg_area']:.2f} < 100)"
            else:
                status_details["Средняя площадь синего контура"] = f"Нормальная ({blue_features.get('avg_area', 0):.2f})"

        # Оценка на основе характеристик зеленых контуров
        green_features = contour_features.get("green", {"count": 0})
        if green_features["count"] > 0:
            # Проверка характеристик зеленых контуров
            if green_features.get("avg_area", 0) > 1000: # Порог можно настроить
                status_details["Средняя площадь зеленого контура"] = f"Большая ({green_features['avg_area']:.2f} > 1000)"
            else:
                status_details["Средняя площадь зеленого контура"] = f"Нормальная ({green_features.get('avg_area', 0):.2f})"

        # Проверка баланса пропорций цветов
        total_contours = sum(f.get("count", 0) for f in contour_features.values()) # Используем .get
        if total_contours > 0:
            for color, features in contour_features.items():
                if features.get("count", 0) > 0: # Используем .get
                    color_ratio = features["count"] / total_contours
                    # Формируем ключ правильно
                    key_name = ""
                    if color == "red": key_name = "Красных"
                    elif color == "blue": key_name = "Синих"
                    elif color == "green": key_name = "Зеленых"
                    else: key_name = color.capitalize() # На случай других цветов
                    status_details[f"Пропорция {key_name} контуров"] = f"{color_ratio:.2%}"


        return status, status_details

    def display_image(self, cv_img, label):
        """Отображение изображения OpenCV на метке Qt"""
        if cv_img is None:
            label.clear() # Очистить метку, если изображение None
            label.setText("Нет изображения") # Показать текст-заполнитель
            return

        try:
            # Изменение размера изображения для соответствия метке
            h, w = cv_img.shape[:2]
            label_w = label.width()
            label_h = label.height()

            # Предотвращение деления на ноль и обработка слишком маленьких размеров метки
            if h == 0 or label_w <= 0 or label_h <= 0:
                 label.clear()
                 label.setText("Некорректный размер")
                 return

            # Масштабирование изображения с сохранением пропорций
            aspect_ratio = w / h
            if label_w / label_h > aspect_ratio:
                new_h = label_h
                new_w = int(new_h * aspect_ratio)
            else:
                new_w = label_w
                new_h = int(new_w / aspect_ratio)

             # Убедиться, что новые размеры положительны
            if new_w <= 0 or new_h <=0:
                label.clear()
                label.setText("Ошибка масштабирования")
                return

            # Изменение размера изображения
            resized_img = cv2.resize(cv_img, (new_w, new_h), interpolation=cv2.INTER_AREA) # Добавлена интерполяция

            # Преобразование в формат Qt
            qt_image = None
            if len(resized_img.shape) == 3: # Цветное BGR
                rgb_image = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
                h_r, w_r, ch = rgb_image.shape
                bytes_per_line = ch * w_r
                qt_image = QImage(rgb_image.data, w_r, h_r, bytes_per_line, QImage.Format_RGB888)
            elif len(resized_img.shape) == 2: # Ч/Б изображение
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

            # Отображение изображения на метке
            label.setPixmap(pixmap)
        except Exception as e:
            print(f"Ошибка в display_image: {e}")
            label.clear()
            label.setText("Ошибка отображения")


    def extract_precise_contours(self):
        """Точное извлечение основных контуров, полное удаление символов вопросительного знака и нецелевых контуров"""
        if not self.image_path:
            QMessageBox.warning(self, "Предупреждение", "Пожалуйста, сначала загрузите изображение")
            return

        try:
            # Работать непосредственно с исходным изображением, а не с изображением с контурами
            original_img = cv_imread(self.image_path)
            if original_img is None:
                QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить исходное изображение: {self.image_path}")
                return

            # Создание изображения для результатов
            result_img = original_img.copy()
            height, width = original_img.shape[:2]

            # Создание чисто черного фонового изображения
            clean_result_img = np.zeros((height, width, 3), dtype=np.uint8)

            # Использование K-means для сегментации изображения
            print("Применение K-means для сегментации изображения...")
            # Преобразование изображения в одномерный массив
            Z = original_img.reshape((-1, 3))
            Z = np.float32(Z)

            # Определение параметров K-means
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            K = 8 # Сегментировать на 8 областей

            # Применение K-means
            _, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

            # Преобразование результата обратно в uint8
            centers = np.uint8(centers)
            segmented_img = centers[labels.flatten()].reshape(original_img.shape)

            # Преобразование в оттенки серого для обнаружения контуров
            gray = cv2.cvtColor(segmented_img, cv2.COLOR_BGR2GRAY)
            _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

            # Использование улучшенных морфологических операций для очистки изображения
            kernel = np.ones((5, 5), np.uint8) # Увеличить размер ядра
            threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
            threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
            # Дополнительные морфологические операции для удаления форм вопросительного знака
            threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)

            # Поиск контуров
            contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Использование оптимизированной функции фильтрации контуров
            filtered_contours = self.filter_contours(contours, min_area=200)

            print(f"Количество отфильтрованных контуров: {len(filtered_contours)}")

            # Сортировка контуров по площади
            sorted_contours = sorted(filtered_contours, key=cv2.contourArea, reverse=True)

            # Поиск самого большого контура (предположительно красный контур)
            red_contour = None
            blue_contour = None

            if len(sorted_contours) > 0:
                red_contour = sorted_contours[0]

                # Попытка найти синий контур среди оставшихся
                # Здесь используется эвристический метод: выбор второго по величине контура как синего
                if len(sorted_contours) > 1:
                    blue_contour = sorted_contours[1]

                    # Проверка наложения контуров, если есть, возможно, неверная сегментация
                    red_mask = np.zeros((height, width), dtype=np.uint8)
                    blue_mask = np.zeros((height, width), dtype=np.uint8)

                    cv2.drawContours(red_mask, [red_contour], -1, 255, -1)
                    cv2.drawContours(blue_mask, [blue_contour], -1, 255, -1)

                    overlap = cv2.bitwise_and(red_mask, blue_mask)
                    overlap_area = np.count_nonzero(overlap)

                    if overlap_area > 0:
                        print(f"Предупреждение: обнаружено наложение контуров, площадь {overlap_area}")
                        # При необходимости можно обработать проблему наложения

            # Создание текста результата
            result_text = ""

            # Сохранение выбранных контуров
            selected_contours = []

            if red_contour is not None:
                red_area = cv2.contourArea(red_contour)
                red_perimeter = cv2.arcLength(red_contour, True)

                selected_contours.append({"contour": red_contour, "color": (0, 0, 255), "name": "Основной красный контур"})
                result_text += f"Основной красный контур:\nПлощадь: {red_area:.2f}\nПериметр: {red_perimeter:.2f}\n\n"

                # Рисование красного контура на результирующем изображении
                cv2.drawContours(result_img, [red_contour], -1, (0, 0, 255), 2)
                cv2.drawContours(clean_result_img, [red_contour], -1, (0, 0, 255), 2)
            else:
                result_text += "Действительный красный контур не обнаружен\n\n"

            if blue_contour is not None:
                blue_area = cv2.contourArea(blue_contour)
                blue_perimeter = cv2.arcLength(blue_contour, True)

                selected_contours.append({"contour": blue_contour, "color": (255, 0, 0), "name": "Основной синий контур"})
                result_text += f"Основной синий контур:\nПлощадь: {blue_area:.2f}\nПериметр: {blue_perimeter:.2f}\n\n"

                # Рисование синего контура на результирующем изображении
                cv2.drawContours(result_img, [blue_contour], -1, (255, 0, 0), 2)
                cv2.drawContours(clean_result_img, [blue_contour], -1, (255, 0, 0), 2)
            else:
                result_text += "Действительный синий контур не обнаружен\n\n"

            # ############################################################
            # ## УДАЛИТЬ ИЛИ ЗАКОММЕНТИРОВАТЬ СЛЕДУЮЩИЙ ЦИКЛ, ЧТОБЫ УБРАТЬ ТЕКСТ МЕТОК ##
            # ############################################################
            # # Добавление меток к контурам
            # for item in selected_contours:
            #     # Вычисление центральной точки контура
            #     M = cv2.moments(item["contour"])
            #     if M["m00"] != 0:
            #         cx = int(M["m10"] / M["m00"])
            #         cy = int(M["m01"] / M["m00"])
            #         # Добавление метки (использование английского текста для совместимости с cv2.putText)
            #         label_text_display = "Main Red" if "красный" in item["name"] else "Main Blue"
            #         # label_text = "".join(char for char in item["name"] if ord(char) < 128) # Удалит кириллицу
            #         cv2.putText(result_img, label_text_display, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, item["color"], 2)
            # ############################################################


            # Оценка состояния дробилки
            status = "Нормальное"
            status_details = {}

            if len(selected_contours) >= 2:
                # Найти красный и синий контуры
                red_idx = [i for i, item in enumerate(selected_contours) if "красный" in item["name"]]
                blue_idx = [i for i, item in enumerate(selected_contours) if "синий" in item["name"]]

                if red_idx and blue_idx:
                    red_contour_obj = selected_contours[red_idx[0]]["contour"] # Переименовать, чтобы избежать конфликта с переменной red_contour
                    blue_contour_obj = selected_contours[blue_idx[0]]["contour"] # Переименовать

                    red_area = cv2.contourArea(red_contour_obj)
                    blue_area = cv2.contourArea(blue_contour_obj)

                    red_perimeter = cv2.arcLength(red_contour_obj, True)
                    blue_perimeter = cv2.arcLength(blue_contour_obj, True)

                    # Вычисление характеристик формы
                    red_complexity = (red_perimeter ** 2) / (4 * np.pi * red_area) if red_area > 0 else 0
                    blue_complexity = (blue_perimeter ** 2) / (4 * np.pi * blue_area) if blue_area > 0 else 0
                    area_ratio = blue_area / red_area if red_area > 0 else 0

                    # Оценка состояния
                    if red_complexity > 2.0:
                        status = "Аномальное"
                        status_details["Сложность красного контура"] = f"Высокая ({red_complexity:.2f} > 2.0)"
                    else:
                        status_details["Сложность красного контура"] = f"Нормальная ({red_complexity:.2f})"

                    if area_ratio > 1.5 or area_ratio < 0.1:
                        status = "Аномальное"
                        status_details["Соотношение площадей синий/красный"] = f"Аномальное ({area_ratio:.2f})"
                    else:
                        status_details["Соотношение площадей синий/красный"] = f"Нормальное ({area_ratio:.2f})"

                    result_text += f"Оценка формы:\n"
                    result_text += f"Сложность красного контура: {red_complexity:.2f}\n"
                    result_text += f"Сложность синего контура: {blue_complexity:.2f}\n"
                    result_text += f"Соотношение площадей синий/красный: {area_ratio:.2f}\n\n"

            result_text += f"Состояние дробилки: {status}\n\n"
            if status_details:
                for key, value in status_details.items():
                    result_text += f"{key}: {value}\n"
            else:
                result_text += "Не удалось выполнить детальную оценку состояния\n"

            # ############################################################
            # ## УДАЛИТЬ ИЛИ ЗАКОММЕНТИРОВАТЬ СЛЕДУЮЩУЮ СТРОКУ, ЧТОБЫ УБРАТЬ ТЕКСТ СТАТУСА ##
            # ############################################################
            # # Отображение состояния на изображении (используя английский текст)
            # status_text_display = f"Status: {'Anomaly' if status == 'Аномальное' else 'Normal'}"
            # # status_text = "".join(char for char in f"Состояние: {status}" if ord(char) < 128)
            # cv2.putText(result_img, status_text_display, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
            #             (0, 0, 255) if status == "Аномальное" else (0, 255, 0), 2)
            # ############################################################


            # Отображение результатов
            self.display_image(result_img, self.processed_image_display)
            self.results_label.setText(result_text)

            # Обновление строки состояния
            self.status_bar.showMessage("Точное извлечение контуров завершено")

            # Сохранение результатов
            if self.image_path:
                try:
                    dir_name = os.path.dirname(self.image_path)
                    file_name = os.path.basename(self.image_path)
                    name, ext = os.path.splitext(file_name)

                    # Создание каталога для хранения результатов обработки
                    results_dir = os.path.join(dir_name, "processing_results_ru") # Добавлено _ru для различения
                    os.makedirs(results_dir, exist_ok=True)

                    # Сохранение обработанного изображения
                    result_image_path = os.path.join(results_dir, f"{name}_precise_contours{ext}")
                    print(f"Попытка сохранения изображения точных контуров в: {result_image_path}")
                    success1 = cv_imwrite(result_image_path, result_img)

                    # Сохранение чистого изображения контуров
                    clean_result_path = os.path.join(results_dir, f"{name}_precise_clean{ext}")
                    print(f"Попытка сохранения чистого изображения точных контуров в: {clean_result_path}")
                    success2 = cv_imwrite(clean_result_path, clean_result_img)

                    if not success1 or not success2:
                        raise Exception("Не удалось сохранить изображения точных контуров")

                    # Сохранение текстового анализа
                    with open(os.path.join(results_dir, f"{name}_precise_analysis.txt"), 'w', encoding='utf-8') as f:
                        f.write(result_text)

                    self.status_bar.showMessage(f"Результаты точного анализа контуров сохранены: {result_image_path}")
                except Exception as e:
                    error_msg = f"Произошла ошибка при сохранении результатов точного анализа контуров: {str(e)}"
                    QMessageBox.warning(self, "Предупреждение", error_msg)
                    print(error_msg)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Произошла ошибка при точном извлечении контуров: {str(e)}")
            print(f"Ошибка точного извлечения контуров: {str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ContourDetectionApp()
    window.show()
    sys.exit(app.exec_())