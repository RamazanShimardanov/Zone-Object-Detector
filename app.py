import sys
import cv2
import numpy as np
import torch
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QComboBox, QPushButton, QCheckBox, QGroupBox, 
                             QLineEdit, QFileDialog, QTextEdit, QFrame, QSizePolicy)
from PyQt5.QtCore import Qt, QRect, QSize, QDateTime, QTimer, QPoint
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QPolygon

class VideoAnalyzerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Система видеонаналитики с YOLOv5")
        self.setMinimumSize(QSize(1000, 700))
        

        self.detection_polygon = []
        self.is_drawing = False
        self.current_video_source = None
        self.cap = None
        self.timer = QTimer()
        self.frame = None
        self.model = None
        self.target_classes = ['person', 'car', 'bus', 'truck']
        self.detected_objects_log = {}  # Словарь для хранения зафиксированных объектов
        
        self.init_ui()
        self.load_model()
        
    def load_model(self):
        try:
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            self.log_event("Модель YOLOv5 успешно загружена")
        except Exception as e:
            self.log_event(f"Ошибка загрузки модели: {str(e)}")
        
    def init_ui(self):
    
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        video_panel = QFrame()
        video_panel.setFrameShape(QFrame.StyledPanel)
        video_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        video_layout = QVBoxLayout()
        video_panel.setLayout(video_layout)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("background-color: black; color: white;")
        self.video_label.setText("Выберите источник видео")
        self.video_label.mousePressEvent = self.mouse_press_event
        video_layout.addWidget(self.video_label)
        
    
        video_control_layout = QHBoxLayout()
        
        self.video_source_combo = QComboBox()
        self.video_source_combo.addItems(["Выберите источник", "Видеофайл", "Веб-камера", "RTSP поток"])
        video_control_layout.addWidget(self.video_source_combo)
        
        self.file_path_edit = QLineEdit()
        self.file_path_edit.setPlaceholderText("Путь к видеофайлу")
        self.file_path_edit.setVisible(False)
        video_control_layout.addWidget(self.file_path_edit)
        
        self.browse_btn = QPushButton("Обзор...")
        self.browse_btn.setVisible(False)
        video_control_layout.addWidget(self.browse_btn)
        
        self.rtsp_url_edit = QLineEdit()
        self.rtsp_url_edit.setPlaceholderText("rtsp://адрес:порт/путь")
        self.rtsp_url_edit.setVisible(False)
        video_control_layout.addWidget(self.rtsp_url_edit)
        
        video_layout.addLayout(video_control_layout)
        
      
        control_btn_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("Запустить")
        self.stop_btn = QPushButton("Остановить")
        self.stop_btn.setEnabled(False)
        self.edit_polygon_btn = QPushButton("Рисовать полигон")
        self.reset_polygon_btn = QPushButton("Сбросить полигон")
        
        control_btn_layout.addWidget(self.start_btn)
        control_btn_layout.addWidget(self.stop_btn)
        control_btn_layout.addWidget(self.edit_polygon_btn)
        control_btn_layout.addWidget(self.reset_polygon_btn)
        
        video_layout.addLayout(control_btn_layout)
        
     
        settings_panel = QFrame()
        settings_panel.setFrameShape(QFrame.StyledPanel)
        settings_panel.setFixedWidth(300)
        settings_layout = QVBoxLayout()
        settings_panel.setLayout(settings_layout)
        
      
        detection_group = QGroupBox("Объекты для детекции")
        detection_layout = QVBoxLayout()
        
        self.detect_person = QCheckBox("Человек")
        self.detect_person.setChecked(True)
        self.detect_vehicle = QCheckBox("Транспорт")
        self.detect_vehicle.setChecked(True)
        self.detect_animal = QCheckBox("Животные")
        self.detect_package = QCheckBox("Грузы/пакеты")
        
        detection_layout.addWidget(self.detect_person)
        detection_layout.addWidget(self.detect_vehicle)
        detection_layout.addWidget(self.detect_animal)
        detection_layout.addWidget(self.detect_package)
        detection_group.setLayout(detection_layout)
        settings_layout.addWidget(detection_group)
        
        notify_group = QGroupBox("Уведомления")
        notify_layout = QVBoxLayout()
        
        self.enable_push = QCheckBox("Отправлять push-уведомления")
        self.push_endpoint_edit = QLineEdit()
        self.push_endpoint_edit.setPlaceholderText("URL endpoint для push")
        self.push_token_edit = QLineEdit()
        self.push_token_edit.setPlaceholderText("Токен авторизации")
        
        notify_layout.addWidget(self.enable_push)
        notify_layout.addWidget(self.push_endpoint_edit)
        notify_layout.addWidget(self.push_token_edit)
        notify_group.setLayout(notify_layout)
        settings_layout.addWidget(notify_group)
        
     
        self.detected_objects_group = QGroupBox("Зафиксированные объекты")
        detected_objects_layout = QVBoxLayout()
        self.detected_objects_log_text = QTextEdit()
        self.detected_objects_log_text.setReadOnly(True)
        detected_objects_layout.addWidget(self.detected_objects_log_text)
        self.detected_objects_group.setLayout(detected_objects_layout)
        settings_layout.addWidget(self.detected_objects_group)
        
       
        self.status_label = QLabel("Статус: система остановлена")
        self.status_label.setStyleSheet("font-weight: bold; color: #d35400;")
        settings_layout.addWidget(self.status_label)
        
     
        self.event_log = QTextEdit()
        self.event_log.setReadOnly(True)
        self.event_log.setPlaceholderText("Журнал событий...")
        settings_layout.addWidget(QLabel("Журнал событий:"))
        settings_layout.addWidget(self.event_log)
        
      
        main_layout.addWidget(video_panel, 3)
        main_layout.addWidget(settings_panel, 1)
        
      
        self.setup_connections()
        
    def setup_connections(self):
        self.video_source_combo.currentIndexChanged.connect(self.update_video_source_ui)
        self.browse_btn.clicked.connect(self.browse_video_file)
        self.start_btn.clicked.connect(self.start_video)
        self.stop_btn.clicked.connect(self.stop_video)
        self.edit_polygon_btn.clicked.connect(self.toggle_polygon_drawing)
        self.reset_polygon_btn.clicked.connect(self.reset_detection_polygon)
        self.enable_push.stateChanged.connect(self.toggle_push_settings)
        self.timer.timeout.connect(self.update_frame)
        
    def update_video_source_ui(self, index):
        self.file_path_edit.setVisible(index == 1)
        self.browse_btn.setVisible(index == 1)
        self.rtsp_url_edit.setVisible(index == 3)
        
    def browse_video_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите видеофайл", "", "Video Files (*.mp4 *.avi *.mov *.mkv)")
        if file_path:
            self.file_path_edit.setText(file_path)
            
    def toggle_push_settings(self, state):
        self.push_endpoint_edit.setEnabled(state)
        self.push_token_edit.setEnabled(state)
        
    def start_video(self):
        source_type = self.video_source_combo.currentIndex()
        
        if source_type == 0:
            self.log_event("Ошибка: не выбран источник видео")
            return
            
        self.current_video_source = source_type
        
        try:
            if source_type == 1:  # Видеофайл
                video_path = self.file_path_edit.text()
                if not video_path:
                    self.log_event("Ошибка: не указан путь к видеофайлу")
                    return
                self.cap = cv2.VideoCapture(video_path)
                self.log_event(f"Запуск воспроизведения файла: {video_path}")
                
            elif source_type == 2:  # Веб камера
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    self.cap = cv2.VideoCapture(1) 
                self.log_event("Запуск веб-камеры")
                
            elif source_type == 3:  # RTSP
                rtsp_url = self.rtsp_url_edit.text()
                if not rtsp_url:
                    self.log_event("Ошибка: не указан RTSP URL")
                    return
                self.cap = cv2.VideoCapture(rtsp_url)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  
                self.log_event(f"Подключение к RTSP потоку: {rtsp_url}")
            
            if not self.cap.isOpened():
                self.log_event("Ошибка: не удалось открыть источник видео")
                return
                
       
            ret, self.frame = self.cap.read()
            if not ret:
                self.log_event("Ошибка: не удалось прочитать первый кадр")
                self.cap.release()
                return
                
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.status_label.setText("Статус: система активна")
            self.status_label.setStyleSheet("font-weight: bold; color: #27ae60;")
            
            # 30 FPS
            self.timer.start(33)
            
        except Exception as e:
            self.log_event(f"Ошибка: {str(e)}")
            if self.cap and self.cap.isOpened():
                self.cap.release()
    
    def stop_video(self):
        self.timer.stop()
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.cap = None
        
        self.video_label.clear()
        self.video_label.setText("Видео остановлено")
        self.video_label.setStyleSheet("background-color: black; color: white;")
        
        self.log_event("Остановка системы")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("Статус: система остановлена")
        self.status_label.setStyleSheet("font-weight: bold; color: #d35400;")
        
    def update_frame(self):
        try:
            ret, self.frame = self.cap.read()
            if not ret:
                self.log_event("Ошибка чтения кадра")
                self.stop_video()
                return
            
       
            display_frame = self.process_frame(self.frame)
            
    
            rgb_image = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
          
            self.video_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
                self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            
        except Exception as e:
            self.log_event(f"Ошибка обновления кадра: {str(e)}")
            self.stop_video()
    
    def process_frame(self, frame):
      
        processed_frame = frame.copy()

        if self.model is not None:
            results = self.model(frame)
            
            for index, row in results.pandas().xyxy[0].iterrows():
                if row['name'] in self.target_classes:
                    name = str(row['name'])
                    x1 = int(row['xmin'])
                    y1 = int(row['ymin'])
                    x2 = int(row['xmax'])
                    y2 = int(row['ymax'])
                    confidence = float(row['confidence'])
                    
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    
             
                    color = (255, 255, 0)  
                    thickness = 2
                    
                
                    if len(self.detection_polygon) >= 3:
                        polygon = np.array([[p.x(), p.y()] for p in self.detection_polygon], dtype=np.int32)
                        dist = cv2.pointPolygonTest(polygon, (center_x, center_y), False)
                        
                        if dist >= 0:  
                            color = (0, 0, 255) 
                            thickness = 3
                            cv2.putText(processed_frame, ".", (x1, y1-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                   
                            obj_id = f"{name}_{center_x}_{center_y}"
                            if obj_id not in self.detected_objects_log:
                                current_time = QDateTime.currentDateTime().toString("hh:mm:ss")
                                log_entry = f"[{current_time}] Обнаружен: {name} ({confidence:.2f}) в точке ({center_x}, {center_y})"
                                self.detected_objects_log[obj_id] = log_entry
                                self.detected_objects_log_text.append(log_entry)
                    
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, thickness)
                    cv2.putText(processed_frame, name, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    cv2.circle(processed_frame, (center_x, center_y), 5, (0, 0, 255), -1)
        
    
        if len(self.detection_polygon) >= 3:
            polygon = np.array([[p.x(), p.y()] for p in self.detection_polygon], dtype=np.int32)
            overlay = processed_frame.copy()
            cv2.fillPoly(overlay, [polygon], (0, 255, 0))
            processed_frame = cv2.addWeighted(overlay, 0.2, processed_frame, 0.8, 0)
            cv2.polylines(processed_frame, [polygon], True, (0, 255, 0), 2)
        
        return processed_frame
        
    def toggle_polygon_drawing(self):
        self.is_drawing = not self.is_drawing
        if self.is_drawing:
            self.edit_polygon_btn.setStyleSheet("background-color: #f39c12;")
            self.log_event("Режим рисования полигона: включен")
        else:
            self.edit_polygon_btn.setStyleSheet("")
            self.log_event("Режим рисования полигона: выключен")
        
    def reset_detection_polygon(self):
        self.detection_polygon = []
        self.log_event("Полигон детекции сброшен")
        if self.frame is not None:
            self.update_frame()
        
    def log_event(self, message):
        self.event_log.append(f"[{QDateTime.currentDateTime().toString('hh:mm:ss')}] {message}")
        
    def mouse_press_event(self, event):
        if self.is_drawing and self.frame is not None:
       
            pos = event.pos()
            x = pos.x() - (self.video_label.width() - self.video_label.pixmap().width()) // 2
            y = pos.y() - (self.video_label.height() - self.video_label.pixmap().height()) // 2

            pixmap = self.video_label.pixmap()
            if pixmap:
                scale_x = self.frame.shape[1] / pixmap.width()
                scale_y = self.frame.shape[0] / pixmap.height()
                
                x_scaled = int(x * scale_x)
                y_scaled = int(y * scale_y)
                
                self.detection_polygon.append(QPoint(x_scaled, y_scaled))
                self.log_event(f"Добавлена точка полигона: ({x_scaled}, {y_scaled})")
                
      
                self.update_frame()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoAnalyzerApp()
    window.show()
    sys.exit(app.exec_())