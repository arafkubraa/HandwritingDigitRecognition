
import sys
import numpy as np
import tensorflow as tf
import time
import ssl
import random
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                               QPushButton, QLabel, QLineEdit, QTextEdit, QGroupBox, QGridLayout, QMessageBox)
from PySide6.QtGui import QPainter, QPen, QImage, QPixmap, QColor, QFont, QPainterPath
from PySide6.QtCore import Qt, QPoint, QThread, Signal, QSize, QRect


import my_process_module


ssl._create_default_https_context = ssl._create_unverified_context


GUI_BACKGROUND_COLOR = "#faf8f9"
CONTROL_PANEL_BG_COLOR = "#faf8f9"
CONTROL_GROUP_BG_COLOR = "#ffffff"
LOG_BG_COLOR = "#ffffff"
LOG_TEXT_COLOR = "#212529"
PLOT_AREA_BG_COLOR = "#faf7f2"


PASTEL_COLORS_DICT = {
    1: QColor(244, 143, 177), 2: QColor(120, 200, 160), 3: QColor(135, 179, 255),
    4: QColor(255, 224, 130), 5: QColor(200, 170, 255), 6: QColor(77, 208, 225),
}
PASTEL_REGION_COLORS_DICT = {
    1: QColor(255, 220, 235, 255), 2: QColor(220, 245, 225, 255), 3: QColor(225, 235, 255, 255),
    4: QColor(255, 248, 225, 255), 5: QColor(240, 230, 255, 255), 6: QColor(204, 247, 255, 255),
}

class LossGraphWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(200)
        self.loss_history = []
        self.setStyleSheet(f"background-color: {PLOT_AREA_BG_COLOR}; border: 1px solid #ddd; border-radius: 5px;")

    def update_data(self, history):
        self.loss_history = history
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        painter.fillRect(self.rect(), QColor(PLOT_AREA_BG_COLOR))
        
        if not self.loss_history:
            painter.setPen(QPen(QColor("#888888"), 1))
            painter.drawText(self.rect(), Qt.AlignCenter, "No training data yet.")
            return

        w = self.width()
        h = self.height()
        padding = 20
        
       
        painter.setPen(QPen(QColor("#cccccc"), 1))
        painter.drawRect(padding, padding, w - 2*padding, h - 2*padding)
        
        max_loss = max(self.loss_history) if self.loss_history else 1.0
        min_loss = 0.0
        
        path = QPainterPath()
        
        points = []
        num_points = len(self.loss_history)
        
        if num_points < 2:
            return

        x_step = (w - 2*padding) / (num_points - 1)
        
        for i, loss in enumerate(self.loss_history):
            x = padding + i * x_step
            
            if max_loss == 0: val = 0
            else: val = loss / max_loss
            
            y = h - padding - (val * (h - 2*padding))
            points.append(QPoint(int(x), int(y)))
            
        if points:
            path.moveTo(points[0])
            for p in points[1:]:
                path.lineTo(p)
                
            
            pen = QPen(PASTEL_COLORS_DICT[1], 3)
            painter.setPen(pen)
            painter.drawPath(path)
            
            
            path.lineTo(points[-1].x(), h - padding)
            path.lineTo(points[0].x(), h - padding)
            path.closeSubpath()
            painter.fillPath(path, PASTEL_REGION_COLORS_DICT[1])

class TrainingWorker(QThread):
    progress_log = Signal(str)
    finished_training = Signal(list, list, list) # losses, x_test, y_test
    
    def __init__(self, epochs, learning_rate):
        super().__init__()
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.nn = None

    print("Worker running")
    def run(self):
        self.progress_log.emit("Loading MNIST data...")
        try:
            mnist = tf.keras.datasets.mnist
            (x_train, y_train), (x_test, y_test) = mnist.load_data()

            train_size = 2000
            
            x_train_sub = x_train[:train_size]
            y_train_sub = y_train[:train_size]
            
            self.progress_log.emit(f"Data loaded. Training on {train_size} samples.")

            
            x_train_flat = x_train_sub.reshape(x_train_sub.shape[0], -1).astype('float32') / 255.0
            y_train_onehot = np.eye(10)[y_train_sub]

            x_train_list = x_train_flat.tolist()
            y_train_list = y_train_onehot.tolist()

            self.progress_log.emit("Initializing Neural Network...")
            layer_sizes = [784, 128, 10]
            self.nn = my_process_module.NeuralNetwork(layer_sizes)

            #Autoencoder
            self.progress_log.emit(f"--- Phase 1: Autoencoder Training ({self.epochs} epochs) ---")
            start_time = time.time()
            
            ae_history = self.nn.train_autoencoder(x_train_list, self.epochs, self.learning_rate)
            
            ae_duration = time.time() - start_time
            self.progress_log.emit(f"AE Training finished in {ae_duration:.2f} seconds.")

            #Classifier
            self.progress_log.emit(f"--- Phase 2: Classifier Training ({self.epochs} epochs) ---")
            
            clf_history = self.nn.train_classifier(x_train_list, y_train_list, self.epochs, self.learning_rate)
            
            total_duration = time.time() - start_time
            self.progress_log.emit(f"Total Training finished in {total_duration:.2f} seconds.")
            
            self.finished_training.emit(clf_history, x_test.tolist(), y_test.tolist())

        except Exception as e:
            self.progress_log.emit(f"Error during training: {str(e)}")
            import traceback
            traceback.print_exc()

class DrawingWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(280, 280) 
        self.image = QImage(self.size(), QImage.Format_RGB32)
        self.image.fill(Qt.white) 
        self.last_point = QPoint()
        self.drawing = False
        self.pen_width = 25 
        self.setStyleSheet("border: 2px solid #ddd; border-radius: 4px;")

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawImage(self.rect(), self.image, self.image.rect())

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.last_point = event.position().toPoint()

    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.LeftButton) and self.drawing:
            painter = QPainter(self.image)
            pen = QPen(PASTEL_COLORS_DICT[1], self.pen_width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)
            current_point = event.position().toPoint()
            painter.drawLine(self.last_point, current_point)
            self.last_point = current_point
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False

    def clear(self):
        self.image.fill(Qt.white)
        self.update()

    def get_input_vector(self):
        scaled_image = self.image.scaled(28, 28, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        pixel_data = []
        for y in range(28):
            for x in range(28):
                pixel = scaled_image.pixelColor(x, y)
                
                green_val = pixel.green()
                
                
                inverted_val = 255 - green_val
                normalized_val = float(inverted_val) / 255.0
                
                if normalized_val < 0.1:
                    final_val = 0.0
                else:
                    
                    final_val = min(1.0, normalized_val * 2.2) 
                
                pixel_data.append(final_val)
        return pixel_data

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Handwriting Digit Recognition")
        self.setGeometry(100, 100, 1000, 600)
        self.setStyleSheet(f"background-color: {GUI_BACKGROUND_COLOR}; color: {LOG_TEXT_COLOR}; font-family: 'Segoe UI', sans-serif;")
        
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # sol kısım
        train_group = QGroupBox("Control Panel")
        train_group.setStyleSheet(f"QGroupBox {{ background-color: {CONTROL_GROUP_BG_COLOR}; border-radius: 8px; border: 1px solid #ddd; padding-top: 20px; font-weight: bold; }}")
        train_layout = QVBoxLayout()
        train_layout.setSpacing(15)
        
        input_layout = QGridLayout()
        label_style = "font-weight: bold; color: #555;"
        
        l_epoch = QLabel("Epochs:")
        l_epoch.setStyleSheet(label_style)
        self.epochs_input = QLineEdit("20")
        self.epochs_input.setStyleSheet("padding: 5px; border: 1px solid #ccc; border-radius: 4px;")
        
        l_lr = QLabel("Learning Rate:")
        l_lr.setStyleSheet(label_style)
        self.lr_input = QLineEdit("0.1")
        self.lr_input.setStyleSheet("padding: 5px; border: 1px solid #ccc; border-radius: 4px;")
        
        input_layout.addWidget(l_epoch, 0, 0)
        input_layout.addWidget(self.epochs_input, 0, 1)
        input_layout.addWidget(l_lr, 1, 0)
        input_layout.addWidget(self.lr_input, 1, 1)
        
        train_layout.addLayout(input_layout)
        
        self.train_btn = QPushButton("Train Model (Load MNIST)")
        
        self.train_btn.setStyleSheet(f"background-color: {PASTEL_COLORS_DICT[2].name()}; color: white; padding: 10px; border: none; border-radius: 5px; font-weight: bold;")
        self.train_btn.clicked.connect(self.start_training)
        train_layout.addWidget(self.train_btn)

        
        train_layout.addWidget(QLabel("Training Loss Error:"))
        self.loss_graph = LossGraphWidget()
        train_layout.addWidget(self.loss_graph)

        
        train_layout.addWidget(QLabel("Logs:"))
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setStyleSheet(f"background-color: {LOG_BG_COLOR}; border: 1px solid #ddd; border-radius: 4px; padding: 5px;")
        train_layout.addWidget(self.log_output)
        
        train_group.setLayout(train_layout)
        main_layout.addWidget(train_group, 4)

        #sağ kısım
        test_group = QGroupBox("Interactive Testing")
        test_group.setStyleSheet(f"QGroupBox {{ background-color: {CONTROL_GROUP_BG_COLOR}; border-radius: 8px; border: 1px solid #ddd; padding-top: 20px; font-weight: bold; }}")
        test_layout = QVBoxLayout()
        test_layout.setAlignment(Qt.AlignCenter)
        
        self.drawer = DrawingWidget()
        test_layout.addWidget(self.drawer)
        
        
        btn_layout = QHBoxLayout()
        self.clear_btn = QPushButton("Clear Canvas")
        
        self.clear_btn.setStyleSheet(f"background-color: {PASTEL_COLORS_DICT[4].name()}; color: #444; padding: 8px; border-radius: 4px; font-weight: bold;")
        self.clear_btn.clicked.connect(self.drawer.clear)
        btn_layout.addWidget(self.clear_btn)
        
        self.predict_btn = QPushButton("Predict Digit")
        self.predict_btn.setStyleSheet(f"background-color: {PASTEL_COLORS_DICT[5].name()}; color: white; padding: 8px; border-radius: 4px; font-weight: bold;")
        self.predict_btn.clicked.connect(self.predict_digit)
        self.predict_btn.setEnabled(False) 
        btn_layout.addWidget(self.predict_btn)
        test_layout.addLayout(btn_layout)
        
        
        self.result_label = QLabel("?")
        self.result_label.setStyleSheet(f"font-size: 48px; font-weight: bold; color: {PASTEL_COLORS_DICT[5].name()};")
        self.result_label.setAlignment(Qt.AlignCenter)
        test_layout.addWidget(self.result_label)

        
        test_layout.addSpacing(20)
        line = QWidget()
        line.setFixedHeight(1)
        line.setStyleSheet("background-color: #ddd;")
        test_layout.addWidget(line)
        
        self.batch_test_btn = QPushButton("Test 100 Random Samples")
        self.batch_test_btn.setStyleSheet(f"background-color: {PASTEL_COLORS_DICT[1].name()}; color: white; padding: 10px; border-radius: 5px; font-weight: bold; margin-top: 10px;")
        self.batch_test_btn.clicked.connect(self.run_batch_test)
        self.batch_test_btn.setEnabled(False)
        test_layout.addWidget(self.batch_test_btn)
        
        self.batch_result_label = QLabel("")
        self.batch_result_label.setAlignment(Qt.AlignCenter)
        test_layout.addWidget(self.batch_result_label)

        test_group.setLayout(test_layout)
        main_layout.addWidget(test_group, 3)

        self.worker = None
        self.trained_nn = None
        self.x_test_data = None
        self.y_test_data = None

    def log(self, text):
        self.log_output.append(text)

    def start_training(self):
        try:
            epochs = int(self.epochs_input.text())
            lr = float(self.lr_input.text())
        except ValueError:
            self.log("Invalid input for Epochs or LR")
            return

        self.train_btn.setEnabled(False)
        self.predict_btn.setEnabled(False)
        self.batch_test_btn.setEnabled(False)
        self.log_output.clear()
        
        self.worker = TrainingWorker(epochs, lr)
        self.worker.progress_log.connect(self.log)
        self.worker.finished_training.connect(self.on_training_finished)
        self.worker.start()

    def on_training_finished(self, loss_history, x_test, y_test):
        self.log("Training Complete. Model ready.")
        self.train_btn.setEnabled(True)
        self.predict_btn.setEnabled(True)
        self.batch_test_btn.setEnabled(True)
        
        self.trained_nn = self.worker.nn
        self.loss_graph.update_data(loss_history)
        
        self.x_test_data = x_test
        self.y_test_data = y_test
        
        self.log(f"Final Loss: {loss_history[-1]:.4f}")

    def predict_digit(self):
        if not self.trained_nn:
            return
        input_vec = self.drawer.get_input_vector()
        prediction_vector = self.trained_nn.predict(input_vec)
        predicted_digit = np.argmax(prediction_vector)
        self.result_label.setText(str(predicted_digit))
        self.log(f"Canvas Prediction: {predicted_digit}")

    def run_batch_test(self):
        if not self.trained_nn or not self.x_test_data:
            return
            
        self.log("Running batch test on 100 random samples...")
        
        indices = random.sample(range(len(self.x_test_data)), 100)
        
        correct = 0
        
        for idx in indices:
            img = np.array(self.x_test_data[idx]).astype('float32') / 255.0
            input_vec = img.flatten().tolist()
            
            true_label = self.y_test_data[idx]
            
            pred_vec = self.trained_nn.predict(input_vec)
            pred_label = np.argmax(pred_vec)
            
            if pred_label == true_label:
                correct += 1
                
        accuracy = correct
        self.batch_result_label.setText(f"Batch Accuracy: {accuracy}/100")
        self.log(f"Batch Test Result: {accuracy}% Accuracy")
        
        if accuracy < 10:
             self.batch_result_label.setStyleSheet("color: red; font-weight: bold; font-size: 14px;")
        elif accuracy > 80:
             self.batch_result_label.setStyleSheet("color: green; font-weight: bold; font-size: 14px;")
        else:
             self.batch_result_label.setStyleSheet("color: orange; font-weight: bold; font-size: 14px;")

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
