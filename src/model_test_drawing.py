import sys
from PySide6.QtCore import Qt, QPoint
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QPushButton,
    QLabel,
    QFileDialog,
    QHBoxLayout,
    QLineEdit,
    QComboBox,
    QSpacerItem,
    QSizePolicy,
)
from PySide6.QtGui import QPainter, QPen, QPixmap
from PIL import Image
import numpy as np
import onnxruntime
import torch
import torch.nn.functional as F
from torchvision import transforms
from pathlib import Path


class DrawingWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.pixmap = QPixmap(500, 500)
        self.pixmap.fill(Qt.white)
        self.last_point = QPoint()
        self.pen_width = 50  # Default pen width
        self.setFixedSize(500, 500)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self.pixmap)  # Adjust the coordinates here

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.last_point = event.position()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            painter = QPainter(self.pixmap)
            pen = QPen(
                Qt.black, self.pen_width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin
            )
            painter.setPen(pen)
            painter.drawLine(self.last_point, event.position())
            self.last_point = event.position()
            self.update()

    def get_image_as_numpy_array(self):
        qimage = self.pixmap.toImage()
        width = qimage.width()
        height = qimage.height()
        array = np.frombuffer(qimage.bits(), dtype=np.uint8).reshape((height, width, 4))
        array = 255 - array  # BW conversion
        img = Image.fromarray(array[:, :, :3])
        img_scaled = img.resize((28, 28))  # downsizing
        return img_scaled

    def clear(self):
        self.pixmap.fill(Qt.white)
        self.update()

    def set_pen_width(self, width):
        self.pen_width = width


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.model_path = ""
        self.model_session = None

        self.central_widget = DrawingWidget()
        self.setCentralWidget(self.central_widget)

        clear_button = QPushButton("Clear")
        clear_button.clicked.connect(self.central_widget.clear)

        self.classify_button = QPushButton("Classify")
        self.classify_button.clicked.connect(self.on_push_classify)

        self.source_folder_view = QHBoxLayout()

        self.button_set_source_folder = QPushButton("Select Model", self)
        self.source_folder_view.addWidget(self.button_set_source_folder)
        self.button_set_source_folder.clicked.connect(self.on_click_set_source_folder)

        self.test_field_source_folder = QLineEdit(self)
        self.source_folder_view.addWidget(self.test_field_source_folder)
        self.test_field_source_folder.setReadOnly(True)
        self.test_field_source_folder.setText(self.model_path)
        self.create_test_set_button = QPushButton("Create Custom Test Set")
        self.create_test_set_button.clicked.connect(self.on_create_test_set_clicked)

        self.prompt_label = QLabel("Please draw the digit 0", self)
        self.prompt_label.setMaximumHeight(20)
        self.prompt_label.setVisible(False)

        self.digit_dropdown = QComboBox(self)
        self.digit_dropdown.addItems([str(i) for i in range(10)])
        self.digit_dropdown.currentIndexChanged.connect(self.on_digit_selected)
        self.digit_dropdown.setVisible(False)

        self.prompt_text = QLabel("Please draw the digit:", self)
        self.prompt_text.setVisible(False)

        self.next_button = QPushButton("Save and Next")
        self.next_button.clicked.connect(self.on_next_button_clicked)
        self.next_button.setVisible(False)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.on_cancel_button_clicked)
        self.cancel_button.setVisible(False)

        self.model_output_view = QLabel(self)
        self.model_output_view.setText("Model Prediction:")
        self.model_output_view.setMaximumHeight(20)

        self.model_confidence_view = QLabel(self)
        self.model_confidence_view.setText("Confidence:")
        self.model_confidence_view.setMaximumHeight(20)

        pen_width_label = QLabel("Pen Width:", self)
        self.pen_width_dropdown = QComboBox(self)
        self.pen_width_dropdown.addItems([str(i) for i in range(10, 101, 10)])
        self.pen_width_dropdown.setMaximumWidth(100)
        self.pen_width_dropdown.currentIndexChanged.connect(self.update_pen_width)
        self.pen_width_dropdown.setCurrentIndex(
            self.pen_width_dropdown.findText(str(self.central_widget.pen_width))
        )

        pen_width_layout = QHBoxLayout()
        pen_width_layout.addWidget(pen_width_label)
        pen_width_layout.addWidget(self.pen_width_dropdown)
        pen_width_layout.addStretch()

        next_cancel_layout = QHBoxLayout()
        next_cancel_layout.addWidget(self.cancel_button)
        next_cancel_layout.addWidget(self.next_button)

        prompt_layout = QHBoxLayout()
        prompt_layout.addWidget(self.prompt_text)
        prompt_layout.addWidget(self.digit_dropdown)

        layout = QVBoxLayout()
        layout.addLayout(self.source_folder_view)
        layout.addLayout(pen_width_layout)
        layout.addWidget(self.central_widget)
        layout.addLayout(prompt_layout)
        layout.addLayout(next_cancel_layout)
        layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
        layout.addWidget(self.model_output_view)
        layout.addWidget(self.model_confidence_view)
        layout.addWidget(self.classify_button)
        layout.addWidget(clear_button)
        layout.addWidget(self.create_test_set_button)

        container = QWidget()
        container.setLayout(layout)

        self.setCentralWidget(container)

    def update_pen_width(self):
        selected_width = int(self.pen_width_dropdown.currentText())
        self.central_widget.set_pen_width(selected_width)

    def on_digit_selected(self, index):
        self.current_digit = index

    def on_next_button_clicked(self):
        image_as_tensor = self.convert_image_to_tensor()
        self.save_image_tensor(image_as_tensor, f"digit_{self.current_digit}.npy")

        self.current_digit += 1

        if self.current_digit > 9:
            self.activate_test_set_canvas(False)
        else:
            self.digit_dropdown.setCurrentIndex(self.current_digit)
            self.prompt_text.setText(f"Please draw the digit {self.current_digit}")
            self.central_widget.clear()

    def on_create_test_set_clicked(self):
        self.activate_test_set_canvas(True)

    def on_cancel_button_clicked(self):
        self.activate_test_set_canvas(False)

    def activate_test_set_canvas(self, enable):
        self.central_widget.clear()
        self.prompt_text.setVisible(enable)
        self.digit_dropdown.setVisible(enable)
        self.next_button.setVisible(enable)
        self.cancel_button.setVisible(enable)
        self.create_test_set_button.setVisible(not enable)
        self.current_digit = 0
        self.model_confidence_view.setVisible(not enable)
        self.model_output_view.setVisible(not enable)
        self.classify_button.setVisible(not enable)

    def convert_image_to_tensor(self):
        img = self.central_widget.get_image_as_numpy_array()
        img = img.convert("L")  # Convert the image to grayscale
        image_as_tensor = transforms.ToTensor()(img).numpy()
        image_as_tensor = np.expand_dims(
            image_as_tensor, axis=0
        )  # shape [batch_size, channels, height, width]
        return image_as_tensor

    def save_image_tensor(self, image_as_tensor, filename):
        fp = Path.cwd() / "handwriting" / filename
        fp.parent.mkdir(parents=True, exist_ok=True)
        np.save(fp, image_as_tensor)

    def on_push_classify(self):
        if self.model_session is None:
            self.model_output_view.setText("Load a model first.")
            return
        image_as_tensor = self.convert_image_to_tensor()
        self.save_image_tensor(image_as_tensor, "my_digit.npy")

        result = self.model_session.run(None, {"image": image_as_tensor})
        probabilities = (
            F.softmax(torch.tensor(np.array(result)), dim=-1).numpy().squeeze()
        )
        predicted_number = np.argmax(probabilities)
        confidence = probabilities[predicted_number]
        self.model_output_view.setText(f"Model Prediction: {predicted_number}")
        self.model_confidence_view.setText(f"Confidence: {100 * confidence:.1f}%")
        print(f"Predicted number: {predicted_number}, Confidence: {confidence:.4f}")

    def on_click_set_source_folder(self):
        self.model_path, _ = QFileDialog.getOpenFileName(self, "Select Model")
        self.test_field_source_folder.setText(self.model_path)
        self.model_session = onnxruntime.InferenceSession(
            self.model_path, providers=["CPUExecutionProvider"]
        )


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.setWindowTitle("GenAI Exercise - MNIST Digits Test")
    window.resize(550, 650)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
