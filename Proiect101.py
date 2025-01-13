import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt
import joblib  
import sys
import pandas as pd
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QPushButton,
    QVBoxLayout, QHBoxLayout, QComboBox, QSpinBox, QDoubleSpinBox, QMessageBox, QSpacerItem, QSizePolicy
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QPalette
from PyQt6.QtWidgets import QLabel


class StrokePredictorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stroke Risk Predictor")
        self.setGeometry(100, 100, 500, 600)
        self.model = joblib.load(r'C:/Users/alexb/Desktop/Facultate/IV/Sem1/AI/xgboost_model.pkl')
        self.scaler = joblib.load(r'C:/Users/alexb/Desktop/Facultate/IV/Sem1/AI/scaler.pkl')
        self.model_columns = self.model.feature_names_in_
        self.numerical_columns = ['age', 'avg_glucose_level', 'bmi']

        self.initUI()

    def initUI(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Layout principal
        self.layout = QVBoxLayout(self.central_widget)
        self.layout.setSpacing(10)  # Modifică spațiul între widgeturi la 10px
        self.layout.setContentsMargins(0, 0, 0, 0)  # Eliminăm marginile suplimentare ale layout-ului

        # Setează fundalul
        self.set_background_image("C:/Users/alexb/Desktop/Facultate/IV/py101/pozafundal.jpg")

        # Spacer înainte pentru a centra pe verticală
        spacer_top = QSpacerItem(20, 100, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        self.layout.addItem(spacer_top)

        # Name and input layout
        name_input_layout = QVBoxLayout()
        self.name_label = QLabel("Numele pacientului:", alignment=Qt.AlignmentFlag.AlignCenter)
        self.name_label.setStyleSheet("font-size: 16px;")
        self.name_input = QLineEdit()
        self.name_input.setFixedWidth(300)
        self.name_input.setAlignment(Qt.AlignmentFlag.AlignCenter)

        name_input_layout.addWidget(self.name_label)
        name_input_layout.addWidget(self.name_input)

        # Grupăm "Numele pacientului" și "Caseta text" într-un QHBoxLayout pentru a le alinia pe centru
        name_input_container = QHBoxLayout()
        name_input_container.setContentsMargins(0, 0, 0, 0)
        name_input_container.addLayout(name_input_layout)
        name_input_container.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.layout.addLayout(name_input_container)

        # Start button layout
        start_button_layout = QVBoxLayout()
        self.start_button = QPushButton("Start Evaluare")
        self.start_button.setFixedSize(300, 50)
        self.start_button.setStyleSheet("font-size: 14px;")
        self.start_button.clicked.connect(self.show_main_form)
        start_button_layout.addWidget(self.start_button)

        # Grupăm butonul într-un container QHBoxLayout pentru a-l alinia pe centru
        start_button_container = QHBoxLayout()
        start_button_container.setContentsMargins(0, 0, 0, 0)
        start_button_container.addLayout(start_button_layout)
        start_button_container.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.layout.addLayout(start_button_container)

        # Adăugăm un spacer între buton și formular
        spacer1 = QSpacerItem(20, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        self.layout.addItem(spacer1)

        # Title for the main form (hidden initially)
        self.title = QLabel("", alignment=Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.title)
        self.title.hide()

        # Main form
        self.main_form = QWidget()
        self.form_layout = QVBoxLayout()
        self.create_form()
        self.main_form.setLayout(self.form_layout)
        self.layout.addWidget(self.main_form)

        # Back button
        self.go_back_btn = QPushButton("Introduceti alt pacient", self)
        self.go_back_btn.clicked.connect(self.go_back)
        self.layout.addWidget(self.go_back_btn, alignment=Qt.AlignmentFlag.AlignCenter)

        # Spacer pentru a centra tot pe verticală
        spacer_bottom = QSpacerItem(20, 100, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        self.layout.addItem(spacer_bottom)

        self.main_form.hide()
        self.go_back_btn.hide()

    def set_background_image(self, image_path):
        # Create a QLabel to set as background
        self.background_label = QLabel(self.central_widget)
        self.background_pixmap = QPixmap(image_path)
        self.background_label.setPixmap(self.background_pixmap)
        self.background_label.setGeometry(0, 0, self.central_widget.width(), self.central_widget.height())
        self.background_label.setScaledContents(True)
        self.background_label.lower()

        self.background_layout = QVBoxLayout(self.central_widget)
        self.background_layout.addWidget(self.background_label, 0, Qt.AlignmentFlag.AlignCenter)
    
        # Adăugați restul elementelor de UI după ce fundalul a fost setat
        # După setarea fundalului, adaugă restul layout-urilor și widgeturilor, cum ar fi self.layout etc.
        self.central_widget.setLayout(self.layout)  

    def resizeEvent(self, event):
        # Redimensionează fundalul când fereastra este redimensionată
        if hasattr(self, 'background_label'):
            background_pixmap = self.background_label.pixmap()
            self.background_label.setGeometry(0, 0, self.width(), self.height())
            self.background_label.setPixmap(background_pixmap.scaled(self.width(), self.height(), Qt.AspectRatioMode.KeepAspectRatioByExpanding))
        super().resizeEvent(event)

        

    def create_form(self):
        # Gender
        self.gender_label = QLabel("Gen:")
        self.gender_input = QComboBox()
        self.gender_input.addItems(["Femeie", "Barbat"])

        # Age
        self.age_label = QLabel("Vârsta:")
        self.age_input = QSpinBox()
        self.age_input.setRange(0, 120)

        # Hypertension
        self.hyper_label = QLabel("Hipertensiune:")
        self.hyper_input = QComboBox()
        self.hyper_input.addItems(["Nu", "Da"])

        # Heart Disease
        self.heart_label = QLabel("Boală cardiacă:")
        self.heart_input = QComboBox()
        self.heart_input.addItems(["Nu", "Da"])

        # Ever Married
        self.married_label = QLabel("Ești căsătorit?")
        self.married_input = QComboBox()
        self.married_input.addItems(["Nu", "Da"])

        # Work Type
        self.work_label = QLabel("Tipul locului de muncă:")
        self.work_input = QComboBox()
        self.work_input.addItems(["Privat", "Independent", "Loc de munca la stat", "Nu am lucrat niciodata", "Copii"])

        # Residence Type
        self.residence_label = QLabel("Tipul locuinței:")
        self.residence_input = QComboBox()
        self.residence_input.addItems(["Urban", "Rural"])

        # Avg Glucose Level
        self.glucose_label = QLabel("Nivelul mediu de glucoză:")
        self.glucose_input = QDoubleSpinBox()
        self.glucose_input.setRange(0, 500)

        # BMI
        self.bmi_label = QLabel("BMI:")
        self.bmi_input = QDoubleSpinBox()
        self.bmi_input.setRange(0, 100)
        self.bmi_input.setDecimals(1)

        # Smoking Status
        self.smoking_label = QLabel("Status fumător:")
        self.smoking_input = QComboBox()
        self.smoking_input.addItems(["Nefumator", "Fost fumator", "Fumator"])

        # Add all fields to the form
        for label, input_widget in [
            (self.gender_label, self.gender_input),
            (self.age_label, self.age_input),
            (self.hyper_label, self.hyper_input),
            (self.heart_label, self.heart_input),
            (self.married_label, self.married_input),
            (self.work_label, self.work_input),
            (self.residence_label, self.residence_input),
            (self.glucose_label, self.glucose_input),
            (self.bmi_label, self.bmi_input),
            (self.smoking_label, self.smoking_input)
        ]:
            self.form_layout.addWidget(label)
            self.form_layout.addWidget(input_widget)

        # Submit button
        self.submit_button = QPushButton("Predicție Risc AVC")
        self.submit_button.clicked.connect(self.calculate_risk)
        self.form_layout.addWidget(self.submit_button)

    def show_main_form(self):
        name = self.name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Eroare", "Te rog introdu numele pacientului.")
            return
        self.title.setText(f"Evaluare pentru {name}")
        self.title.show()
        self.name_label.hide()
        self.name_input.hide()
        self.start_button.hide()
        self.main_form.show()
        self.go_back_btn.show()

    def go_back(self):
        # Reset and return to name input form
        self.name_input.clear()
        self.title.hide()
        self.main_form.hide()
        self.name_label.show()
        self.name_input.show()
        self.start_button.show()
        self.go_back_btn.hide()

    def calculate_risk(self):
        # Convert inputs to the format needed by your model
        data = {
            'gender': self.gender_input.currentText(),
            'age': self.age_input.value(),
            'hypertension': 1 if self.hyper_input.currentText() == "Da" else 0,
            'heart_disease': 1 if self.heart_input.currentText() == "Da" else 0,
            'ever_married': 1 if self.married_input.currentText() == "Da" else 0,
            'work_type': self.work_input.currentText(),
            'Residence_type': self.residence_input.currentText(),
            'avg_glucose_level': self.glucose_input.value(),
            'bmi': self.bmi_input.value(),
            'smoking_status': self.smoking_input.currentText()
        }

        # Map categorical inputs to columns (following training schema)
        input_df = pd.DataFrame([data])
        input_encoded = pd.get_dummies(input_df, columns=[
            'gender', 'ever_married', 'work_type', 
            'Residence_type', 'smoking_status'
        ], drop_first=True)

        # Ensure input_encoded matches model's training columns
        for col in [col for col in self.model_columns if col not in input_encoded.columns]:
            input_encoded[col] = 0

        input_encoded = input_encoded[self.model_columns]

        # Standardize numerical features
        numerical_columns = ['age', 'avg_glucose_level', 'bmi']
        input_encoded[numerical_columns] = self.scaler.transform(input_encoded[numerical_columns])

        # Prediction
        try:
            prediction = self.model.predict(input_encoded)[0]
            prediction_prob = self.model.predict_proba(input_encoded)[0][1]
        except Exception as e:
            QMessageBox.critical(self, "Eroare", f"Eroare la predicție: {str(e)}")
            return

        # Display result
        if prediction == 1:
            QMessageBox.information(
                self, "Rezultat", 
                f"Pacientul are risc de AVC cu o probabilitate de {prediction_prob * 100:.2f}%."
            )
        else:
            QMessageBox.information(
                self, "Rezultat", 
                f"Pacientul NU are risc de AVC."
            )

        # Save data to CSV
        self.save_to_csv(data)

    def save_to_csv(self, patient_data):
        # Check if file exists
        try:
            df = pd.read_csv("patients_data.csv")
        except FileNotFoundError:
            df = pd.DataFrame(columns=['Index', 'Name', 'Gender', 'Age', 'Hypertension', 'Heart Disease', 'Ever Married', 'Work Type', 'Residence Type', 'Glucose Level', 'BMI', 'Smoking Status'])
            patient_data['Index'] = len(df) + 1  # Index nou
        df = df._append(patient_data, ignore_index=True)

        # Salvăm în CSV
        df.to_csv("patients_data.csv", index=False)
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = StrokePredictorApp()
    window.show()
    sys.exit(app.exec())
