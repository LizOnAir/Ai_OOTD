import sys
from PyQt5.QtWidgets import (QApplication, QLabel, QPushButton, QWidget, QVBoxLayout, QLineEdit, QHBoxLayout)
from PyQt5.QtCore import Qt
from google import genai

class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Ai OOTD')
        self.prompt_input = QLineEdit(self)
        self.get_idea_button = QPushButton('Get Ideas',self)
        self.idea_desc = QLabel(self)

        self.initUI()
        self.get_idea_button.clicked.connect(self.ai_response())

    def initUI(self):

        vbox = QVBoxLayout()
        vbox.addWidget(self.prompt_input)
        vbox.addWidget(self.get_idea_button)
        self.setLayout(vbox)

        self.prompt_input.setObjectName("promt_input")
        self.get_idea_button.setObjectName("get_idea_button")

        self.setStyleSheet("""
            QLineEdit#prompt_input{
                font-size:50px;            
            }
            QPushButton#get_idea_button{
                font-size:30px;
                font-weight: bold;
                border: 3px solid white;
                border-radius: 10px;
                background-color: hsl(0, 2%, 18%)
            }
            QPushButton#get_idea_button:hover{
                background-color: hsl(0, 0%, 31%);
            }
        """)

        self.prompt_input.setPlaceholderText("What is your style?")\
    #
    # def ai_response(self):
    #     API_key = 'AIzaSyBbpj5Wpwl1uxAGBfCf1bF1SVBrU9_xLTA'
    #     client = genai.Client(api_key=API_key)
    #     prompt = self.prompt_input.text()
    #
    #     response = client.models.generate_content(
    #         model='gemini-2.5-flash', contents=f'Give me 3 idea of ootd based on my style{prompt}'
    #     )
    #
    #     self.idea_desc.setText(response.text)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    Window = Window()
    Window.show()
    sys.exit(app.exec_())
