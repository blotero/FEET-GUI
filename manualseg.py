# This Python file uses the following encoding: utf-8
import sys
from PySide2.QtWidgets import QApplication, QWidget


class manualSeg(QWidget):
    def __init__(self):
        QWidget.__init__(self)

if __name__ == "__main__":
    sub_app = QApplication([])
    sub_window = manualSeg()
    sub_window.show()
    #sys.exit(app.exec_())
