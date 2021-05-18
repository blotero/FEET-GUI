#GCPDS - Universidad Nacional de Colombia
#Proyecto caracterización termográfica de extremidades inferiores durante aplicación de anestesia epidural
# Mayo de 2021

import os
from pathlib import Path
import sys
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from PySide2.QtWidgets import QApplication, QMainWindow, QGraphicsView, QLabel, QPushButton, QTextBrowser, QAction, QFileDialog, QDialog, QDialog
from PySide2.QtCore import QFile, QObject, SIGNAL, QDir
from PySide2.QtUiTools import QUiLoader
from PySide2.QtGui import QImage, QPixmap
from segment import Image_2_seg
from manualseg import manualSeg


class Window(QMainWindow):
    def __init__(self):
        super(Window, self).__init__()
        self.load_ui()        
        self.figlabels()
        self.imgs = []
        self.subj = []
        self.figlabels = Image.open('figlabels.png')

    def load_ui(self):
        loader = QUiLoader()        
        path = os.fspath(Path(__file__).resolve().parent / "form.ui")
        ui_file = QFile(path)
        ui_file.open(QFile.ReadOnly)
        self.ui_window = loader.load(ui_file, self)
        ui_file.close()

    def feet_segment(self):
        im = Image_2_seg()
        im.extract()
        print(im.bin_img)

    def segment(self):
        self.ui_window.textBrowser.setSource('out.html')
        self.feet_segment()

    def manual_segment(self):
        print("Se abrirá diálogo de extracción manual")
        self.manual=manualSeg()
        self.manual.show()

    def temp_extract(self):
        print("Se extraerá temperatura")
    
    def figlabels(self):
        img=Image.open('figlabels.png')

    def abrir_imagen(self):
        self.filedialog=QFileDialog(self)
        self.filedialog.setDirectory(QDir.currentPath())        
        if self.filedialog.exec_() == QDialog.Accepted:
            return self.imgs.append( self.filedialog.selectedUrls()[0])           
       
    def abrir_carpeta(self):
        print("Se abrirá la carpeta")
        self.filedialog=QFileDialog(self)
        self.filedialog.setDirectory('/home/brandon/FEET_GUI/piecitos/database/Images/users/lauralrh')        
        #if self.filedialog.exec_() == QDialog.Accepted:
        return self.filedialog.selectedUrls()[0]      
        
    def abrir_proyecto(self):
        print("Se abrirá un proyecto de extensión .feet")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Window()

    QObject.connect(window.ui_window.segButton, SIGNAL ('clicked()'), window.segment)
    QObject.connect(window.ui_window.tempButton, SIGNAL ('clicked()'), window.temp_extract)
    QObject.connect(window.ui_window.manualSegButton, SIGNAL ('clicked()'), window.manual_segment)
    QObject.connect(window.ui_window.actionCargar_imagen, SIGNAL ('triggered()'), window.abrir_imagen)
    QObject.connect(window.ui_window.actionCargar_imagen, SIGNAL ('triggered()'), window.abrir_imagen)
    QObject.connect(window.ui_window.actionCargar_imagen, SIGNAL ('triggered()'), window.abrir_imagen)
    QObject.connect(window.ui_window.actionCarpeta_de_imagenes , SIGNAL ('triggered()'), window.abrir_carpeta)
    window.ui_window.show()   

    sys.exit(app.exec_())
