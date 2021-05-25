#GCPDS - Universidad Nacional de Colombia
#Proyecto caracterización termográfica de extremidades inferiores durante aplicación de anestesia epidural
# Mayo de 2021
#Disponible en https//:github.com/blotero/FEET-GUI

import os
from PIL import Image
from pathlib import Path
import sys
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
       # self.figlabels = cv2.imread('figlabels.png')
        self.make_connect()
        self.inputExists = False
        self.defaultDirectoryExists = False

    def make_connect(self):
        QObject.connect(self.ui_window.segButton, SIGNAL ('clicked()'), self.segment)
        QObject.connect(self.ui_window.tempButton, SIGNAL ('clicked()'), self.temp_extract)
        QObject.connect(self.ui_window.manualSegButton, SIGNAL ('clicked()'), self.manual_segment)
        QObject.connect(self.ui_window.actionCargar_imagen, SIGNAL ('triggered()'), self.openImage)
        QObject.connect(self.ui_window.actionCargar_carpeta , SIGNAL ('triggered()'), self.openFolder)
        QObject.connect(self.ui_window.refreshTimePlot , SIGNAL ('clicked()'), self.makeTimePlot)
 
 
    def load_ui(self):
        loader = QUiLoader()        
        path = os.fspath(Path(__file__).resolve().parent / "form.ui")
        ui_file = QFile(path)
        ui_file.open(QFile.ReadOnly)
        self.ui_window = loader.load(ui_file, self)
        ui_file.close()

    def feet_segment(self):
        self.i2s = Image_2_seg()
        self.i2s.setPath(self.opdir)
        self.i2s.extract()
        print(self.i2s.Y_pred)
        self.ui_window.outputImg.setPixmap(self.opdir)

    def segment(self):
        out_file = open("out.html" , "w")
        if self.inputExists:
            out_file.write("Imagen segmentada exitosamente")
            out_file.close()
            self.ui_window.textBrowser.setSource('out.html')
            self.ui_window.textBrowser.reload()
            self.feet_segment()
        else:
            out_file.write("No se ha seleccionado imagen de entrada")
            out_file.close()
            self.ui_window.textBrowser.setSource('out.html')
            self.ui_window.textBrowser.reload()

    def manual_segment(self):
        print("Se abrirá diálogo de extracción manual")
        self.manual=manualSeg()
        self.manual.show()
        return

    def temp_extract(self):
        out_file = open("out.html" , "w")
        if self.inputExists:
            out_file.write("Se extrajo temperatura exitosamente")
            out_file.close()
            self.ui_window.textBrowser.setSource('out.html')
            self.ui_window.textBrowser.reload()
        else:
            out_file.write("No se ha seleccionado imagen de entrada")
            out_file.close()
            self.ui_window.textBrowser.setSource('out.html')
            self.ui_window.textBrowser.reload()
    
    def figlabels(self):
        pass

    def openImage(self):
        self.fileDialog=QFileDialog(self)
        if self.defaultDirectoryExists:
            self.fileDialog.setDirectory(self.defaultDirectory)
        else:
            self.fileDialog.setDirectory(QDir.currentPath())        
        filters =  ["*.png", "*.xpm", "*.jpg"]
        self.fileDialog.setNameFilters("Images (*.png *.jpg)")
        self.fileDialog.selectNameFilter("Images (*.png *.jpg)")
        #self.fileDialog.setFilter(self.fileDialog.selectedNameFilter())
        self.opdir = self.fileDialog.getOpenFileName()[0]
        if self.opdir:
            self.inputExists = True
            self.ui_window.inputImg.setPixmap(self.opdir)

    def openFolder(self):
        self.folderDialog=QFileDialog(self)
        self.folderDialog.setDirectory(QDir.currentPath())        
        self.folderDialog.setFileMode(QFileDialog.FileMode.Directory)
        self.defaultDirectory = self.folderDialog.getExistingDirectory()
        if self.defaultDirectory:
            self.defaultDirectoryExists = True
            print(self.defaultDirectory) 

    def makeTimePlot(self):
        x=np.array([0,1,5,10,15,20])
        y=np.array([35.5, 35.7 , 36 , 37.2 , 37.3, 37.5])
        fig=plt.figure(figsize=(9.6,4))
        plt.plot(x,y,label='Paciente 1')
        plt.legend()
        plt.grid()
        plt.xlabel("Tiempo [minutos]")
        plt.ylabel("Temperatura [°C]")
        plt.title("Time plot")
        #plt.show()
        plt.savefig('fresh.png')
        self.ui_window.timePlot.setPixmap('fresh.png')


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    window.ui_window.show()  
    sys.exit(app.exec_())
