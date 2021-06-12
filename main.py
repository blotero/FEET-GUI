# GCPDS - Universidad Nacional de Colombia
# Proyecto caracterización termográfica de extremidades inferiores durante aplicación de anestesia epidural
# Mayo de 2021
# Disponible en https//:github.com/blotero/FEET-GUI

import os
from PIL import Image
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
import cv2
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

    def load_ui(self):
        loader = QUiLoader()        
        path = os.fspath(Path(__file__).resolve().parent / "form.ui")
        ui_file = QFile(path)
        ui_file.open(QFile.ReadOnly)
        self.ui_window = loader.load(ui_file, self)
        ui_file.close()
    
    def make_connect(self):
        QObject.connect(self.ui_window.segButton, SIGNAL ('clicked()'), self.segment)
        QObject.connect(self.ui_window.tempButton, SIGNAL ('clicked()'), self.temp_extract)
        QObject.connect(self.ui_window.manualSegButton, SIGNAL ('clicked()'), self.manual_segment)
        QObject.connect(self.ui_window.actionCargar_imagen, SIGNAL ('triggered()'), self.openImage)
        QObject.connect(self.ui_window.actionCargar_carpeta , SIGNAL ('triggered()'), self.openFolder)
        QObject.connect(self.ui_window.refreshTimePlot , SIGNAL ('clicked()'), self.makeTimePlot)
        QObject.connect(self.ui_window.nextImageButton , SIGNAL ('clicked()'), self.nextImage)
        QObject.connect(self.ui_window.previousImageButton , SIGNAL ('clicked()'), self.previousImage)
        QObject.connect(self.ui_window.saveButton , SIGNAL ('clicked()'), self.saveImage)

    def messagePrint(self, message):
        #INPUT: string to print
        #OUTPUT: none
        #ACTION: generate out.html file and refresh it in Messages QTextArea
        out_file = open("out.html" , "w")
        out_file.write(message)
        out_file.close()
        self.ui_window.textBrowser.setSource('out.html')
        self.ui_window.textBrowser.reload()

    def findImages(self):
        self.fileList = []
        for root, dirs, files in os.walk(self.defaultDirectory):
            for file in files:
                if (file.endswith(".jpg")):
                    self.fileList.append(os.path.join(root,file))
        self.fileList.sort()
        self.imageIndex = 0

    def nextImage(self):
        if self.imageIndex < len(self.fileList):
            self.imageIndex += 1
            self.ui_window.inputImg.setPixmap(self.fileList[self.imageIndex])

    def previousImage(self):
        if self.imageIndex > 0:
            self.imageIndex -= 1
            self.ui_window.inputImg.setPixmap(self.fileList[self.imageIndex])

    def saveImage(self):
        #Saves segmented image
        pass
 
    def feet_segment(self):
        self.i2s = Image_2_seg()
        self.i2s.setPath(self.opdir)
        self.i2s.extract()
        threshold =  0.5
        img = plt.imread(self.opdir)/255
        Y = self.i2s.Y_pred
        Y = Y / Y.max()
        Y = np.where( Y >= threshold  , 1 , 0)
        Y = cv2.resize(Y[0], (img.shape[1],img.shape[0]), interpolation = cv2.INTER_NEAREST) # Resize the prediction to have the same dimensions as the input 
        #Y_im = cv2.cvtColor(Y, cv2.COLOR_BGR2RGB)
        #cv2.imwrite("output.jpg", Y_im) 
        plt.imsave("output.jpg" , Y*0.5+img[:,:,0] , cmap='gray')
        self.ui_window.outputImg.setPixmap("output.jpg")

    def segment(self):
        if self.inputExists:
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
        if self.inputExists:
            self.messagePrint("Se extrajo la temperatura exitosamente")
        else:
            self.messagePrint("No se ha seleccionado un imagen de entrada")
    
    def figlabels(self):
        #  Get info from directory path name and obtain time indexes based on name
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
            first_image = str(self.defaultDirectory + "/t0.jpg")
            print(first_image)
            self.ui_window.inputImg.setPixmap(first_image)
            self.opdir = first_image
            self.inputExists = True
            self.findImages()

    def makeTimePlot(self):
        if self.inputExists:
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
            self.messagePrint("Se ha actualizado el TimePlot")
        else:
            self.messagePrint("No se puede actualizar el TimePlot. No se ha seleccionado una imagen de entrada")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    window.ui_window.show()  
    sys.exit(app.exec_())
