# GCPDS - Universidad Nacional de Colombia
# Proyecto caracterización termográfica de extremidades inferiores durante aplicación de anestesia epidural
# Mayo de 2021
# Disponible en https//:github.com/blotero/FEET-GUI

import os
import re
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
from segment import ImageToSegment, SessionToSegment, remove_small_objects
from manualseg import manualSeg
from temperatures import mean_temperature

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
        self.annotationExists : False
        self.isSegmented = False
        self.files = None
        self.temperaturesWereAcquired = False

    def load_ui(self):
        loader = QUiLoader()        
        path = os.fspath(Path(__file__).resolve().parent / "form.ui")
        ui_file = QFile(path)
        ui_file.open(QFile.ReadOnly)
        self.ui_window = loader.load(ui_file, self)
        ui_file.close()
    
    def make_connect(self):
        QObject.connect(self.ui_window.actionCargar_imagen, SIGNAL ('triggered()'), self.openImage)
        QObject.connect(self.ui_window.actionCargar_carpeta , SIGNAL ('triggered()'), self.openFolder)
        QObject.connect(self.ui_window.segButton, SIGNAL ('clicked()'), self.segment)
        QObject.connect(self.ui_window.tempButton, SIGNAL ('clicked()'), self.temp_extract)
        QObject.connect(self.ui_window.manualSegButton, SIGNAL ('clicked()'), self.manual_segment)
        QObject.connect(self.ui_window.refreshTimePlot , SIGNAL ('clicked()'), self.makeTimePlot)
        QObject.connect(self.ui_window.nextImageButton , SIGNAL ('clicked()'), self.nextImage)
        QObject.connect(self.ui_window.previousImageButton , SIGNAL ('clicked()'), self.previousImage)
        QObject.connect(self.ui_window.saveButton , SIGNAL ('clicked()'), self.saveImage)
        QObject.connect(self.ui_window.fullPlotButton , SIGNAL ('clicked()'), self.fullPlot)
        QObject.connect(self.ui_window.reportButton , SIGNAL ('clicked()'), self.exportReport)

    def messagePrint(self, message):
        #INPUT: string to print
        #OUTPUT: none
        #ACTION: generate out.html file and refresh it in Messages QTextArea
        log_path = "outputs/logs.html"
        out_file = open(log_path , "w")
        out_file.write(message)
        out_file.close()
        self.ui_window.textBrowser.setSource(log_path)
        self.ui_window.textBrowser.reload()

    def findImages(self):
        self.fileList = []
        for root, dirs, files in os.walk(self.defaultDirectory):
            for file in files:
                if (file.endswith(".jpg")):
                    self.fileList.append(os.path.join(root,file))
        self.imageQuantity = len(self.fileList)
        self.imageIndex = 0
        self.files = files
        self.sortFiles()
        self.outfiles=[]
        for i in range(len(files)):
            self.outfiles.append("outputs/" + files[i]) #Creating future output file names
        self.ui_window.inputLabel.setText(self.files[self.imageIndex])

    def sortFiles(self):
        """Sort file list to an alphanumeric reasonable sense"""         
        convert = lambda text: int(text) if text.isdigit() else text 
        alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
        self.fileList =  sorted(self.fileList, key = alphanum_key)
        self.files =  sorted(self.files, key = alphanum_key)

    def getTimes(self):
        """
        Converts standarized names of file list into a list of 
        integers with time capture in minutes
        """
        if (type(self.fileList)==str):
            self.timeList =  int(self.fileList).rsplit(".")[0][1:]
        elif type(self.fileList)==list:    
            out_list = []
            for i in range(len(self.fileList)):
                out_list.append(int(self.files[i].rsplit(".")[0][1:]))
            self.timeList =  out_list
        else:
            return None

    def nextImage(self):
        if self.imageIndex < len(self.fileList)-1:
            self.imageIndex += 1
            self.ui_window.inputImg.setPixmap(self.fileList[self.imageIndex])
            self.opdir = self.fileList[self.imageIndex]
            self.ui_window.inputLabel.setText(self.files[self.imageIndex])

            if self.sessionIsSegmented:
                #Sentences to display next output image if session was already
                #segmented
                self.showOutputImageFromSession()
                if self.temperaturesWereAcquired:
                    self.messagePrint("La temperatura media es: " + str(self.meanTemperatures[self.imageIndex]))
                    self.ui_window.temperatureLabel.setText(str(np.round(self.meanTemperatures[self.imageIndex], 3)))
                
    def previousImage(self):
        if self.imageIndex >= 1:
            self.imageIndex -= 1
            self.ui_window.inputImg.setPixmap(self.fileList[self.imageIndex])
            self.opdir = self.fileList[self.imageIndex]
            self.ui_window.inputLabel.setText(self.files[self.imageIndex])

            if self.sessionIsSegmented:
                #Sentences to display next output image if session was already
                #segmented
                self.showOutputImageFromSession()
                if self.temperaturesWereAcquired:
                    self.messagePrint("La temperatura media es: " + str(self.meanTemperatures[self.imageIndex]))
                    self.ui_window.temperatureLabel.setText(str(np.round(self.meanTemperatures[self.imageIndex], 3)))

    def saveImage(self):
        #Saves segmented image
        pass

    def feetSegment(self):
        self.messagePrint("Segmentando imagen")
        self.i2s = ImageToSegment()
        self.i2s.setPath(self.opdir)
        self.i2s.extract()
        self.showSegmentedImage()
        self.isSegmented = True
        self.messagePrint("Imagen segmentada exitosamente")

    def sessionSegment(self):
        self.messagePrint("Segmentando toda la sesion...")
        self.s2s = SessionToSegment()
        self.s2s.setPath(self.defaultDirectory)
        self.s2s.whole_extract(self.fileList)
        self.produceSegmentedSessionOutput()
        self.showOutputImageFromSession()
        self.messagePrint("Se ha segmentado exitosamente la sesion")
        self.sessionIsSegmented = True

    def showSegmentedImage(self):
        #Applies segmented zone to input image, showing only feet
        threshold =  0.5
        img = plt.imread(self.opdir)/255
        Y = self.i2s.Y_pred
        Y = Y / Y.max()
        Y = np.where( Y >= threshold  , 1 , 0)
        self.Y =remove_small_objects( Y[0])     #Eventually required by temp_extract
        Y = cv2.resize(Y[0], (img.shape[1],img.shape[0]), interpolation = cv2.INTER_NEAREST) # Resize the prediction to have the same dimensions as the input 
        plt.imsave("outputs/output.jpg" , Y*img[:,:,0] , cmap='gray')
        self.ui_window.outputImg.setPixmap("outputs/output.jpg")
    
    def produceSegmentedSessionOutput(self):
        #Recursively applies showSegmentedImage to whole session
        self.Y=[]
        for i in range(len(self.outfiles)):
            threshold =  0.5
            img = plt.imread(self.fileList[i])/255
            Y = self.s2s.Y_pred[i]
            Y = Y / Y.max()
            Y = np.where( Y >= threshold  , 1 , 0)
            self.Y.append( remove_small_objects(Y))     #Eventually required by temp_extract
            Y = cv2.resize(Y, (img.shape[1],img.shape[0]), interpolation = cv2.INTER_NEAREST) # Resize the prediction to have the same dimensions as the input 
            plt.imsave(self.outfiles[i], Y*img[:,:,0] , cmap='gray')


    def showOutputImageFromSession(self):
        self.ui_window.outputImg.setPixmap(self.outfiles[self.imageIndex])

    def segment(self):
        if self.ui_window.sessionCheckBox.isChecked():

            if self.defaultDirectoryExists:
                self.sessionSegment()
                print("Entering session segment")
            else:
                self.messagePrint("No se ha seleccionado sesion de entrada")
        else:
            if self.inputExists:
                self.feetSegment()
                print("Entering image segment")
            else:
                self.messagePrint("No se ha seleccionado sesion de entrada")

    def manual_segment(self):
        print("Se abrirá diálogo de extracción manual")
        self.manual=manualSeg()
        self.manual.show()
        return

    def temp_extract(self):
        if (self.inputExists and (self.isSegmented or self.sessionIsSegmented)):
            if self.ui_window.sessionCheckBox.isChecked():   #If segmentation was for full session
                self.meanTemperatures = []   #Whole feet mean temperature for all images in session
                for i in range(len(self.outfiles)):
                    self.meanTemperatures.append(mean_temperature(self.s2s.Xarray[i,:,:,0] , self.Y[i][:,:,0] , plot = False))
                self.messagePrint("La temperatura media es: " + str(self.meanTemperatures[self.imageIndex]))
                self.temperaturesWereAcquired = True
            else:      #If segmentation was for single image
                mean = mean_temperature(self.i2s.Xarray[:,:,0] , self.Y[:,:,0] , plot = False)
                self.messagePrint("La temperatura media es: " + str(mean))

            if (self.ui_window.plotCheckBox.isChecked()):  #If user asked for plot
                self.messagePrint("Se generara plot de temperatura...")
                self.getTimes()
                print(self.timeList)
                self.tempPlot()

        elif self.inputExists:
            self.messagePrint("No se ha segmentado previamente la imagen ")
        else:
            self.messagePrint("No se han seleccionado imagenes de entrada")



    def tempPlot(self):
        plt.figure()
        plt.plot(self.timeList , self.meanTemperatures, '-o')
        plt.title("Temperatura media de pies")
        plt.xlabel("Tiempo (s)")
        plt.ylabel("Temperatura (°C)")
        plt.grid()
        plt.show()
        #Produce plot 

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
        self.imagesDir = os.path.dirname(self.opdir) 
        if self.opdir:
            self.inputExists = True
            self.ui_window.inputImg.setPixmap(self.opdir)

    def openFolder(self):
        self.folderDialog=QFileDialog(self)
        self.folderDialog.setDirectory(QDir.currentPath())        
        self.folderDialog.setFileMode(QFileDialog.FileMode.Directory)
        self.defaultDirectory = self.folderDialog.getExistingDirectory()
        self.imagesDir = self.defaultDirectory
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
            plt.savefig('/ouputs/fresh.png')
            self.ui_window.timePlot.setPixmap('/outputs/outputs//fresh.png')
            self.messagePrint("Se ha actualizado el TimePlot")
        else:
            self.messagePrint("No se puede actualizar el TimePlot. No se ha seleccionado una imagen de entrada")

    def fullPlot(self):
        self.messagePrint("Preparando full plot...")
        #show_temperatures("paciente" , fn="mean" , range_ = [22.5 , 33.5])
        self.messagePrint("Full plot generado exitosamente")
        pass

    def exportReport(self):
        self.messagePrint("Generando reporte...") 
        #GENERATE A PDF REPORT FOR THE PATIENT
        #INPUT: SELF, PATIENT DIR
        #RETURN: NONE
        #ACTION: COMPILE PDF TEXT BASED ON
        self.messagePrint("Reporte generado exitosamente")
        pass

    def animate(self):      
        """
        Produces gif animation based on mean temperatures for whole session
        Initially, all feet has same color, for section segmentation has been not implemented yet
        """
        self.messagePrint("Iniciando animacion...")
        pass


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    window.ui_window.show()  
    sys.exit(app.exec_())
