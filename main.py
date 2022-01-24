# GCPDS - Universidad Nacional de Colombia
# Proyecto caracterización termográfica de extremidades inferiores durante aplicación de anestesia epidural
# Mayo de 2021
# Disponible en https//:github.com/blotero/FEET-GUI

import os
import re
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
from PySide2.QtWidgets import QApplication, QMainWindow, QFileDialog 
from PySide2.QtCore import QFile, QObject, SIGNAL, QDir, QTimer
from PySide2.QtUiTools import QUiLoader 
from segment import ImageToSegment, SessionToSegment, remove_small_objects
from manualseg import manualSeg
from temperatures import mean_temperature
from scipy.interpolate import make_interp_spline 
import cv2
from PySide2.QtWidgets import *
from PySide2.QtCore import *
from PySide2.QtGui import *
from datetime import datetime
import tflite_runtime.interpreter as tflite
import easyocr

class RemotePullException(Exception):
    def __init__(self, repoURL):
        self.message = "Error pulling new changes into local DB from origin: " + str(repoURL)
        super.__init__(self.message)


class RemoteOriginUnauthorizedException(Exception):
    """
    Exception raised when there is no authorization for actions on remote image repository
    """
    def __init__(self, URL):
        self.URL = URL
        self.message = f'Error in authorization with remote image repository {URL}.'
        super.__init__(self.message)


class NotImplementedError(Exception):
    """
    Error raised from methods that have not been implemented
    """
    def __init__(self):
        self.message = "This feature has not been implemented" 
        super.__init__(self.message)

    
    

class Window(QMainWindow):
    def __init__(self):
        super(Window, self).__init__()
        self.load_UI()        
        self.imgs = []
        self.subj = []
        self.make_connect()
        self.inputExists = False
        self.defaultDirectoryExists = False
        self.isSegmented = False
        self.files = None
        self.temperaturesWereAcquired = False
        self.scaleModeAuto = True
        self.modelsPathExists = False
        self.model = 'default_model.tflite'
        self.fullScreen = True
        #Loading segmentation models
        self.s2s = SessionToSegment()
        self.i2s = ImageToSegment()
        self.s2s.setModel(self.model)
        self.i2s.setModel(self.model)
        self.s2s.loadModel()
        self.i2s.loadModel()
        self.ui_window.loadedModelLabel.setText(self.model)
        self.camera_index = 0
        self.setup_camera()
        self.sessionIsCreated = False
        self.driveURL = None
        self.rcloneIsConfigured = False
        self.repoUrl = 'https://github.com/blotero/FEET-GUI.git' 
        self.digits_model = tflite.Interpreter(model_path = './digits_recognition.tflite')
        self.digits_model.allocate_tensors()
        self.reader = easyocr.Reader(['en'])
        
    def predict_number(self,image):
        """
        Predicts digit value from a certain region image
        """

        image_2 = cv2.resize(image, (28, 28), interpolation = cv2.INTER_NEAREST)
        
        image_2 = cv2.cvtColor(np.uint8(image_2), cv2.COLOR_BGR2GRAY)
        
        image_2 = np.expand_dims(image_2, -1)

        image_2 = np.expand_dims(image_2, 0)
        input_details = self.digits_model.get_input_details()
        output_details = self.digits_model.get_output_details()

        input_shape = input_details[0]['shape']
        input_data = np.float32(image_2)

        self.digits_model.set_tensor(input_details[0]['index'], input_data)

        self.digits_model.invoke()  # predict

        output_data = self.digits_model.get_tensor(output_details[0]['index'])

        return np.argmax(output_data)  
        
     
    def extract_scales_2(self,x):
        
        x = np.uint8(x[:,560:,:])
        
        result = self.reader.readtext(x,detail=0)
        print(result)    
        result = [float(number) for number in result]
        
        lower = min(result)
        upper = max(result)
        
        return lower, upper
     
     
    def extract_scales(self, x):
        """
        Extracts float lower and upper scales from a thermal image
        """
        lower_digit_1 = self.predict_number(x[445: 467, 575: 591])
        lower_digit_2 = self.predict_number(x[445: 467, 589: 605])
        lower_digit_3 = self.predict_number(x[445: 467, 609: 625])
        
        upper_digit_1 = self.predict_number(x[14: 34, 576: 590])
        upper_digit_2 = self.predict_number(x[14: 34, 590: 604])
        upper_digit_3 = self.predict_number(x[14: 34, 610: 624])

        lower_bound = lower_digit_1*10 + lower_digit_2 + lower_digit_3*0.1
        upper_bound = upper_digit_1*10 + upper_digit_2 + upper_digit_3*0.1

        return lower_bound, upper_bound

    def extract_multiple_scales(self, X):
        """
        Extracts scales from a whole imported session
        """
        scales = []
        for i in range(X.shape[0]):
            scales.append(self.extract_scales_2(X[i]))
            
        return scales

    def setup_camera(self):
        """
        Initialize camera.
        """
        self.capture = cv2.VideoCapture(self.camera_index)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.timer = QTimer()
        self.timer.timeout.connect(self.display_frame)
        self.timer.start(30)


    def display_frame(self):
        """
        Refresh frame from camera
        """
        try:
            self.ret, self.frame = self.capture.read()
            self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            # image = qimage2ndarray.array2qimage(self.frame)
            self.image = QImage(self.frame, self.frame.shape[1], self.frame.shape[0], 
                        self.frame.strides[0], QImage.Format_RGB888)
            self.ui_window.inputImg.setPixmap(QPixmap.fromImage(self.image))
        except:
            time.sleep(1)
            self.message_print(f'No se detectó cámara {self.camera_index}. Reintentando...')
            print(f'Camera was not detected on index {self.camera_index}')
            if self.camera_index < 5:
                self.camera_index += 1
                print(f'Retrying with index {self.camera_index}...')
            else:
                self.message_print("Error detectando cámara. Por favor revisar conexión.")
                self.timer.stop()
                pass

    def load_UI(self):
        """
        Load xml file with visual objects for the interface
        """
        loader = QUiLoader()        
        path = os.fspath(Path(__file__).resolve().parent / "form.ui")
        ui_file = QFile(path)
        ui_file.open(QFile.ReadOnly)
        self.ui_window = loader.load(ui_file, self)
        self.ui_window.showFullScreen()
        ui_file.close()
    
    def capture_image(self):
        """
        Captures a new image. Creates a new session with current timestamp if a session had
        not been created previously
        """
        if (not self.sessionIsCreated):
            self.message_print("No se ha creado una sesion. Creando nueva...")
            time.sleep(1)
            self.create_session()
        
        if len(os.listdir(self.session_dir)) <= 1:
            image_number = len(os.listdir(self.session_dir))
        else:
            image_number = 5*len(os.listdir(self.session_dir)) - 5
        
        self.save_name = f't{image_number}.jpg'
        plt.imsave(os.path.join(self.session_dir, self.save_name), self.frame)
        self.ui_window.outputImg.setPixmap(QPixmap.fromImage(self.image))
        self.ui_window.imgName.setText(self.save_name[:-4])
        
        if self.ui_window.autoScaleCheckBox.isChecked():
            # Read and set the temperature range:
            temp_scale = self.extract_scales_2(self.frame)
            self.ui_window.minSpinBox.setValue(temp_scale[0])
            self.ui_window.maxSpinBox.setValue(temp_scale[1])
            self.message_print(f"Escala leida: {temp_scale}. Por favor verifique que sea la correcta y corrijala en caso de que no lo sea.")

    def create_session(self):
        """
        Creates a new session, including a directory in ./outputs/<session_dir> with given input parameters
        from GUI
        The session is named as the current timestamp if current session_dir is null
        """
        self.name = self.ui_window.nameField.text()
        self.dir_name = self.name.replace(' ','_')
        if self.dir_name == '':
            today = datetime.today()
            self.dir_name = today.strftime("%Y-%m-%d_%H:%M")            
        try:
            self.session_dir = os.path.join('outputs',self.dir_name)
            os.mkdir(self.session_dir)
            self.sessionIsCreated = True
            self.message_print("Sesión " + self.session_dir + " creada exitosamente." )
        except:
            self.message_print("Fallo al crear la sesión. Lea el manual de ayuda para encontrar solución, o reporte bugs al " + self.bugsURL)
        
    def sync_local_info_to_drive(self):
        """
        Syncs info from the output directory to the configured sync path
        """
        self.message_print("Sincronizando información al repositorio remoto...")
        try:
            status = os.system("rclone copy outputs drive:")
            self.message_print("Sincronizando información al repositorio remoto...")
            if self.rcloneIsConfigured:
                if status == 0:
                    self.message_print("Se ha sincronizado exitosamente la información")
                    return
                raise Exception("Error sincronizando imagenes al repositorio remoto")
            raise RemoteOriginUnauthorizedException(self.driveURL)
        except RemoteOriginUnauthorizedException as ue:
            self.message_print("Error de autorización durante la sincronización. Dirígase a Ayuda > Acerca de para más información.")
            print(ue)
        except Exception as e:
            self.message_print("Error al sincronizar la información al repositorio. Verifique que ha seguido los pasos de instalación y configuración de rclone. Para más información, dirígase a Ayuda > Acerca de.")
            print(e)

    def repo_config_dialog(self):
        """
        Shows a dialog window for first time configuring the remote repository sync for the current device
        """
        raise NotImplementedError()

    def make_connect(self):
        """
        Makes all connections between singleton methods and objects in UI xml
        """
        QObject.connect(self.ui_window.actionCargar_imagen, SIGNAL ('triggered()'), self.open_image)
        QObject.connect(self.ui_window.actionCargar_carpeta , SIGNAL ('triggered()'), self.open_folder)
        QObject.connect(self.ui_window.actionCargar_modelos , SIGNAL ('triggered()'), self.get_models_path)
        QObject.connect(self.ui_window.actionPantalla_completa , SIGNAL ('triggered()'), self.toggle_fullscreen)
        QObject.connect(self.ui_window.actionSalir , SIGNAL ('triggered()'), self.exit_)
        QObject.connect(self.ui_window.actionC_mo_usar , SIGNAL ('triggered()'), self.display_how_to_use)
        QObject.connect(self.ui_window.actionUpdate , SIGNAL ('triggered()'), self.update_software)
        QObject.connect(self.ui_window.actionRepoSync , SIGNAL ('triggered()'), self.sync_local_info_to_drive)
        QObject.connect(self.ui_window.actionRepoConfig , SIGNAL ('triggered()'), self.repo_config_dialog)
        QObject.connect(self.ui_window.segButtonImport, SIGNAL ('clicked()'), self.segment)
        #QObject.connect(self.ui_window.tempButton, SIGNAL ('clicked()'), self.temp_extract)
        QObject.connect(self.ui_window.tempButtonImport, SIGNAL ('clicked()'), self.temp_extract)
        QObject.connect(self.ui_window.captureButton, SIGNAL ('clicked()'), self.capture_image)
        QObject.connect(self.ui_window.nextImageButton , SIGNAL ('clicked()'), self.next_image)
        QObject.connect(self.ui_window.previousImageButton , SIGNAL ('clicked()'), self.previous_image)
        QObject.connect(self.ui_window.reportButton , SIGNAL ('clicked()'), self.export_report)
        QObject.connect(self.ui_window.loadModelButton , SIGNAL ('clicked()'), self.toggle_model)
        QObject.connect(self.ui_window.createSession, SIGNAL ('clicked()'), self.create_session)
        QObject.connect(self.ui_window.segButton, SIGNAL ('clicked()'), self.segment_capture)
    
    def segment_capture(self):
        """
        Segment newly acquired capture with current loaded segmentation model
        """
        self.message_print("Segmentando imagen...")
        self.i2s.setModel(self.model)
        self.i2s.setPath(os.path.join(self.session_dir,self.save_name))
        self.i2s.loadModel()
        self.i2s.extract()
        threshold =  0.5
        img = plt.imread(os.path.join(self.session_dir, self.save_name))/255
        Y = self.i2s.Y_pred
        Y = Y / Y.max()
        Y = np.where( Y >= threshold  , 1 , 0)
        self.Y =remove_small_objects( Y[0])     #Eventually required by temp_extract
        Y = cv2.resize(Y[0], (img.shape[1],img.shape[0]), interpolation = cv2.INTER_NEAREST) # Resize the prediction to have the same dimensions as the input 
        if self.ui_window.rainbowCheckBoxImport.isChecked():
            cmap = 'rainbow'
        else:
            cmap = 'gray'
        # plt.figure()
        # plt.plot(Y*img[:,:,0])
        # plt.savefig("outputs/output.jpg")
        plt.imsave("outputs/output.jpg" , Y*img[:,:,0] , cmap=cmap)
        self.ui_window.outputImg.setPixmap("outputs/output.jpg")
        self.isSegmented = True
        self.message_print("Imagen segmentada exitosamente")

    def set_default_config_settings(self, model_dir, session_dir):
        """
        Sets default config settings
        """
        self.config = {'models_directory': model_dir,
                'session_directory': session_dir }

    def update_user_configuration(self):
        """
        Updates basic configuration
        """
        self.modelsPath = self.config['models_directory']
        self.defaultDirectory = self.config['session_directory']

    def message_print(self, message):
        """
        Prints on interface console
        """
        log_path = "outputs/logs.html"
        out_file = open(log_path , "w")
        out_file.write(message)
        out_file.close()
        self.ui_window.textBrowser.setSource(log_path)
        self.ui_window.textBrowser.reload()

    def find_images(self):
        """
        Finds image from the path established in self.defaultDirectory obtained 
        from the method self.open_folder
        """
        self.fileList = []  #Absolute paths
        self.files = []     #Relative paths
        self.outfiles=[]    #Relative path to output files
        for root, dirs, files in os.walk(self.defaultDirectory):
            for file in files:
                if (file.endswith(".jpg")):
                    self.fileList.append(os.path.join(root,file))
                    self.files.append(file) 
                    self.outfiles.append("outputs/" + file) #Creating future output file names
        self.imageQuantity = len(self.fileList)
        self.imageIndex = 0
        self.sort_files()
        self.ui_window.inputLabel.setText(self.files[self.imageIndex])

    def sort_files(self):
        """
        Sort file list to an alphanumeric reasonable sense
        """         
        convert = lambda text: int(text) if text.isdigit() else text 
        alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
        self.fileList =  sorted(self.fileList, key = alphanum_key)
        self.files =  sorted(self.files, key = alphanum_key)

    def get_times(self):
        """
        Converts standarized names of file list into a list of 
        integers with time capture in minutes from the acquired self.fileList 
        from self.find_images
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

    def next_image(self):
        """
        Displays next image from self.fileList
        """
        if self.imageIndex < len(self.fileList)-1:
            self.imageIndex += 1
            self.ui_window.inputImgImport.setPixmap(self.fileList[self.imageIndex])
            self.opdir = self.fileList[self.imageIndex]
            self.ui_window.inputLabel.setText(self.files[self.imageIndex])

            if self.sessionIsSegmented:
                #Sentences to display next output image if session was already
                #segmented
                self.show_output_image_from_session()
                if self.temperaturesWereAcquired:
                    self.message_print("La temperatura media es: " + str(self.meanTemperatures[self.imageIndex]))
                    self.ui_window.temperatureLabelImport.setText(str(np.round(self.meanTemperatures[self.imageIndex], 3)))
                
    def previous_image(self):
        """
        Displays previous image from self.fileList
        """
        if self.imageIndex >= 1:
            self.imageIndex -= 1
            self.ui_window.inputImgImport.setPixmap(self.fileList[self.imageIndex])
            self.opdir = self.fileList[self.imageIndex]
            self.ui_window.inputLabel.setText(self.files[self.imageIndex])

            if self.sessionIsSegmented:
                #Sentences to display next output image if session was already
                #segmented
                self.show_output_image_from_session()
                if self.temperaturesWereAcquired:
                    self.message_print("La temperatura media es: " + str(self.meanTemperatures[self.imageIndex]))
                    self.ui_window.temperatureLabelImport.setText(str(np.round(self.meanTemperatures[self.imageIndex], 3)))

    def save_image(self):
        """
        Saves segmented image
        """
        raise NotImplementedError()

    def get_models_path(self):
        """
        Display a file manager dialog for selecting model list root directory
        """
        self.modelDialog=QFileDialog(self)
        self.modelDialog.setDirectory(QDir.currentPath())        
        self.modelDialog.setFileMode(QFileDialog.FileMode.Directory)
        self.modelsPath = self.modelDialog.getExistingDirectory()
        if self.modelsPath:
            self.modelsPathExists = True
            self.modelList = []
            for root, dirs, files in os.walk(self.modelsPath):
                for file in files:
                    self.modelList.append(os.path.join(root,file))
            self.modelQuantity = len(self.modelList)
            self.modelIndex = 0
            self.models = files
            self.ui_window.modelComboBox.addItems(self.models)


    def feet_segment(self):
        """
        Segments a single feet image
        """
        self.message_print("Segmentando imagen...")
        self.i2s.setModel(self.model)
        self.i2s.setPath(self.opdir)
        self.i2s.extract()
        self.show_segmented_image()
        self.isSegmented = True
        self.message_print("Imagen segmentada exitosamente")

    def session_segment(self):
        """
        Segments a whole feet session
        """
        self.message_print("Segmentando toda la sesion...")
        self.sessionIsSegmented = False
        self.s2s.setModel(self.model)
        self.s2s.setPath(self.defaultDirectory)
        self.s2s.whole_extract(self.fileList)
        self.produce_segmented_session_output()
        self.show_output_image_from_session()
        self.message_print("Se ha segmentado exitosamente la sesion con "+ self.i2s.model)
        self.sessionIsSegmented = True

    def show_segmented_image(self):
        """
        Shows segmented image
        """
        #Applies segmented zone to input image, showing only feet
        threshold =  0.5
        img = plt.imread(self.opdir)/255
        Y = self.i2s.Y_pred
        Y = Y / Y.max()
        Y = np.where( Y >= threshold  , 1 , 0)
        self.Y =remove_small_objects( Y[0])     #Eventually required by temp_extract
        Y = cv2.resize(Y[0], (img.shape[1],img.shape[0]), interpolation = cv2.INTER_NEAREST) # Resize the prediction to have the same dimensions as the input 
        if self.ui_window.rainbowCheckBoxImport.isChecked():
            cmap = 'rainbow'
        else:
            cmap = 'gray'
        plt.figure()
        plt.plot(Y*img[:,:,0])
        plt.savefig("outputs/output.jpg")
        #plt.imsave("outputs/output.jpg" , Y*img[:,:,0] , cmap=cmap)
        self.ui_window.outputImgImport.setPixmap("outputs/output.jpg")
    
    def produce_segmented_session_output(self):
        """
        Produce output images from a whole session and         """
        #Recursively applies show_segmented_image to whole session
        self.Y=[]
        for i in range(len(self.outfiles)):
            threshold =  0.5
            img = plt.imread(self.fileList[i])/255
            Y = self.s2s.Y_pred[i]
            Y = Y / Y.max()
            Y = np.where( Y >= threshold  , 1 , 0)

            self.Y.append(remove_small_objects(Y[0]))     #Eventually required by temp_extract
            print(self.Y[0].shape)
            Y = cv2.resize(Y[0], (img.shape[1],img.shape[0]), interpolation = cv2.INTER_NEAREST) # Resize the prediction to have the same dimensions as the input 
            if self.ui_window.rainbowCheckBox.isChecked():
                cmap = 'rainbow'
            else:
                cmap = 'gray'
            # plt.figure()
            # plt.imshow(Y*img[:,:,0])
            # plt.axis('off')
            # plt.savefig(self.outfiles[i])
            plt.imsave(self.outfiles[i], Y*img[:,:,0] , cmap=cmap)


    def show_output_image_from_session(self):
        """
        Display segmented image from current one selected from the index 
        established by self.previous_image or self.next_image methods
        """
        self.ui_window.outputImgImport.setPixmap(self.outfiles[self.imageIndex])

    def segment(self):
        """
        Makes segmentation action depending on the current state (single image or whole session)
        """
        if self.ui_window.sessionCheckBox.isChecked():
            if self.defaultDirectoryExists and self.i2s.model!=None and self.s2s.model!=None:
                self.session_segment()
            else:
                self.message_print("Error. Por favor verifique que se ha cargado el modelo y la sesion de entrada.")
        else:
            if self.inputExists and self.modelsPathExists and self.model!=None:
                self.feet_segment()
            else:
                self.message_print("No se ha seleccionado sesion de entrada")

    def manual_segment(self):
        """
        Display manual segmentation utility
        """
        #print("Se abrirá diálogo de extracción manual")
        #self.manual=manualSeg()
        #self.manual.show()
        raise NotImplementedError()

    def temp_extract(self):
        """
        Extract temperatures from a segmented image or a whole session
        """
        if (self.inputExists and (self.isSegmented or self.sessionIsSegmented)):
            if self.ui_window.autoScaleCheckBoxImport.isChecked and self.ui_window.sessionCheckBox.isChecked():
                #Get automatic scales
                scale_range = self.extract_multiple_scales(self.s2s.img_array)
                print(scale_range)
                
            elif not self.ui_window.autoScaleCheckBoxImport.isChecked():
                scale_range = [self.ui_window.minSpinBoxImport.value() , self.ui_window.maxSpinBoxImport.value()] 

            if self.ui_window.sessionCheckBox.isChecked():   #If segmentation was for full session
                self.meanTemperatures = []   #Whole feet mean temperature for all images in session
                if self.ui_window.autoScaleCheckBoxImport.isChecked():
                    for i in range(len(self.outfiles)):
                        self.meanTemperatures.append(mean_temperature(self.s2s.Xarray[i,:,:,0] , self.Y[i][:,:,0] , scale_range[i], plot = False))
                else:
                    for i in range(len(self.outfiles)):
                        self.meanTemperatures.append(mean_temperature(self.s2s.Xarray[i,:,:,0] , self.Y[i][:,:,0] , scale_range, plot = False))
                self.message_print("La temperatura media es: " + str(self.meanTemperatures[self.imageIndex]))
                self.message_print(f"La escala leida es: {scale_range[self.imageIndex]}")
                self.ui_window.minSpinBoxImport.setValue(scale_range[self.imageIndex][0])
                self.ui_window.maxSpinBoxImport.setValue(scale_range[self.imageIndex][1])
                self.temperaturesWereAcquired = True
            else:      #If segmentation was for single image
                if ui_window.autoScaleCheckBoxImport.isChecked():
                    scale_range = self.extract_scales_2(self.i2s.Xarray)
                else:
                    scale_range = [self.ui_window.minSpinBoxImport.value() , self.ui_window.maxSpinBoxImport.value()]
                mean = mean_temperature(self.i2s.Xarray[:,:,0] , self.Y[:,:,0] , scale_range, plot = False)
                self.message_print("La temperatura media es: " + str(mean))

            if (self.ui_window.plotCheckBox.isChecked()):  #If user asked for plot
                self.message_print("Se generara plot de temperatura...")
                self.get_times()
                print(self.timeList)
                self.temp_plot()

        elif self.inputExists:
            self.message_print("No se ha segmentado previamente la imagen ")
        else:
            self.message_print("No se han seleccionado imagenes de entrada")

    def toggle_model(self):
        """
        Change model loaded if user changes the model modelComboBox
        """
        self.modelIndex = self.ui_window.modelComboBox.currentIndex()
        self.message_print("Cargando modelo: " + self.models[self.modelIndex]
                        +" Esto puede tomar unos momentos...")
        try:
            self.model = self.modelList[self.modelIndex]
            self.s2s.setModel(self.model)
            self.i2s.setModel(self.model)
            self.s2s.loadModel()
            self.i2s.loadModel()
            self.ui_window.loadedModelLabel.setText(self.model)
            self.message_print("Modelo " + self.models[self.modelIndex] + " cargado exitosamente")
        except:
            self.message_print("Error al cargar el modelo "+ self.models[self.modelIndex])

    def temp_plot(self):
        """
        Plots from acquired temperature samples from a session
        """
        plt.figure()
        x = np.linspace(min(self.timeList), max(self.timeList), 200)
        spl = make_interp_spline(self.timeList, self.meanTemperatures, k=3)
        y = spl(x) 
        plt.plot(x , y, '-.', color='salmon')
        plt.plot(self.timeList , self.meanTemperatures , '-o', color='slategrey')
        plt.title("Temperatura media de pies")
        plt.xlabel("Tiempo (min)")
        plt.ylabel("Temperatura (°C)")
        plt.grid()
        plt.show()
        self.message_print("Plot de temperatura generado exitosamente")
        #Produce plot 

    def open_image(self):
        """
        Displays a dialog for loading a single image
        """
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
            self.ui_window.inputImgImport.setPixmap(self.opdir)

    def open_folder(self):
        """
        Displays a dialog for loading a whole session
        """
        self.folderDialog=QFileDialog(self)
        self.folderDialog.setDirectory(QDir.currentPath())        
        self.folderDialog.setFileMode(QFileDialog.FileMode.Directory)
        self.defaultDirectory = self.folderDialog.getExistingDirectory()
        self.imagesDir = self.defaultDirectory
        if self.defaultDirectory:
            self.defaultDirectoryExists = True
            first_image = str(self.defaultDirectory + "/t0.jpg")
            print(first_image)
            self.ui_window.inputImgImport.setPixmap(first_image)
            self.opdir = first_image
            self.inputExists = True
            self.find_images()
            self.sessionIsSegmented = False

    def toggle_fullscreen(self):
        """
        Toggles fullscreen state
        """
        if self.fullScreen:
            self.ui_window.showNormal()
            self.fullScreen = False
        else:
            self.ui_window.showFullScreen()
            self.fullScreen = True

    def exit_(self):
        sys.exit(app.exec_())

    def display_how_to_use(self):
        """
        Displays user manual on system's default pdf viewer
        """
        os.system("xdg-open README.html")

    def export_report(self):
        """
        GENERATE A PDF REPORT FOR THE PATIENT
        INPUT: SELF, PATIENT DIR
        RETURN: NONE
        ACTION: COMPILE PDF TEXT BASED ON
        """
        self.message_print("Generando reporte...") 
        self.message_print("Desarrollo no implementado, disponible en futuras versiones.")
        raise NotImplementedError()

    def animate(self):      
        """
        Produces gif animation based on mean temperatures for whole session
        Initially, all feet has same color, for section segmentation has been not implemented yet
        """
        self.message_print("Iniciando animacion...")
        self.message_print("Desarrollo no implementado, disponible en futuras versiones.")
        raise NotImplementedError()

    def update_software(self):
        """
        Updates software from remote origin repository
        """
        try:
            self.message_print(f"Actualizando software desde {self.repoUrl}...")
            time.sleep(2)
            exit_value = os.system("git pull")
            if exit_value == 0:
                self.message_print("Se ha actualizado exitosamente la interfaz. Se sugiere reiniciar interfaz")
                return
            self.message_print("Error al actualizar.")
            raise RemotePullException(self.repoUrl)
        except:
            self.message_print("Error al actualizar.")
            raise RemotePullException(self.repoUrl)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    window.ui_window.show()  
    sys.exit(app.exec_())
