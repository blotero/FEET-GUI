# GCPDS - Universidad Nacional de Colombia
# Proyecto caracterización termográfica de extremidades inferiores durante aplicación de anestesia epidural
# Mayo de 2021
# Disponible en https//:github.com/blotero/FEET-GUI

import os
import re
from pathlib import Path
import json
import sys
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
import pytesseract
from PySide2.QtWidgets import QApplication, QMainWindow, QFileDialog 
from PySide2.QtCore import QFile, QObject, SIGNAL, QDir, QTimer
from PySide2.QtUiTools import QUiLoader 
from segment import ImageToSegment, SessionToSegment
from manualseg import manualSeg
from temperatures import mean_temperature, dermatomes_temperatures
from scipy.interpolate import make_interp_spline 
import cv2
from PySide2.QtWidgets import *
from PySide2.QtCore import *
from PySide2.QtGui import *
from datetime import datetime
import tflite_runtime.interpreter as tflite
from postprocessing import PostProcessing
from report import plot_report
import threading
from PySide2.QtCore import QTimer


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

    

class Window:
    def __init__(self):
        super(Window, self).__init__()
        self.load_UI()        
        self.imgs = []
        self.subj = []
        self.make_connect()
        self.init_logs()
        self.inputExists = False
        self.defaultDirectoryExists = False
        self.isSegmented = False
        self.files = None
        self.temperaturesWereAcquired = False
        self.scaleModeAuto = True
        self.modelsPathExists = True   #As soon as the model is present in the expected path
        self.model = 'default_model.tflite'
        self.fullScreen = True
        #Loading segmentation models
        self.s2s = SessionToSegment()
        self.i2s = ImageToSegment()
        self.s2s.setModel(self.model)
        self.i2s.setModel(self.model)
        self.s2s.loadModel()
        self.i2s.loadModel()
        self.ui.loadedModelLabel.setText(self.model)
        self.camera_index = 0
        self.setup_camera()
        self.sessionIsCreated = False
        self.driveURL = None
        self.rcloneIsConfigured = False
        self.repoUrl = 'https://github.com/blotero/FEET-GUI.git' 
        self.digits_model = tflite.Interpreter(model_path = './digits_recognition.tflite')
        self.digits_model.allocate_tensors()
        self.set_default_input_cmap()
        self.file_system_model = QFileSystemModel()
        self.file_system_model.setRootPath(QDir.currentPath())
        self.ui.treeView.setModel(self.file_system_model)
        plt.style.use('bmh')
        self.session_info = {}
        self.ui.progressBar.setVisible(False)
        self.timer = QTimer()
        self.timer.timeout.connect(self.tick)
        
    def tick(self):
        if self.current_secs < 10:
            self.ui.lcdNumber.display(f'{self.current_mins}:0{self.current_secs}')
        else:
            self.ui.lcdNumber.display(f'{self.current_mins}:{self.current_secs}')
        self.current_secs += 1
        if self.current_secs%60 == 0:
            self.current_mins += 1
            self.current_secs = 0
        
    
    def load_UI(self):
        """
        Load xml file with visual objects for the interface
        """
        loader = QUiLoader()        
        path = os.fspath(Path(__file__).resolve().parent / "form.ui")
        ui_file = QFile(path)
        ui_file.open(QFile.ReadOnly)
        self.ui = loader.load(ui_file)
        self.ui.showFullScreen()
        ui_file.close()

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
        
    def predict_number_with_pytesseract(self, img):
        """
        Obtain number from section of an image
        """
        uint8img = img.astype("uint8")
        #print(np.unique(uint8img))
        #print(uint8img.shape)
        thresh = cv2.threshold(uint8img , 100, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
        #plt.figure()
        # plt.imshow(thresh)
        text = pytesseract.image_to_string(thresh,   config = '--psm 7')
        #Text cleaning and replacement...
        clean_text = text.replace('\n','').replace('-]', '4').replace(']', '1').replace(' ', '').replace(',', '.').replace('%', '7').replace('€','9').replace('[','').replace('&', '5').replace('-','3')
        # plt.title(clean_text)
        # plt.show()
        try:
            num = float(clean_text)
            if num>=100:
                num/=10
        except:
            print(f"Could not convert string {clean_text} into number")
            self.message_print(f"No se ha podido detectar escalas automáticamente de: Texto base: {text}, Texto limpio: {clean_text}. Dejando rango por defecto: [25, 45]")
            return -100
        return num


    def extract_scales_with_pytesseract(self,x):
        """
        Extracts float lower and upper scales from a thermal image with pytesseract
        """
        lower_seg = x[445: 467, 575: 625,0]
        upper_seg = x[14: 34, 576: 624,0]
        lower_prediction = self.predict_number_with_pytesseract(lower_seg)
        upper_prediction = self.predict_number_with_pytesseract(upper_seg)
        
        if lower_prediction == -100:
            lower_prediction = 25
        if upper_prediction == -100:
            upper_prediction = 45
        return lower_prediction, upper_prediction

     
    def extract_scales_2(self,x):
        """
        Exctract scales with easyocr (deprecated)
        """
        x = np.uint8(x[:,560:,:])
        #print(x) 
        result = self.reader.readtext(x,detail=0)
        #print(result)    
        try:
            result = [float(number) for number in result]
        except:
            self.message_print("No se ha podido detectar escalas automáticamente. Dejando rango por defecto: [25, 45]")
            return 25, 45
        
        lower = min(result)
        upper = max(result)
        self.message_print(f"Escala leida: {lower, upper}. Por favor verifique que sea la correcta y corríjala en caso de que no lo sea.")
        
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
            scales.append(self.extract_scales_with_pytesseract(X[i]))
            
        return scales


    def populate_session_info(self):
        """
        Fill dictionary attribute with the parms given by the info tab
        """
        #Initial information obtained during session creation
        self.session_info['Nombre'] = self.ui.nameField.text()
        self.session_info['Edad'] = self.ui.ageField.text()
        self.session_info['Tipo_de_documento'] = self.ui.weightField.value()           #Spinbox
        self.session_info['Nro_de_documento'] = self.ui.weightField.text()
        self.session_info['Semanas_de_gestacion'] = self.ui.weeksField.value()         #Spinbox
        self.session_info['Peso'] = self.ui.weightField.value()                        #Spinbox
        self.session_info['Estatura'] = self.ui.heightField.value()                    #Spinbox
        self.session_info['IMC'] = self.session_info['Peso']  / ( ( self.session_info['Estatura'] / 100 ) ** 2 ) #IMC=PESO/ESTATURA^2
        self.session_info['ASA'] = self.ui.ASAField.currentText()                      #Combobox
        self.session_info['Membranas'] = self.ui.membField.currentText()               #Combobox
        self.session_info['Dilatación'] = self.ui.dilatationField.value()              #Spinbox
        self.session_info['Paridad'] = self.ui.parityField.currentText()               #Combobox

        #Calculated additional information
        self.session_info['Temperaturas_medias'] = self.meanTemperatures
        self.session_info['Escalas_de_temperatura'] = self.scale_range
        self.session_info['Temperaturas_de_dermatomas'] = self.dermatomes_temps.tolist()

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
            self.ui.inputImg.setPixmap(QPixmap.fromImage(self.image))
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

    
    def capture_image(self):
        """
        Captures a new image. Creates a new session with current timestamp if a session had
        not been created previously
        """
        self.current_secs = 1
        self.current_mins = 0
        self.timer.start(1000)
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
        self.ui.outputImg.setPixmap(QPixmap.fromImage(self.image))
        self.ui.imgName.setText(self.save_name[:-4])
        this_image = f"{self.defaultDirectory}/t{image_number}.jpg"
        self.ui.inputImgImport.setPixmap(this_image)
        self.find_images()
        
        if self.ui.autoScaleCheckBox.isChecked():
            # Read and set the temperature range:
            temp_scale = self.extract_scales_with_pytesseract(self.frame)
            self.ui.minSpinBox.setValue(temp_scale[0])
            self.ui.maxSpinBox.setValue(temp_scale[1])

    def wipe_outputs(self, hard=False):
        self.message_print("Limpiando sesión...")
        self.imgs = []
        self.subj = []
        self.inputExists = False
        self.defaultDirectoryExists = False
        self.isSegmented = False
        self.files = None
        self.temperaturesWereAcquired = False
        self.s2s = SessionToSegment()
        self.i2s = ImageToSegment()
        self.s2s.setModel(self.model)
        self.i2s.setModel(self.model)
        self.s2s.loadModel()
        self.i2s.loadModel()
        self.sessionIsCreated = False
        self.ui.outputImgImport.setPixmap("")
        self.ui.inputImgImport.setPixmap("")
        self.ui.outputImg.setPixmap("")
        self.ui.temperatureLabelImport.setText("")

        if hard:
            self.ui.nameField.setText("")
            self.ui.ageField.setText("")
            self.ui.weightField.setValue(0)                 
            self.ui.weightField.setValue(0)
            self.ui.weeksField.setValue(0)                  
            self.ui.weightField.setValue(0)                 
            self.ui.heightField.setValue(0)                 
            self.ui.ASAField.setCurrentIndex(0)             
            self.ui.membField.setCurrentIndex(0)            
            self.ui.dilatationField.setValue(0)             
            self.ui.parityField.setCurrentIndex(0)          


    def create_session(self):
        """
        Creates a new session, including a directory in ./outputs/<session_dir> with given input parameters
        from GUI
        The session is named as the current timestamp if current session_dir is null
        """
        self.name = self.ui.nameField.text()
        formatted_today = datetime.today().strftime("%Y-%m-%d_%H:%M")            
        self.dir_name = f"{self.name.replace(' ','_')}{formatted_today}"
        try:
            self.wipe_outputs()
            self.session_dir = os.path.join('outputs',self.dir_name)
            os.mkdir(self.session_dir)
            self.sessionIsCreated = True
            self.message_print("Sesión " + self.session_dir + " creada exitosamente." )
            self.defaultDirectoryExists = True
            self.defaultDirectory = os.path.abspath(self.session_dir)
            self.inputExists = True
            self.sessionIsSegmented = False
            self.input_type = 2 #Video input capture
        except Exception as ex:
            self.message_print("Fallo al crear la sesión. Lea el manual de ayuda para encontrar solución, o reporte bugs al " + self.bugsURL)
            print(ex)
        
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
        QObject.connect(self.ui.actionCargar_imagen, SIGNAL ('triggered()'), self.open_image)
        QObject.connect(self.ui.actionCargar_carpeta , SIGNAL ('triggered()'), self.open_folder)
        QObject.connect(self.ui.actionCargar_modelos , SIGNAL ('triggered()'), self.get_models_path)
        QObject.connect(self.ui.actionPantalla_completa , SIGNAL ('triggered()'), self.toggle_fullscreen)
        QObject.connect(self.ui.actionSalir , SIGNAL ('triggered()'), self.exit_)
        QObject.connect(self.ui.actionC_mo_usar , SIGNAL ('triggered()'), self.display_how_to_use)
        QObject.connect(self.ui.actionUpdate , SIGNAL ('triggered()'), self.update_software)
        QObject.connect(self.ui.actionRepoSync , SIGNAL ('triggered()'), self.sync_local_info_to_drive)
        QObject.connect(self.ui.actionRepoConfig , SIGNAL ('triggered()'), self.repo_config_dialog)
        QObject.connect(self.ui.segButtonImport, SIGNAL ('clicked()'), self.segment)
        QObject.connect(self.ui.tempButtonImport, SIGNAL ('clicked()'), self.temp_extract)
        QObject.connect(self.ui.captureButton, SIGNAL ('clicked()'), self.capture_image)
        QObject.connect(self.ui.nextImageButton , SIGNAL ('clicked()'), self.next_image)
        QObject.connect(self.ui.previousImageButton , SIGNAL ('clicked()'), self.previous_image)
        #QObject.connect(self.ui.reportButton , SIGNAL ('clicked()'), self.export_report)
        QObject.connect(self.ui.reportButton , SIGNAL ('clicked()'), self.generate_full_session_plot)
        QObject.connect(self.ui.loadModelButton , SIGNAL ('clicked()'), self.toggle_model)
        QObject.connect(self.ui.createSession, SIGNAL ('clicked()'), self.create_session)
        QObject.connect(self.ui.segButton, SIGNAL ('clicked()'), self.segment_capture)
        #Comboboxes:
        self.ui.inputColormapComboBox.currentIndexChanged['QString'].connect(self.toggle_input_colormap)

    def toggle_input_colormap(self):
        self.input_cmap = self.accepted_cmaps[self.ui.inputColormapComboBox.currentIndex()]
        self.message_print(f"Se ha cambiado exitosamente el colormap de entrada a {self.input_cmap}")

    def set_default_input_cmap(self):
        self.accepted_cmaps = ['Gris', 'Hierro', 'Arcoiris', 'Lava']
        self.input_cmap = self.accepted_cmaps[0]
        self.ui.inputColormapComboBox.addItems(self.accepted_cmaps)


    def segment_capture(self):
        """
        Segment newly acquired capture with current loaded segmentation model
        """
        self.message_print("Segmentando imagen...")
        self.i2s.setModel(self.model)
        self.i2s.setPath(os.path.join(self.session_dir,self.save_name))
        self.i2s.loadModel()
        self.i2s.extract(cmap = self.input_cmap)
        threshold =  0.5   
        img = plt.imread(os.path.join(self.session_dir, self.save_name))/255
        Y = self.i2s.Y_pred
        Y = Y / Y.max()
        Y = np.where( Y >= threshold  , 1 , 0)
        post_processing = PostProcessing(self.ui.morphoSpinBox.value())
        u = post_processing.execute(Y[0])
        self.Y = u[0]     #Eventually required by temp_extract
        Y = np.copy(u)
        Y = cv2.resize(Y[0], (img.shape[1],img.shape[0]), interpolation = cv2.INTER_NEAREST) # Resize the prediction to have the same dimensions as the input 
        if self.ui.rainbowCheckBoxImport.isChecked():
            cmap = 'rainbow'
        else:
            cmap = 'gray'
        # plt.figure()
        # plt.plot(Y*img[:,:,0])
        # plt.savefig("outputs/output.jpg")
        plt.imsave("outputs/output.jpg" , Y*img[:,:,0] , cmap=cmap)
        self.ui.outputImg.setPixmap("outputs/output.jpg")
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

    def init_logs(self):
        log_path = "outputs/logs.html"
        open(log_path, 'w').close()
        out_file = open(log_path , "a")
        final_msg = f'<meta charset="UTF-8">\n'
        out_file.write(final_msg)
        out_file.close()

    def message_print(self, message):
        """
        Prints on interface console
        """
        log_path = "outputs/logs.html"
        out_file = open(log_path , "a")
        final_msg = f"\n <br> >>> </br>  {message}\n"
        out_file.write(final_msg)
        out_file.close()
        self.ui.textBrowser.setSource(log_path)
        self.ui.textBrowser.reload()
        self.ui.textBrowser.moveCursor(QTextCursor.End)


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
        self.ui.inputLabel.setText(self.files[self.imageIndex])

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
            self.ui.inputImgImport.setPixmap(self.fileList[self.imageIndex])
            self.opdir = self.fileList[self.imageIndex]
            self.ui.inputLabel.setText(self.files[self.imageIndex])

            if self.sessionIsSegmented:
                #Sentences to display next output image if session was already
                #segmented
                self.show_output_image_from_session()
                if self.temperaturesWereAcquired:
                    self.message_print(f"La temperatura media de pies es:  {np.round(self.meanTemperatures[self.imageIndex], 2)} para el tiempo:{self.files[self.imageIndex].replace('.jpg','')}")
                    rounded_temp = np.round(self.meanTemperatures[self.imageIndex], 2)
                    self.ui.temperatureLabelImport.setText(f'{rounded_temp} °C')
                    self.ui.minSpinBoxImport.setValue(self.scale_range[self.imageIndex][0])
                    self.ui.maxSpinBoxImport.setValue(self.scale_range[self.imageIndex][1])
                
    def previous_image(self):
        """
        Displays previous image from self.fileList
        """
        if self.imageIndex >= 1:
            self.imageIndex -= 1
            self.ui.inputImgImport.setPixmap(self.fileList[self.imageIndex])
            self.opdir = self.fileList[self.imageIndex]
            self.ui.inputLabel.setText(self.files[self.imageIndex])

            if self.sessionIsSegmented:
                #Sentences to display next output image if session was already
                #segmented
                self.show_output_image_from_session()
                if self.temperaturesWereAcquired:
                    self.message_print(f"La temperatura media de pies es:  {np.round(self.meanTemperatures[self.imageIndex], 2)} para el tiempo:{self.files[self.imageIndex].replace('.jpg','')}")
                    rounded_temp = np.round(self.meanTemperatures[self.imageIndex], 2)
                    self.ui.temperatureLabelImport.setText(f'{rounded_temp} °C')
                    self.ui.minSpinBoxImport.setValue(self.scale_range[self.imageIndex][0])
                    self.ui.maxSpinBoxImport.setValue(self.scale_range[self.imageIndex][1])

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
            self.ui.modelComboBox.addItems(self.models)


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
        self.ui.progressBar.setVisible(True)
        self.ui.progressBar.setFormat("Segmentando..%p%")
        self.ui.progressBar.setValue(0)
        time.sleep(0.5)
        self.sessionIsSegmented = False
        self.s2s.setModel(self.model)
        self.s2s.setPath(self.defaultDirectory)
        self.s2s.whole_extract(self.fileList, cmap = self.input_cmap, progressBar = self.ui.progressBar)
        self.produce_segmented_session_output()
        self.show_output_image_from_session()
        self.message_print("Se ha segmentado exitosamente la sesion con "+ self.i2s.model)
        self.sessionIsSegmented = True
        self.ui.progressBar.setValue(100)
        # time.sleep(0.5)
        self.ui.progressBar.setVisible(False)
        self.ui.progressBar.setFormat("%p%")
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
        self.Y =posprocessing( Y[0])[0]     #Eventually required by temp_extract
        Y = posprocessing(Y[0])
        Y = cv2.resize(Y[0], (img.shape[1],img.shape[0]), interpolation = cv2.INTER_NEAREST) # Resize the prediction to have the same dimensions as the input 
        if self.ui.rainbowCheckBoxImport.isChecked():
            cmap = 'rainbow'
        else:
            cmap = 'gray'
        plt.figure()
        plt.plot(Y*img[:,:,0])
        plt.savefig("outputs/output.jpg")
        plt.imsave("outputs/output.jpg" , Y*img[:,:,0] , cmap=cmap)
        self.ui.outputImgImport.setPixmap("outputs/output.jpg")
    
    def produce_segmented_session_output(self):
        """
        Produce output images from a whole session and         """
        #Recursively applies show_segmented_image to whole session
        self.Y=[]
        post_processing = PostProcessing(self.ui.morphoSpinBox.value())
        for i in range(len(self.outfiles)):
            threshold =  0.5
            img = plt.imread(self.fileList[i])/255
            Y = self.s2s.Y_pred[i]
            Y = Y / Y.max()
            Y = np.where( Y >= threshold  , 1 , 0)
            Y = post_processing.execute(Y[0])
            
            self.Y.append(Y)    #Eventually required by temp_extract
            
            #print(f"Dimensiones de la salida: {Y.shape}")
            Y = cv2.resize(Y, (img.shape[1],img.shape[0]), interpolation = cv2.INTER_NEAREST) # Resize the prediction to have the same dimensions as the input 
            
            if self.ui.rainbowCheckBox.isChecked():
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
        self.ui.outputImgImport.setPixmap(self.outfiles[self.imageIndex])

    def segment(self):
        """
        Makes segmentation action depending on the current state (single image or whole session)
        """
        if self.input_type >= 1:
            #Session
            if self.defaultDirectoryExists and self.i2s.model!=None and self.s2s.model!=None:
                self.message_print("Segmentando toda la sesión...")
                self.session_segment()
            else:
                self.message_print("Error. Por favor verifique que se ha cargado el modelo y la sesión de entrada.")
        elif self.input_type == 0:
            #Single image
            if self.inputExists and self.modelsPathExists and self.model!=None:
                self.feet_segment()
            else:
                self.message_print("No se ha seleccionado imagen de entrada")



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
        self.ui.progressBar.setVisible(True)
        self.ui.progressBar.setValue(0)
        self.message_print("Obteniendo temperaturas...")
        if (self.inputExists and (self.isSegmented or self.sessionIsSegmented)):
            self.message_print("Obteniendo temperaturas de la sesión...")
            self.ui.progressBar.setFormat("Extrayendo temperaturas... %p%")

            if self.ui.autoScaleCheckBoxImport.isChecked and self.input_type>=1:
                #Get automatic scales
                self.scale_range = self.extract_multiple_scales(self.s2s.img_array)
                
            elif not self.ui.autoScaleCheckBoxImport.isChecked():
                self.scale_range = [self.ui.minSpinBoxImport.value() , self.ui.maxSpinBoxImport.value()] 

            if self.input_type>=1:   #If segmentation was for full session
                self.meanTemperatures = []   #Whole feet mean temperature for all images in session
                segmented_temps = []
                original_temps = []
                dermatomes_temps = []
                dermatomes_masks = []
                if self.ui.autoScaleCheckBoxImport.isChecked():
                    for i in range(len(self.outfiles)):
                        mean_out, temp, original_temp = mean_temperature(self.s2s.Xarray[i,:,:,0] , self.Y[i][:,:,0] , self.scale_range[i], plot = False)
                        derm_temps, derm_mask = dermatomes_temperatures(original_temp, self.Y[i])
                        self.meanTemperatures.append(mean_out)
                        segmented_temps.append(temp)
                        original_temps.append(original_temp)
                        dermatomes_temps.append(derm_temps)
                        dermatomes_masks.append(derm_mask)
                        self.ui.progressBar.setValue((100*i+1)/len(self.outfiles))
                    self.dermatomes_temps = np.array(dermatomes_temps)
                    self.dermatomes_masks = np.array(dermatomes_masks)
                    self.segmented_temps = np.array(segmented_temps)
                    self.original_temps = np.array(original_temps)

                else:
                    for i in range(len(self.outfiles)):
                        mean_out, temp, original_temp = mean_temperature(self.s2s.Xarray[i,:,:,0] , self.Y[i][:,:,0] , self.scale_range, plot = False)
                        derm_temps, derm_mask = dermatomes_temperatures(original_temp, self.Y[i])
                        self.meanTemperatures.append(mean_out)
                        segmented_temps.append(temp)
                        original_temps.append(original_temp)
                        dermatomes_temps.append(derm_temps)
                        dermatomes_masks.append(derm_mask)
                        self.ui.progressBar.setValue((100*i+1)/len(self.outfiles))
                    self.dermatomes_temps = np.array(dermatomes_temps)
                    self.dermatomes_masks = np.array(dermatomes_masks)
                    self.segmented_temps = np.array(segmented_temps)
                    self.original_temps = np.array(original_temps)


                self.message_print("La temperatura media es: " + str(self.meanTemperatures[self.imageIndex]))
                self.message_print(f"La escala leida es: {self.scale_range[self.imageIndex]}")
                rounded_temp = np.round(self.meanTemperatures[self.imageIndex], 3)
                self.ui.temperatureLabelImport.setText(f'{rounded_temp} °C')
                self.ui.minSpinBoxImport.setValue(self.scale_range[self.imageIndex][0])
                self.ui.maxSpinBoxImport.setValue(self.scale_range[self.imageIndex][1])
                self.temperaturesWereAcquired = True
            else:      #If segmentation was for single image
                if self.ui.autoScaleCheckBoxImport.isChecked():
                    self.scale_range = self.extract_scales_with_pytesseract(self.i2s.img)
                else:
                    self.scale_range = [self.ui.minSpinBoxImport.value() , self.ui.maxSpinBoxImport.value()]
                time.sleep(1.5)
                mean, _ = mean_temperature(self.i2s.Xarray[:,:,0] , self.Y[:,:,0] , self.scale_range, plot = False)
                self.message_print("La temperatura media es: " + str(mean))
                rounded_temp = np.round(mean, 3)
                self.ui.temperatureLabelImport.setText(f'{rounded_temp} °C')

            if (self.ui.plotCheckBoxImport.isChecked() and self.input_type>=1):  #If user asked for plot
                #self.message_print("Se generara plot de temperatura...")
                self.get_times()
                # self.temp_plot()

        elif self.inputExists:
            #If input exists but session has not been segmented
            self.message_print("No se ha segmentado previamente la sesión. Segmentando... ")
            time.sleep(1)
            self.session_segment()
            self.temp_extract()
        elif self.ui.tabWidget.currentIndex() == 0:
            #Live video tab
            self.message_print("Obteniendo temperaturas para la última captura...")
            time.sleep(1)
            if not self.sessionIsCreated:
                self.message_print("No se ha creado una sesión de entrada. Presione capturar para crear una sesión por defecto o cree una con los parámetros deseados")
                return
            if len(os.listdir(self.session_dir)) < 1:
                self.message_print("No se ha hecho ninguna captura.")
                return
            #HERE COMES TO LOGIC FOR OBTAINING FULL PLOT FOR LIVE VIDEO

        else:
            self.message_print("No se han seleccionado imagenes de entrada")

        self.ui.progressBar.setVisible(False)
        self.ui.progressBar.setFormat("%p%")
    def toggle_model(self):
        """
        Change model loaded if user changes the model modelComboBox
        """
        self.modelIndex = self.ui.modelComboBox.currentIndex()
        self.message_print("Cargando modelo: " + self.models[self.modelIndex]
                        +" Esto puede tomar unos momentos...")
        try:
            self.model = self.modelList[self.modelIndex]
            self.s2s.setModel(self.model)
            self.i2s.setModel(self.model)
            self.s2s.loadModel()
            self.i2s.loadModel()
            self.ui.loadedModelLabel.setText(self.model)
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


    def generate_full_session_plot(self):
        if not self.temperaturesWereAcquired :
            self.message_print("No se han extraido las temperaturas, extrayendo...")
            self.temp_extract()
            self.generate_full_session_plot()
        else:
            exit_value = plot_report(img_temps = self.original_temps, segmented_temps = self.segmented_temps, mean_temps = self.meanTemperatures, times = self.timeList, 
                        path = os.path.join(self.defaultDirectory,'report'), dermatomes_temps = self.dermatomes_temps, dermatomes_masks = self.dermatomes_masks)
            if exit_value == 0:
                #Generación de información extra para la sesión
                self.message_print("Se ha generado exitosamente el plot completo de sesión")
            else:
                self.message_print("Advertencia, se ha encontrado un valor no válido (nan) en los dígitos de escala de temperatura. Verifique que la imagen es del formato y referencia de cámara correctos")
            self.populate_session_info()
            self.export_report()
        

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
            self.wipe_outputs(hard=True)
            self.input_type = 0
            self.inputExists = True
            self.ui.inputImgImport.setPixmap(self.opdir)
            self.message_print(f"Se ha importado exitosamente la imagen {self.opdir} ")
            self.ui.tabWidget.setProperty('currentIndex', 1)

    def open_folder(self):
        """
        Displays a dialog for loading a whole session
        """
        self.folderDialog=QFileDialog()
        self.folderDialog.setDirectory(QDir.currentPath())        
        self.folderDialog.setFileMode(QFileDialog.FileMode.Directory)
        self.defaultDirectory = self.folderDialog.getExistingDirectory()
        self.imagesDir = self.defaultDirectory
        if self.defaultDirectory:
            self.wipe_outputs(hard=True)
            self.input_type = 1
            self.defaultDirectoryExists = True
            first_image = str(self.defaultDirectory + "/t0.jpg")
            self.ui.inputImgImport.setPixmap(first_image)
            self.opdir = first_image
            self.inputExists = True
            self.find_images()
            self.sessionIsSegmented = False
            self.ui.tabWidget.setProperty('currentIndex', 1)
            self.message_print(f"Se ha importado exitosamente la sesión {self.defaultDirectory} ")
            #self.file_system_model.setRootPath(QDir(self.defaultDirectory))
            #self.ui.treeView.setModel(self.file_system_model)

    def toggle_fullscreen(self):
        """
        Toggles fullscreen state
        """
        if self.fullScreen:
            self.ui.showNormal()
            self.fullScreen = False
        else:
            self.ui.showFullScreen()
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
        Generates a json document with session information
        """
        with open(f"{self.defaultDirectory}/report.json", "w") as outfile:
            json.dump(self.session_info, outfile)

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
    window.ui.show()
    #window.ui.show()
    sys.exit(app.exec_())
