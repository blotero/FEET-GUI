# FEET-GUI

## Brief description

FEET-GUI is a python qt-5 based application, designed to work comfortably with thermal feet images, more specifically, for characterizing epidural anesthesia performance during birth.


## 1. System requirements
FEET-GUI is 200% cross plaftorm. It can be runned on any machine able to handle ```python3.9``` and ```qt5```. This includes Windows, MacOS,  GNU/Linux machines, BSD, and also ARM based OS's like Rasbperry PI OS, Manjaro ARM, etc.

Apps is strongly suggested **not to be runned in anaconda-like environments**, for those migth not have the latest version of libraries, cores and packages.

Suggested distros for running FEET-GUI in Raspberry PI 4 (top being the most suggested):

<ol>
  <li>Manjaro ARM KDE Plasma https://manjaro.org/downloads/arm/raspberry-pi-4/arm8-raspberry-pi-4-kde-plasma/</li>
  <li>Manjaro ARM XCFE https://manjaro.org/downloads/arm/raspberry-pi-4/arm8-raspberry-pi-4-xfce/</li>
  <li>ArchLinux ARM https://archlinuxarm.org/platforms/armv8/broadcom/raspberry-pi-4</li>
  <li> Raspberry PI OS (not sorted first for debian packages issues) https://www.raspberrypi.org/software/</li>
</ol>


## 2. Installation and running instructions


## 2.1 Install on Windows
For installing FEET-GUI on Windows you need to have python 3.9 installed in your system. The easy way to achieve this is simply typing ```python```in your PowerShell or CMD. If you don't have python installed, an instance of the official Microsoft Store will be opened to install the latest available python core.

Additionally, you might want to install git on your machine from https://git-scm.com/downloads. 
After this, you are ready to run the native installation simply by lauching the ```feet-gui.bat```file (you can skip native installation).

## 2.2 Native


If you are using amd64 or x86 architecture, then start by simply cloning the repository into your desired directory. Type in terminal (or PowerShell in Windows):

```
git clone https://github.com/blotero/FEET-GUI
cd FEET-GUI
```

Install requirements (make sure you have latest version of ```pip``` by running ```pip install pip```).

```
pip install -r requirements.txt
```
Finally, run in arch-linux based distributions and Windows:
```
python main.py
```
Run in Debian based distributions:
```
python3 main.py
```

## 2.3 Virtual environment

One of the requirements for ```feet-gui``` is a module known as ```opencv-contrib-python-headless``` which has conflict with the traditional  ```opencv-python``` package.
The reason for using the contrib-headless version is due to conflicts with the original version with the PySide package, for both (opencv and PySide2) packages, work on instantes with qt that are not compatible with each other.

For this reason, it is suggested to run feet-gui on a virtual environment (specially if you have installed other packages related to opencv or qt).
This can be done in Linux as follows:

```
python -m venv feet-gui
source feet-gui/bin/activate
```

The resulting terminal-emulator state must look some like this:

![image](https://user-images.githubusercontent.com/43280129/123793715-a556cf80-d8a7-11eb-961d-66229c20d986.png)


Once you have done this, your shell should be working in this venv. If you had not previously cloned the repository, it is a good moment to do so, by following same steps as 2.2, and then installing the requirements with no dependencies issues.

# 3. How to use

FEET-GUI is thought to be lightweight and minimal, yet intuitive for obstetrics healthcare environments.
Out of the box you will have a main window, 

![image](https://user-images.githubusercontent.com/43280129/130433628-d22f450a-4e74-41ba-b6e8-87b8f4e15ab0.png)

## 3.1 Import your models
Feet-GUI is based on computer vision algorithms using different CNN architectures, which have been previously trained in with several architectures. All training process is avaliable in https://github.com/Rmejiaz/FEET_U-Net. 

You can get some example models in  https://drive.google.com/file/d/1-KLtcwjRtwhcib5pHKFOETuMyRsnyJTN/view?usp=sharing. Unzip this file, and then load the directory from File>>Load models (or Ctrl + Shift + O). 

![image](https://user-images.githubusercontent.com/43280129/130434004-b6f9c374-9ffb-492f-a9be-9261b57544fd.png)

Once your models have been loaded, you should find them available in the right bottom side, where the *Segmentation model* combo box will display the found models. 
*Observation: the directory you pick for your models can only contain models, otherwise any other file will be read as a model and will not be valid as a tensorflow model.*

![image](https://user-images.githubusercontent.com/43280129/130434468-b9ac8233-b034-40e3-bb6a-8463a9b01159.png)

Finally, pick the model you want to test and press *Load model*, this might take a couple seconds depending on your hardware.

## 3.2 Import a patient folder
Now, keep in *Input* (entradas) tab, then, press ```Ctrl + O``` to open a directory to a whole patient session (*or File>>Open Folder*). This path will now be the new default directory for opening images, and for full session processing tools. Make sure that images in this directory are ```jpg``` format and named as ```t#.jpg``` where # is the amount of minutes elapsed to the capture, for example, ```t15.jp``` if 15minutes elapsed. Feet-GUI will use this to automatically set times later. 


![image](https://user-images.githubusercontent.com/43280129/130434757-81ca6ca3-74c6-4f58-bd48-8d5207db3589.png)
![image](https://user-images.githubusercontent.com/43280129/130434803-57a2fc27-717b-43e7-a6d7-cff4bcc994b0.png)

Once the direcotory is loaded, you will see a preview of the earliest in time found image (according to the naming rules).

![image](https://user-images.githubusercontent.com/43280129/130435148-ef8aadf8-aa36-4ecb-b232-694e8b845abb.png)



## 3.3 Import a single thermal image
If desired, you can import a single image instead of a whole session. Press ```Ctrl + I``` and select the image file you want to work with (format must be png or jpg). If you previously made the *Import folder* action, the File Dialog shown should be in your established path, otherwise, cloned directory will be shown by default.
*Note: by using this action, you can't use the tools for full session processing, like "full plot". This is only avaliable when your last opening action was opening a folder.*


### 3.4 Apply AI segmentation
By pressing *Segment* button or ```Ctrl + F``` to an already displaying image (loaded either with open folder or open image)), the AI model predicts segmentation area.

![image](https://user-images.githubusercontent.com/43280129/130435712-56671f89-83e7-4489-99f5-b69aa13a0e5a.png)

### 3.5 Set temperature scale
If you want to force scale values for highest and lowest intensities found in the image you can do so by changing the values in the *Temperature scale* frame and picking the custom mode, otherwise, these scale would be extracted automatically.

![image](https://user-images.githubusercontent.com/43280129/130436073-52e9b805-477c-487b-80d0-839da87e4316.png)

In case AI does not offer desired results, user can choose the *manual segmentation*. By pressing this button, a new Dialog Window will be opened for the user to make the segmentation area in two stages. *(More on this in future versions)*

### 3.6 Get mean temperatures and curves
If everything has been set up corectly, now you can get the mean temperatures and, if the Plot checkbox is set, you will also get the curves. You can press ```Ctrl + T``` for this action. If ```Plot``` is enabled, inmediately, a window will appear, showing the temperature curves based on the input images. It is important to have imported a directory, not an image.

![image](https://user-images.githubusercontent.com/43280129/130436802-b96aa011-7b01-48cb-8cfe-4b5315def4c1.png)


### 3.7 Make full plot!
By clicking the *patient full plot* button, a new dialog will open and show the entire sessions plot for all zones. This is only avaliable when a folder has been the last file opening action. *(More on this in future versions)*


### 3.8 Save report

Once you have done all processing, you can save your report as a pdf file by clicking the button "Export report". This action will only make sense if you have imported  a whole directory with the correctly named files. If you execute this action without making all processing previously, it will be done automatically by default. *(More on this in future versions)*

## 4. Design 

FEET-GUI is developed in such a way that it can work as a research tool or a live tool in healthcare conditions.

### 4.1 Preloaded images

As a research, diagnosis, or general interest tool for healthcare in obstetrics, FEET-GUI can work with preloaded images. With this purpose, the _local_ design flow is thought to work robustly with simple local images, manually loaded by the user.

![feet_gui](https://user-images.githubusercontent.com/43280129/123895317-3f109200-d925-11eb-868d-828ce68e41a0.png)

### 4.2 Real time hardware

FEET-GUI is suitabe for ARM for a reason, and that is to achieve real time use during a real birth giving session in which an additional tool for detecting analgesia effects, might be required. For this design, the user will no longer require to manually load the image files, but instead will simply shoot the IR image during the session with the extensile hardware avaliable for this task (WHICH???).

