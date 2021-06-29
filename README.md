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

## 2.1 Native

First, choose the desired fork. If you are running ARM architectures (like raspberry pi), it is suggested to clone the ```no-tf``` branch, so you can avoid installing heavy libraries like ```tensorflow``` 


If you are using amd64 or x86 architecture, then start by simply cloning the repository into your desired directory. Type in terminal:

```
git clone https://github.com/blotero/FEET-GUI
cd FEET-GUI
```
Otherwise, if you preffer the ligth version, then clone the ```no-tf``` *(no tensorflow)* repository:

```
git clone https://github.com/blotero/FEET-GUI/tree/no-tf
cd FEET-GUI
```
Install requirements (make sure you have latest version of ```pip```).

```
pip install -r requirements.txt
```
Run in arch based distros:
```
python main.py
```
Run in other OS:
```
python3 main.py
```

## 2.2 Virtual environment

One of the requirements for ```feet-gui``` is a module known as ```opencv-contrib-python-headless``` which has conflict with the traditional  ```opencv-python``` package.
The reason for using the contrib-headless version is due to conflicts with the original version with the PySide package, for both (opencv and PySide) packages, work on instantes with qt that are not compatible with each other.

For this reason, it is suggested to run feet-gui on a virtual environment (specially if you have installed other packages related to opencv or qt).
This can be done in Linux as follows:

```
python -m venv feet-gui
source feet-gui/bin/activate
```

The resulting terminal-emulator state must look some like this:

![image](https://user-images.githubusercontent.com/43280129/123793715-a556cf80-d8a7-11eb-961d-66229c20d986.png)


Once you have done this, your shell should be working in this venv. If you had not previously cloned the repository, it is a good moment to do so, by following same steps as 2.1, and then installing the requirements with no probabilities of dependencies issues.

# 3. How to use

FEET-GUI is thought to be lightweight and minimal, yet intuitive for obstetrics healthcare environments.
Out of the box you will have a main window, 

![image](https://user-images.githubusercontent.com/43280129/121790986-b1a21380-cbaa-11eb-8cba-317c951ff241.png)


## 3.1 Import a patient folder
First, stand in *Input* (entradas) tab, then, press ```Ctrl + O``` to open a directory to a whole patient session. This path will now be the new default directory for opening images, and for full session processing tools.


![dialog](https://user-images.githubusercontent.com/43280129/121792260-321b4100-cbb8-11eb-8ee5-b319fa4ae136.gif)



## 3.2 Import a single thermal image
Press ```Ctrl + I``` and select the image file you want to work with (format must be png or jpg). If you previously made the *Import folder* action, the File Dialog shown should be in your established path, otherwise, cloned directory will be shown by default.
*Note: by using this action, you can't use the tools for full session processing, like "full plot". This is only avaliable when your last opening action was opening a folder.*


### 3.3 Apply AI segmentation
Segmentation algorithm is based on U-Net neural network. All training process is avaliable in https://github.com/Rmejiaz/FEET_U-Net. 
By pressing *Segment* button or ```Ctrl + F``` to an already displaying image (loaded either with open folder or open image)), the AI model makes the predict segmentation.

![segment](https://user-images.githubusercontent.com/43280129/121792635-385fec00-cbbd-11eb-87f4-840644c1e1b6.gif)



In case AI does not offer desired results, user can choose the *manual segmentation*. By pressing this button, a new Dialog Window will be opened for the user to make the segmentation area in two stages. *(More on this in future versions)*



### 3.4 Import annotations

By launching File/Load Annotations , you can set the directory for annotations for the patient you previously loaded as a directory. This action can also be done with ```Ctrl  + Shift + O ```

![annotations](https://user-images.githubusercontent.com/43280129/122077272-349cb700-cdc1-11eb-9703-69c4450b5f66.gif)



### 3.5 Get temperature curves

Once you have already imported a directory for patient, you can get the temperature curves. You can press ```Ctrl + T``` for this action. Inmediately, a window will appear, showint the temperature curves based on the input images. It is important to have imported a directory, not an image.


![temp](https://user-images.githubusercontent.com/43280129/122077755-9f4df280-cdc1-11eb-9dbb-879b14504842.gif)



### 3.6 Make full plot!
By clicking the *patient full plot* button, a new dialog will open and show the entire sessions plot. This is only avaliable when a folder has been the last file opening action.



### 3.7 Save report

Once you have done all processing, you can save your report as a pdf file by clicking the button "Export report". This action will only make sense if you have imported  a whole directory with the correctly named files. If you execute this action without making all processing previously, it will be done automatically by default.

