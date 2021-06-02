# FEET-GUI

## Brief description

FEET-GUI is a python qt-5 based application, designed to work comfortably with thermal feet images, more specifically, for characterizing epidural anesthesia performance during birth.


## 1. System requirements
FEET-GUI is 200% cross plaftorm. It can be runned on any machine able to handle ```python3.9``` and ```qt5```. This includes Windows, MacOS,  GNU/Linux machines, and also ARM based OS's like Rasbperry PI OS, Manjaro ARM, etc.

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


If you are using amd64 architecture, then start by simply cloning the repository into your desired directory. Type in terminal:
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

STRONG FOCUS ON THIS!!!!!!!!!!!
```
python -m venv feet-gui
```

# 3. How to use

FEET-GUI is thought to be lightweight and minimal, yet intuitive for obstetrics healthcare environments.
Out of the box you will have a main window, 

![image](https://user-images.githubusercontent.com/43280129/120423770-b22be600-c330-11eb-8d28-e747cbc8d6b1.png)

## 3.1 Import a patient folder
First, stand in *Input* (entradas) tab, then, press ```Ctrl + O``` to open a directory to a whole patient session. This path will now be the new default directory for opening images.


## 3.2 Import a thermal image
Press ```Ctrl + I``` and select the .png file you want to work with. If you previously made the *Import folder* action, the File Dialog shown should be in your established path, otherwise, cloned directory will be shown by default.

### 3.3 Apply AI segmentation
Segmentation algorithm is based on U-Net neural network. All training process is avaliable in https://github.com/Rmejiaz/FEET_U-Net. 
By pressing *Segment* button to an already imported image, the AI model makes the two stages predict segmentation, first, predicting areas for wich label is a feet, and then, the interest regions acording to literature.

In case AI does not offer desired results, user can choose the *manual segmentation*. By pressing this button, a new Dialog Window will be opened for the user to make the segmentation area in two stage.
