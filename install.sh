#!/bin/sh
sudo pacman -S pyside2 python-pytesseract qt5 gcc make tesseract-data-eng

pip install --pre SimpleITK --find-links https://github.com/SimpleITK/SimpleITK/releases/tag/v2.1rc2
pip install tflite_runtime-2.9.0-cp310-cp310-linux_aarch64.whl
pip install -r requirements.txt


install_path=$HOME/.local/bin/feetgui
echo "Installing executable into " $install_path " ..."
touch $install_path
echo "#!/bin/sh" > $install_path
echo "cd " $path >> $install_path
echo "python main.py" >> $install_path
chmod +x $install_path
ln $install_path $HOME/Desktop/
ln $install_path $HOME/.local/bin/feetgui



