#!/bin/sh

path=$(pwd)
install_path=/usr/local/bin/feetgui
PYENV_VERSION=3.9.0
python_env_path=$HOME/.pyenv/versions/$PYENV_VERSION/
python=$python_env_path/bin/python
pip=$python_bin -m

#Create python environment

pacman -S pyenv
echo "Succesfully installed pyenv"
pyenv install 3.9.0


echo "Using path " $path " as source path"
$pip install -r requirements.txt
echo "Installing executable into " $install_path " ..."
touch $install_path
echo "#!/bin/sh" > $install_path
echo "cd " $path >> $install_path
echo $python main.py >> $install_path
chmod +x $install_path
ln $install_path $HOME/Desktop/
ln $install_path $HOME/.local/bin/feetgui

echo "Succesfully installed FEET-GUI"
echo "Run it by simply typing feetgui in your terminal"
