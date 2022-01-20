#!/bin/sh

path=$(pwd)
install_path=/usr/bin/feetgui

echo "Using path " $path " as source path"
pip install -r requirements.txt
echo "Installing executable into " $install_path " ..."
touch $install_path
echo "#!/bin/sh" > $install_path
echo "cd " $path >> $install_path
echo "python main.py" >> $install_path
ln $install_path ~/Desktop/
ln $install_path /usr/local/bin
chmod +x $install_path

echo "Succesfully installed FEET-GUI"
echo "Run it by simply typing feetgui in your terminal"
