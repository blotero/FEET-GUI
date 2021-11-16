#!/bin/sh

path=$(pwd)
install_path=~/.local/bin/feetgui

echo "Using path " $path " as source path"
echo "Installing executable into " $install_path " ..."
touch $install_path
echo "#!/bin/sh" > $install_path
echo "cd " $path >> $install_path
echo "python main.py" >> $install_path
ln $install_path ~/Desktop/
chmod +x ~/.local/bin/feetgui

echo "Succesfully installed FEET-GUI"
echo "Run it by simply typing feetgui in your terminal"
