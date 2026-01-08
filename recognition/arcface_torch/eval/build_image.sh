# Since Docker does not allow to refer to ../ folders, copy those folders here
cp -r ../backbones ./
cp -r ../work_dirs ./
# Build an image
sudo docker build -t arcface-jamal .
