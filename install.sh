if (ls | grep insightface); then
  echo "insightface already installed"
else
  git clone https://github.com/deepinsight/insightface
fi

if (ls | grep ssh); then
  echo "enhanced-ssh-mxnet already installed"
else
  git clone https://github.com/deepinx/enhanced-ssh-mxnet
fi


pip install mxnet-cu101

cd insightface/RetinaFace
make


if (ls model | grep R50); then
  echo "model already downloaded"
else
  wget https://www.dropbox.com/s/53ftnlarhyrpkg2/retinaface-R50.zip
  mkdir -p model
  mv retinaface-R50.zip model
  cd model
  unzip retinaface-R50.zip
  cd ../../..
fi

mv enhanced-ssh-mxnet essh

mkdir -p videos images

# download video baby1