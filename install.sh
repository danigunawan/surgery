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

if [ $1 == 'gpu' ]; then
  pip install mxnet-cu101
elif [ $1 == 'cpu' ]; then
  pip install mxnet
fi

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
  cd ..
fi

cd ../..

if (ls | grep enhanced); then
  mv enhanced-ssh-mxnet essh
fi

mkdir -p videos images

cp detect_retina.py insightface/RetinaFace
cp common.py insightface/RetinaFace
cp detect_essh.py essh

if (ls insightface/RetinaFace | grep __init__); then
  echo "package already inited"
else
  touch insightface/RetinaFace/__init__.py
fi

# download video baby1