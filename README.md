
# Face Detection in Operating Room

### Install

1. Clone the repository
2. Run `./install.sh cpu` if you use cpu or `./install.sh gpu` if you use gpu

### Run

```
cd <dir>
python <filename> --vroot <path to videos> -s <save directory>
```

Here save directory is path where you want to place videos with detections. Filename is a name of python script and dir is name of repository with corresponding model. So there are two possible options:

1. `dir` = essh, `filename` = detect_essh.py
2. `dir` = insightface/RetinaFace, `filename` = detect_retina.py