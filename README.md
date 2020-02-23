# Key Features Detection in Operating Room

Operating Room i.e. OR is a room where a surgery takes place. 

## Related work

#### Surgeon face tracking

Real-Time Visual Tracking of the Surgeon’s Face for Laparoscopic Surgery

Irise, marker

### 2018

#### Iterative self-supervised Approach for Face Detection in the OR

The idea is to fine-tune existing state of the art face detection models to the specific
problem of face detection in OR.

1. Generation of unlabeled dataset

20k images generated from videos captured in the OR. Test videos and train videos were collected on different days to avoid overlapping.
OpenPose model (a multi-person pose estimator) is used to get the approximate number of persons in each frame. Computations are rather efficient.
Images are divided into categories with respect to the number of people detected. The images with the highest confidence (provided with OpenPose) are selected.


2. Iterative refinement using a self-supervised approach





## Used technologies

### Face detection

####  Haar cascade

Introduced by Viola Jones. It is a classifier trained with given some input faces and non-faces. 
Basic idea is to extract features from an image via special filters similar to CNN kernels.
We traverse an image with a window of the fixed size and apply filters.


### Face recognition

The objective is to classify detected faces into doctor faces and patient faces. Doctor faces
have some comprehensive features such as mask and gown while the patient faces do not.
Тем не менее на пациенте может быть кислородная маска.

To solve the problem I suggest using face embeddings + deep neural networks.
That is how we are going to train our classifier i.e. recognition model. Siamese networks???