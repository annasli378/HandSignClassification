# Application with Pilish Sign Language Recognition

<!-- PROJECT LOGO -->
<br />
  <p align="left">
The goal of the project is to create a system with a classifier that enables recognition of the sign shown to the device's camera in applications that can then be used for sign language learning, verification and classification of the signs shown.
  </p>

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#dataset">MediaPipe for sign languages recognition</a></li>
        <li><a href="#models">Dataset</a></li>
        <li><a href="#saliency-methods">Data pre-processing</a></li>
        <li><a href="#saliency-based-evaluation">Model parameters</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#bibliography">Bibliography</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project
In Poland, the number of deaf people is estimated between 500,000 and 900,000. Each year there are more than 100 people who lose their hearing due to accidents, illness or age.  Members of a linguistic and cultural minority of deaf people call themselves Deaf. The primary language of the Deaf, which is, in a way, a marker of their cultural distinctiveness, is the sign language used in the country. The term sign language is a visual-spatial language, using the sense of sight, used by Deaf parents to communicate with their Deaf children. Polish Sign Language is a completely separate entity, having its own different grammar from Polish. Sign ideographic signs, which are equivalents of words and idioms, form the basis, and are complemented by dactylographic and supplementary signs.

### MediaPipe for sign languages recognition
A solution for hand landmarks detection and tracking is the MediaPipe:
https://developers.google.com/mediapipe

It was used in hand gestures recognition in a few project already, here are few of them:

Real-time Assamese Sign Language Recognition using MediaPipe and Deep Learning
Bora, J. and Dehingia, S. and Boruah, A. and Chetia, A. A. and Gogoi, D.  
https://doi.org/10.1016/j.procs.2023.01.117.

Sign Language Recognition - using MediaPipe & DTW 
https://www.sicara.fr/blog-technique/sign-language-recognition-using-mediapipe

Arabic Sign Language Recognition (ArSL) Approach Using Support Vector Machine, 
Ali, M. A.  and  Ewis, M. R. and Mohamed, G. E. and  Ali, H. H. and  Moftah, H. M. 
doi: 10.1109/ICCTA43079.2017.9497164.


### Dataset
Dataset for learning classyfier consisted of 29 945 images, divided in 29 classes. Those classes were:

TODO: table



### Data pre-processing

![HR_CAM](https://github.com/annasli378/HandSignClassification/blob/main/images/schemat_analiza_modelu(1).png)


### Model parameters

Random Forest classifier






<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple steps.

### Prerequisites
To build this project, you require:
* Python with installed the required libraries

### Installation
1. Clone the repo
   ```sh
   git clone https://github.com/annasli378/HandSignClassification.git
   ```
2. Open project in python environment (i.e PyCharm)

## Bibliography

https://arxiv.org/abs/1312.6034  <br>
`Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps
Scott M. Lundberg, Su-In Lee`








<!-- README created using the following template -->
<!-- https://github.com/othneildrew/Best-README-Template -->












