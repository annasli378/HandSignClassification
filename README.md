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

| Sign from PSL      | Class label | Sign from PSL           | Class label |
|--------------------|-------------|-------------------------|-------------|
| number 0, letter O | 0           | letter M                | 15          |
| number 1           | 1           | letter N                | 16          |
| number 2           | 2           | letter P                | 17          |
| number 3           | 3           | letter R                | 18          |
| number 4           | 4           | letter S                | 19          |
| number 5           | 5           | letter U                | 20          |
| letter A           | 6           | letter W                | 21          |
| letter B           | 7           | letter Y                | 22          |
| letter C           | 8           | additional character Aw | 23          |
| letter D           | 9           | additional character Bk | 24          |
| letter E           | 10          | additional character Cm | 25          |
| letter F and T     | 11          | additional character Ik | 26          |
| letter H           | 12          | additional character Om | 27          |
| letter I and J     | 13          | additional character Um | 28          |
| letter L           | 14          |                         |             |

To learn more about Polish Sign Language visit those sites :)
https://kulturawrazliwa.pl/cykl/lekcje-online-pjm/
https://www.spreadthesign.com/pl.pl/search/
https://www.slownikpjm.uw.edu.pl
https://cwn.uph.edu.pl/dictlessons_thema1P


### Data pre-processing

![schema](https://github.com/annasli378/HandSignClassification/blob/main/images/schema.png)

### Model parameters

For classifing our normalized data we used Random Forest (more info here: https://github.com/annasli378/ChoosingModelForPSLRecognition)


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





<!-- README created using the following template -->
<!-- https://github.com/othneildrew/Best-README-Template -->
