####  Birds-Vs-Drone-Detection


## AIM
 
To create a Birds & Drones Detection System, the challenge goal of te system  is to detect a drone appearing at some time in a short video sequence where birds are also present.

## Objectives

• The main objective of the project is to create a program which can be either run on Jetson nano or any pc with YOLOv5 installed and start detecting using the camera module on the device.

• Using appropriate datasets for recognizing and interpreting data using machine learning.

• To show on the optical viewfinder of the camera module, the challenge goal is to detect a drone appearing at some time in a short video sequence where birds are also present.


## Abstract :

The challenge aims at attracting research efforts to identify novel solutions to the problem outlined above, i.e., discrimination between birds and drones at far distance, by providing a video dataset that may be difficult to obtain (drone flying require special conditions and permissions, and shore areas are needed for the considered problem). 

The challenge goal is to detect a drone appearing at some time in a short video sequence where birds are also present: the algorithm should raise an alarm and provide a position estimate only when a drone is present, while not issuing alarms on birds.

We have completed this project on jetson nano which is a very small computational device.

 A lot of research is being conducted in the field of Computer Vision and Machine Learning (ML), where machines are trained to identify various objects from one another. Machine Learning provides various techniques through which various objects can be detected.

 One such technique is to use YOLOv5 with Roboflow model, which generates a small size trained model and makes ML integration easier.

The dataset is continually increased over consecutive installments of the challenge and made available to the community afterwards.

Small drones are a rising threat due to their possible misuse for illegal activities such as smuggling of drugs as well as for terrorism attacks using explosives or chemical weapons. 

## Introduction

This project is based on a Drone and bird detection model with modifications. We are going to implement this project with Machine Learning and this project can be even run on jetson nano which we have done.


The use of drones, whose origin is in the military domain, has been extended to several application fields including traffic and weather monitoring [1], precision agriculture [2], and many more.

Technology can be well utilized in such cases to reduce human effort and provide analytical data and representation. Artificial Intelligence (AI) is one such field that tries to mimic how humans learn and is an active research field.

The detection of drones or birds using machine learning based approaches, more specifically deep learning algorithms. The need for such sophisticated methods stems from the recognised difficulty for air traffic controllers to detect and differentiate between birds and drones.

Neural networks and machine learning have been used for these tasks and have obtained good results.

Machine learning algorithms have proven to be very useful in pattern recognition and classification, and hence can be used for Helmet detection as well.


## Literature Review

➢ Birds Vs Drone detection is crucial for safeguarding Birds  life. They detect the presence of Birds and Drones. 

➢ With the advent of computer vision and image processing, vision based detection techniques are widely used in recent times. This technique provides numerous advantages over the conventional system such as quicker response and wider coverage area. Many algorithms are available for Birds and drone detection. These algorithms use convolution neural networks which give way better accuracy in detection than the conventional methods of detection.

➢ Deep learning algorithms can learn the useful features for Birds and drone detection from a video source. Convolutional neural networks are a branch of deep learning that can extract topological properties from an image.

➢ In our approach we use convolutional neural networks to train the system for intelligently detecting Birds and drone. This is done by training the system on a very diverse dataset of Birds and drone images. This can successfully improve the accuracy of detection. This method will improve the accuracy of detection than the existing vision based models.

➢ Drone detection has become an essential task in object detection as drone costs have decreased and drone technology has improved.


## Jetson Nano Compatibility

➢ The power of modern AI is now available for makers, learners, and embedded developers everywhere.

➢ NVIDIA® Jetson Nano™ Developer Kit is a small, powerful computer that lets you run multiple neural networks in parallel for applications like image classification, object detection, segmentation, and speech processing. All in an easy-to-use platform that runs in as little as 5 watts.

➢ Hence due to ease of process as well as reduced cost of implementation we have used Jetson nano for model detection and training.

➢ NVIDIA JetPack SDK is the most comprehensive solution for building end-to-end accelerated AI applications. All Jetson modules and developer kits are supported by JetPack SDK.

➢ In our model we have used JetPack version 4.6 which is the latest production release and supports all Jetson modules.

## Proposed System

Study basics of machine learning and image recognition.

Start with implementation

 ➢ Front-end development
 ➢ Back-end development
 
Testing, analysing and improvising the model. An application using python and Roboflow and its machine learning libraries will be using machine learning to identify whether objects are plastics, plastic waste like bottles or garbage.

Use datasets to interpret the object and suggest whether the object birds or drones or aircrafts.


## Methodology

The Birds and drones  system is a program that focuses on implementing real time Birds and Drone detection.

It is a prototype of a new product that comprises of the main module:  Birds and Drone detection and then showing on viewfinder whether the object is Bird  or drone or any other aircrfts.

## Birds and Drone detection Module

1]  Birds and Drone detection Module

➢ Ability to detect the location of object in any input image or frame. The output is the bounding box coordinates on the detected object.

➢ For this task, initially the Dataset library Kaggle was considered. But integrating it was a complex task so then we just downloaded the images from gettyimages.ae and google images and made our own dataset.

➢ This Datasets identifies object in a Bitmap graphic object and returns the bounding box image with annotation of object present in a given image.

2] Classification

➢ Classification of the object based on whether it is Birds and drones or not.

➢ Hence YOLOv5 which is a model library from roboflow for image classification and vision was used.

➢ There are other models as well but YOLOv5 is smaller and generally easier to use in production.

➢ YOLOv5 was used to train and test our model for various classes like Birds and Drones. We trained it for 149 epochs and achieved an better accuracy.



# Setup

# Installation

Initial Setup

Remove unwanted Applications.

      sudo apt-get remove --purge libreoffice*
      sudo apt-get remove --purge thunderbird*


 Create Swap file
      sudo fallocate -l 10.0G /swapfile1
      sudo chmod 600 /swapfile1
      sudo mkswap /swapfile1
      sudo vim /etc/fstab
      
      #################add line###########
      /swapfile1 swap swap defaults 0 0

Cuda Configuration

vim ~/.bashrc
     
     
     #############add line #############
     export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
     export
     LD_LIBRARY_PATh=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
     export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
     
     
     source ~/.bashrc
     
     
  Update & Upgrade

        sudo apt-get update
        sudo apt-get upgrade
     
  Install some required Packages

        sudo apt install curl
        curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
        sudo python3 get-pip.py
        sudo apt-get install libopenblas-base libopenmpi-dev

        sudo pip3 install pillow
        
 Install Torch

         curl -LO https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl
         mv p57jwntv436lfrd78inwl7iml6p13fzh.whl torch-1.8.0-cp36-cp36m-linux_aarch64.whl
         sudo pip3 install torch-1.8.0-cp36-cp36m-linux_aarch64.whl

         #Check Torch, output should be "True" 
         sudo python3 -c "import torch; print(torch.cuda.is_available())
         
   Installation of torchvision
   
          git clone --branch v0.9.1 https://github.com/pytorch/vision torchvision
          cd torchvision/
          sudo python3 setup.py install
     
  Clone Yolov5

          git clone https://github.com/ultralytics/yolov5.git
          cd yolov5/
          sudo pip3 install numpy==1.19.4

          #comment torch,PyYAML and torchvision in requirement.txt

          sudo pip3 install --ignore-installed PyYAML>=5.3.1
          sudo pip3 install -r requirements.txt
          
Download weights and Test Yolov5 Installation on USB webcam

          sudo python3 detect.py
          sudo python3 detect.py --weights yolov5s.pt  --source 0

     
   
 # Birds And Drones Dataset Training
 
# We used Google Colab And Roboflow

train your model on colab and download the weights and past them into yolov5 folder link of project.


# Running Birds And Drones  Detection Model

source '0' for webcam

!python detect.py --weights best.pt --img 416 --conf 0.1 --source 0


## Advantages

➢ Video-based Birds Drone- detection is currently a standard technology due to image processing, computer vision, and Artificial Intelligence. These systems have remarkable potential advantages over traditional methods, such as a fast response and wide detection areas.
Drone detection has become an essential task in object detection as drone costs have decreased and drone technology has improved

➢ Deep learning techniques have the advantage of extracting the features automatically, making this process more effective and dramatically improving the state-of-the-art in Image Classification and object detection methods

➢ It can then convey to the person who present in control room if it needs to be completely automated

➢ When completely automated no user input is required and therefore works with absolute efficiency and speed.

## Application

➢ Detects object class like Birds and Drones in a given image frame or view finder using a camera module.

➢ Can be used in various places.

➢ Can be used as a refrence for other ai models based on fire and smoke detection.


## Refrences

1] Roboflow :- https://roboflow.com/
2] Datasets or images used:https://www.kaggle.com/datasets
3] Google images




