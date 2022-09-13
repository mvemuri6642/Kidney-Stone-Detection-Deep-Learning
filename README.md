# Kidney_Stone_Detection_DeepLearning
Kidney Stone Detection using Image Processing and Deep Neural Networks

http://ijream.org/papers/IJREAMV08I0387055.pdf

The Major Steps involved in the Detection of kidney stone using Deep Neural Networks are as follows:
	Gathering data
	Data Pre-Processing
	Image Processing
	Choosing the Deep Learning Model 
	GATHERING DATA:
The process of gathering the dataset depends upon the type of problem we are trying to solve. As this project is mainly focused on Image Classification, we need to acquire the required resources from open-source websites such as Kaggle, Github etc.
The dataset was uploaded to the Github repository and available in the following link: https://github.com/yildirimozal/Kidney_stone_detection/tree/main/Dataset
	DATA PRE-PROCESSING:
Data Preprocessing is a crucial step when you’re dealing with Image Datasets. As the images are of variable size, we need to convert them into fixed size. The dataset contains two folders “Kidney_Stone” and “Normal”. As the images doesn’t contain any label, we use these folder names to classify the images for training the Deep Learning Model. We resize all the images into 128 x 128 pixels. We then convert the image into Numpy array with the class label. We split the data into 90% for the training and 10% for testing the model.
	IMAGE PROCESSING:
Image Processing is divided into 2 modules:



	Image Pre-processing: 
It is basically one of the critical tasks because CT Scan images may have noise. In this operation we apply methods to enhance and filter the image using the Median filter and histogram equalization or Power Law Transformation.

Median Filter:
Python OpenCV provides the cv2.medianBlur() function to blur the image with a median kernel. This is a non-linear filtering technique. It is highly effective in removing impulse noise and salt-and-pepper noise. This takes a median of all the pixels under the kernel area and replaces the central component with this median value. 
Syntax: cv2.medianBlur(image,ksize)

 
 
 
Histogram Equalization:
Histogram Equalization technique is used for modifying the intensities of the image. The image we get is of lower quality therefore the image enhancing is done to improve the quality of the image. In this, each pixel intensity is modified so if the image is towards the darker side, then it gets stretched towards more white side and hence, we can say that the image is enhanced.
	Power Law Transformation: 
This method is better option for image enhancement. Here, the value of constant should be assumed on the basis of trial-and-error method. Gamma correction is useful when you want to change the contrast and brightness of an image.


	Image Segmentation:
Image Segmentation means to partition the image into different regions to extract relevant information. 
Segmentation is a vital aspect of medical imaging. It aids in the visualization of medical data and diagnostics of various diseases. 
Thresholding Method is applied on the image resulting from gamma adjustment to allow segmentation of image the foreground (stone and bones) and background. 
Thresholding value is based on the intensity of the pixel and intensities below this level becomes zero.
  
  
  APPLIED MODELS:

  Convolutional Neural Networks (CNN): 

Building a model to process the data collected is the most important step in the entire project. The algorithm we are using is Convolutional Neural Networks (CNN). CNNs are a class of Deep Neural Networks inspired by the visual cortex of human brain. A Convolutional Neural Network, or CNN, is a deep learning neural network designed for processing structured arrays of data such as images. CNNs are widely used in Computer Vision and have become the state of the art for many visual applications such as Image Classification and have also found success in Natural Language Processing for Text Classification. 
 


Here in our project the architecture we are using is 2-D CNN layer along with RELU (Rectified Linear Unit) activation function. The input frame to this layer is of dimensions 128x128 
The above figure is just a proposed architecture. Actual parameters and layers are yet to be tested using hyperparameter tuning.
In the above proposed architecture, we are also performing pooling to decrease the size of the feature maps, so that the number of parameters that a model must learn is decreased.
Pooling is used in reduction of the size of an image without losing patterns or details, because the oversized image can become a complex task for the machine to handle. There are two types of pooling can be performed:
Max Pooling – This identifies only important features of the previous feature map.
Average Pooling – This computes the average of the elements presents in the region of the feature map covered by filter. 
 

We are going to use two dense layers with RELU and Sigmoid activation functions.
ReLU Activation Function: The rectified linear activation function, or ReLU, is a linear function that, if the input is positive, outputs the input directly; else, it outputs zero. Because a model that uses this is quicker to train and generally achieves higher performance, it has become the default activation function for many types of neural networks. 
 


Sigmoid Activation Function: This function takes any real value as input and outputs values in the range of 0 to 1. A weighted sum of inputs is passed through an activation function and this output serves as an input to the next layer. When the activation function for a neuron is a sigmoid function, it is a guarantee that the output of this unit will always be between 0 and 1. The larger the input (more positive), the closer the output value will be to 1, whereas the smaller the input (more negative), the closer the output will be to 0, as shown below. We use this Activation function in output layer.
S(x) = □(1/(1+e^(-x) ))
 

Here, we are using the kernel size of 3x3 in the proposed architecture.
Convolution: Convolution is a process of identifying certain features in a feature map. Convolution helps the model or machine learn some important qualities of a picture through edge detection, noise reduction, blurring, sharpening and more.
Flattening: A two-dimensional matrix of features is flattened into vector of features that can be input to the Dense layer.



This project consists of the following modules:

5.1.	Read Images using OpenCV: We use OpenCV to read the images from the Hard Disk. Images Dataset is collected from GitHub repository. It contains around 1700 images in two different folders with class labels.

5.2.	Data Splitting: After reading them, we Split the dataset into Train and Test using train_test_split method from sklearn.model_selection module.

5.3.	Image Processing: We used Median Blur, Power Law Transformation and Thresholding methods in the Image processing stage. Image processing is used to remove the noise. 

  5.3.1.	Median Blur: Python OpenCV provides the cv2.medianBlur() function to blur the image with a median kernel
  
  5.3.2.	Power Law Transformation: This method is better option for image enhancement. 
  
  5.3.3.	Thresholding Method: Thresholding Method is applied on the image resulting from gamma adjustment to allow segmentation of image the foreground (stone and bones) and background. 
  
 5.4.	Building a Model: Convolutional Neural Networks (CNN) model is used for training a model. We use Random Search method to choose the model parameters. The following figure 
 
 5.5.	Saving the Model: Random Search algorithm randomly chooses different parameters every time we run the code. After Identifying the best parameters using Random Search, we need to save the model using keras.models module.
Syntax: model.save(‘/path’) 
Best parameters will be saved in file and can be used for further testing. We don’t need to train every time in order to test the data. We can just load the model and use predict method to get the desired output. 
Syntax: keras.models.load_model(‘/path’)


Accuracy: The XResNet50 has acquired an accuracy around 96 percent and the CNN has achieved an accuracy around 98 percent. The CNN is trained with the images after applying the image processing techniques on them. Random Search was used to find the model parameters. 

