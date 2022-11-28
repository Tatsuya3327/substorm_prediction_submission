# Substorm Prediction
This is a submission repository for UC Berkeley Data Science Discovery Fall 2022.

## Code Description
`code` folder has four notebooks.
1. `convert_cdf_to_png.ipynb`: convert and preprocess auroral images in CDF (Common Data Format) to png files.
2. `image_to_probability.ipynb`: convert images to probability of six types of auroras (arc, discrete, diffuse, cloudy, moon, clear).
3. `preprocess_datasets.ipynb`: combine magnetometer data and probabilities of auroral images and split to 30-minute datapoints.
4. `substorm_prediction.ipynb`: classify if a 30-minute time window contains a substorm onset or not using a pytorch LSTM model.

## Problem
Our team’s goal was to develop a model to predict the onset of auroral substorms. As streams of charged particles originating from the Sun in the form of Solar Winds come in contact with the Earth’s magnetosphere, the particles in the Earth’s atmosphere are energized and ionized and emit bright lights forming the spectacular sights of auroras. An auroral substorm is the phenomenon of the sudden brightening of these auroral arcs, caused by disturbances in the Earth’s magnetosphere due to a sudden release of accumulated energy from the magnetosphere. Very intense substorms can lead to outages of GPS signals and create intense electric currents that can damage electronic systems in satellites and on the ground.

In order to capture these brief moments when the substorm occurs, we wish to develop a system that is capable of predicting the onset of substorms ahead of time, similarly to that of a weather forecasting system, but for substorms of auroras. By analyzing historical data of magnetometer readings collected by satellites from NASA’s THESIS mission as well as aurora images taken by ground stations of the NASA THEMIS project near the Earth’s pole, our team aimed to studying patterns in the fluctuation of the Earth’s magnetic field and movements of the auroral arcs preceding the onset of substorms, to train a machine learning model to predict future onsets based.

## Data
We combined time series magnetometer readings and features extracted from time series auroral images.

### Magnetometer data
Readings of H, D, Z components of the magnetic field. We choose readings starting from t = 0, 15, 30, 45 mins etc. for 30 min intervals (3 second intervals).

### Auroral image data
1. Readings of image CDF converted to image png files with their respective time labels at every 3 seconds. Using the image classifier, we have predicted probabilities of those images in 6 different classes - arc, discrete, diffuse, cloudy, moon, clear. We choose these probability readings starting every hour from t = 0, 15, 30, 45 mins etc. for 30 min intervals. 
2. Readings of brightness of images. We choose readings starting from t = 0, 15, 30, 45 mins etc. for 30 min intervals (3 second intervals)

### Processed Data Input to Neural Network
Data matrix of dimensions 10 * 600
1. 10 rows -> arc, discrete, diffuse, cloudy, moon, clear, H, D, Z, total brightness
2. 600 columns -> 30 (mins) * 60 (seconds/min) / 3 (seconds/reading) = 600

## Solution
We first needed to distill our collected data into image classifications before we could begin predicting substorms. To achieve this first goal, we used a TAME classifier, which took in aurora imaging and classified it into one of six classes (“arc,” “diffuse,” “discrete,” “cloud,” “moon,” and “clear”). The classifier trained the most accurate pretrained pytorch neural network on a support vector machine to determine these classes.

Once we have classified the set of images, we combined the classification data with the magnetometer data and total brightness of the image to create our dataset for our predictive model. This model used LSTM (Long short-term memory) modules as a base for the machine learning model, as this specific model is good for analyzing time series data. LSTM modules have feedback connections between each module (hence the name), allowing subsequent data points that are temporally adjacent to each other have a more meaningful effect on how the model detects overall change between regular activity and a possible substorm onset.

During our research on types of models to use, we came across a ResNet model which could classify time series magnetometer data as a possible substorm onset within a 1 hour prediction period (taking in a 2 hour input period). A ResNet model is a special type of Convolutional Neural Net that adds skip connections within the net to reduce problems with vanishing or exploding gradients that could damage the test accuracy of the model when deeper layers are added to the model. Although this model had similar goals as our project, we decided to use an LSTM-based approach because our model incorporates multiple sources of input data to get a more holistic view of the environment when making predictions. We also found that the ResNet model took in only 1-dimensional time series data to process, while our project wanted to also incorporate 2-dimensional image time series data to improve model robustness.

## Result and Next Steps
Our model achieved around 70 % accuracy on the validation data (history of losses and accuracies available in `result`). 

We can keep using the probability or other features from aurora images (e.g. splitting an image to four quadrants and find total brightness in each area).

Our model is currently a descriptive model while we eventually want forecast the probability of occurrence of a substorm in a near future. That requires more labels of substorms (we only have one month but need tens of years). We also have to implement a fully forecasting model rather than a descriptive sequential model. To better performances, we need to work more on feature engineering especially on auroral images.

Due to the lack of labelled data, our model is experiencing overfitting (our model has 3000 learning parameters but 300 datapoints). Once we have a larger database we can look for small changes that actually precede the substorm onset, like change in one magnetic field component while the others remain steady, fluctuations in aurora brightness, or changes in shape of arcs but not their brightness.
