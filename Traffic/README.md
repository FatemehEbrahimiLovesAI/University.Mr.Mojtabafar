# Traffic Volume Prediction Project

This project is a machine learning-based application designed to predict traffic volume on a binary scale (high or low) based on weather-related data. It includes a graphical user interface (GUI) built using PyQt6 to display predictions.

## Project Description

This application loads a traffic dataset, preprocesses the data, trains a classification model, evaluates its performance, and presents predictions on a randomly selected subset of the test set via a user interface.

## Dataset

The dataset used is the Metro Interstate Traffic Volume dataset. It contains hourly weather and traffic volume data collected from a sensor on Interstate 94 in Minneapolis, USA.

Key preprocessing steps include:
- Removing rows with missing values.
- Binarizing the target variable `traffic_volume` based on its median.
- Encoding the categorical feature `weather_main` using label encoding.

## Dataset Source

The dataset used in this project is provided by the U.S. Department of Transportation:  
Metro Interstate Traffic Volume Dataset.  
Retrieved from Kaggle: https://www.kaggle.com/datasets/chebotinaa/metro-interstate-traffic-volume

## Features Used

The model is trained on the following features:
- temp
- rain_1h
- snow_1h
- clouds_all
- weather_main (encoded)

## Model and Evaluation

The prediction model used is a Random Forest Classifier from scikit-learn. The dataset is split into training and test sets (80/20 ratio). Model accuracy is printed to the console after training.

## GUI Overview

A PyQt6-based GUI displays predictions for 10 random samples from the test set. It shows each sample's predicted and actual labels, along with an indication of correctness.

## Installation and Usage

### Requirements

- Python 3.10 or higher
- pandas
- numpy
- scikit-learn
- PyQt6

### Running the Program

1. Make sure the dataset CSV file path is correctly specified in the script.
2. Install required packages (optional):
   ```
   pip install pandas numpy scikit-learn PyQt6
   ```
3. Run the program:
   ```
   python traffic_prediction.py
   ```

## Authors

This project was developed by Reyhaneh Khorshidi and Fatemeh Ebrahimi as part of a university course project.

## License

This project is free to use.