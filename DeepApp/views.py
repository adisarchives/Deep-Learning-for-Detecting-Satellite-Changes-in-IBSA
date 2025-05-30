import os
import numpy as np
from django.shortcuts import render
from django.conf import settings
from keras.preprocessing import image
from keras.models import load_model
from keras.utils import to_categorical
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split
import io
import base64
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint


# Global Variables
dataset_dir = os.path.join(settings.MEDIA_ROOT, 'data')
class_names = ["cloudy", "desert", "green_area", "water"]

# Load the dataset
def LoadDataset(request):
    if request.method == 'GET':
        images = []
        labels = []

        # Traverse through the class directories to load images and labels
        for class_id, class_name in enumerate(class_names):
            class_dir = os.path.join(dataset_dir, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_name)
                    if img_path.endswith('.jpg') or img_path.endswith('.png'):
                        img = image.load_img(img_path, target_size=(224, 224))  # Resize the image
                        img_array = image.img_to_array(img)  # Convert the image to an array
                        images.append(img_array)
                        labels.append(class_id)

        images = np.array(images)
        labels = np.array(labels)

        # Normalize image data
        images = images / 255.0

        # Display images and their corresponding labels
        output = '<table border=1 align=center width=100%><tr><th>Image</th><th>Label</th></tr>'
        for i in range(min(20, len(images))):
            output += f'<tr><td><img src="/media/data/{class_names[labels[i]]}/{os.listdir(os.path.join(dataset_dir, class_names[labels[i]]))[i]}" width="100" height="100"></td>'
            output += f'<td>{class_names[labels[i]]}</td></tr>'
        output += '</table>'

        context = {'data': output}
        return render(request, 'ViewOutput.html', context)

# Train the CNN model for image classification
def TrainCNN(request):
    if request.method == 'GET':
        images = []
        labels = []

        # Load the images and their labels
        for class_id, class_name in enumerate(class_names):
            class_dir = os.path.join(dataset_dir, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_name)
                    if img_path.endswith('.jpg') or img_path.endswith('.png'):
                        img = image.load_img(img_path, target_size=(224, 224))  # Resize to 224x224
                        img_array = image.img_to_array(img)  # Convert to array
                        images.append(img_array)
                        labels.append(class_id)

        images = np.array(images)
        labels = np.array(labels)

        # Normalize the image data
        images = images / 255.0

        # Convert labels to one-hot encoding
        labels = to_categorical(labels, num_classes=len(class_names))

        # Split dataset into training and testing
        X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2)

        # Define CNN model
        cnn_model = Sequential()
        cnn_model.add(Convolution2D(32, (3, 3), input_shape=(224, 224, 3), activation='relu'))
        cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
        cnn_model.add(Convolution2D(64, (3, 3), activation='relu'))
        cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
        cnn_model.add(Flatten())
        cnn_model.add(Dense(256, activation='relu'))
        cnn_model.add(Dense(len(class_names), activation='softmax'))

        # Compile the model
        cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Train the model
        model_checkpoint = ModelCheckpoint('model/cnn_weights.hdf5', save_best_only=True, verbose=1)
        cnn_model.fit(X_train, y_train, batch_size=8, epochs=1, validation_data=(X_test, y_test), callbacks=[model_checkpoint])

        # Save the model
        cnn_model.save('model/cnn_weights.hdf5')

        context = {'data': 'CNN model trained and saved successfully!'}
        return render(request, 'ViewOutput.html', context)
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from django.shortcuts import render
from PIL import Image
import io

# Class names for predictions
class_names = ["cloudy", "desert", "green_area", "water"]

def PredictImageAction(request):
    if request.method == 'POST':
        # Clear any existing session to avoid conflicts
        from tensorflow.keras import backend as K
        K.clear_session()

        # Load the image file from the POST request
        image_file = request.FILES['image']
        
        # Open the uploaded image using PIL and convert to a format suitable for Keras
        img = Image.open(image_file)
        # Convert to RGB if necessary
        if img.mode != "RGB":
         img = img.convert("RGB")

        # Resize to match the expected input shape of the CNN model
        img = img.resize((224, 224))  # Ensure resizing to (224, 224)

        # Convert the PIL image to a NumPy array
        img_array = np.array(img)

        # Ensure the image array has the correct shape (224, 224, 3)
        if img_array.shape != (224, 224, 3):
            raise ValueError(f"Incorrect image shape: {img_array.shape}. Expected (224, 224, 3).")

        # Expand dimensions to match the model's expected input shape (batch_size, height, width, channels)
        img_array = np.expand_dims(img_array, axis=0)

        # Normalize the image
        img_array = img_array / 255.0

        # Load the trained CNN model
        cnn_model = load_model('model/cnn_weights.hdf5')

        # Make prediction
        prediction = cnn_model.predict(img_array)
        class_idx = np.argmax(prediction)  # Get the index of the class with the highest probability
        predicted_class = class_names[class_idx]

        # Return prediction to the template
        context = {'data': f'This image belongs to the class: {predicted_class}'}
        return render(request, 'ViewOutput.html', context)

    return render(request, 'DetectSatillite.html', {})




def PredictImage(request):
    if request.method == 'GET':
       return render(request, 'DetectSatillite.html', {})    

def index(request):
    if request.method == 'GET':
       return render(request, 'index.html', {})    

def UserLogin(request):
    if request.method == 'GET':
       return render(request, 'UserLogin.html', {})

def Register(request):
    if request.method == 'GET':
       return render(request, 'Register.html', {})

def UserLoginAction(request):
    if request.method == 'POST':
        global username
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        status = "UserLogin.html"
        context= {'data':'Invalid login details'}                      
        if "admin" == username and "admin" == password:
            context = {'data':"Welcome "+username}
            status = 'UserScreen.html'            
        return render(request, status, context)              


    
