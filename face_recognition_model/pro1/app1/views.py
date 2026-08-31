from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os

# Load model once when server starts
# MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model_name.h5')
model = load_model('face_regonition_demo (2).h5')
class_names=['Angry',
    'Disgust',
    'Fear',
    'Happy',
    'Neutral',
    'Sad',
    'Surprise']
#class_names=[]
def face_predict(request):
    prediction = None
    image_url = None

    if request.method == 'POST' and request.FILES.get('image'):
        uploaded_image = request.FILES['image']

        fs = FileSystemStorage()
        filename = fs.save(uploaded_image.name, uploaded_image)

        image_url = fs.url(filename)
        image_path = fs.path(filename)

        # Image preprocessing
        img = Image.open(image_path).convert('RGB')
        img = img.resize((75,75))

        img_array = np.array(img)
        #img_array = img_array.reshape(1, 784)
        img_array = img_array.astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        pred = model.predict(img_array)
        prediction = class_names[np.argmax(pred)]


    return render(
        request,
        'face_predict.html',
        {
            'prediction': prediction,
            'image_url': image_url
        }
    )


