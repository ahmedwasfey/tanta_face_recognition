from flask import Flask, request, render_template
import cv2
import numpy as np
import base64
from detector import recognize_faces    
app = Flask(__name__)
@app.route('/')
def home():
    return render_template('main.html')

@app.route('/process_image', methods=['POST'])
def process_image():
    # Get image data from request
    image_data = request.json['image_data']
    # Decode image from base64 string
    image = base64.b64decode(image_data.split(',')[1])
    # Convert image to OpenCV format
    nparr = np.frombuffer(image, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # save image then detect faces
    # convert to rgb
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite('image.jpg', img)
    result = recognize_faces("image.jpg")
    # convert to cv2 image
    result = np.array(result)
    # Convert result to base64 string
    retval, buffer = cv2.imencode('.jpg', result)
    result_data = base64.b64encode(buffer).decode('utf-8')
    # Send result back to client
    return result_data


if __name__ == '__main__':
    app.run(host="0.0.0.0" ,port = 5500 , debug=True)