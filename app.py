model.load_weights('my_h5_model.h5')
#-------------------------------------------------------------------------------------------#

COUNT = 0
#from flask import Flask, render_template, request, send_from_directory, *
from flask import *
#from flask_ngrok import run_with_ngrok
import cv2
app = Flask(__name__)
#run_with_ngrok(app)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 1
 
@app.route('/')
def man():
    return render_template('index.html')
 
@app.route('/home', methods=['POST'])
def home():
    global COUNT
    img = request.files['image']
 
    img.save('static/{}.png'.format(COUNT))    
    print("**image saved")
    #img_arr = cv2.imread('static/{}.jpg'.format(COUNT))
    img = cv2.imread('static/{}.png'.format(COUNT))
 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28,28), interpolation = cv2.INTER_AREA)
    newing = tf.keras.utils.normalize (resized, axis = 1)
    newing = np.array(newing).reshape(-1, IMG_SIZE, IMG_SIZE,1)
    predictions = model.predict(newing)
 
    preds=(np.argmax(predictions))
    print("**preds=", preds)
    COUNT = COUNT + 1
    print("**COUNT= ", COUNT)
    return render_template('prediction.html', data=preds)
 
@app.route('/load', methods=['GET'])
def load():
  global COUNT
  print("count= ", COUNT)
  return send_from_directory('static', "{}.png".format(COUNT-1))
 
 
if __name__ == '__main__':
    app.run()
