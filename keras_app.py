from gevent.pywsgi import WSGIServer

import Experiments.Keras_inception.predictor_keras_model as prdict
from flask import Flask, render_template, request, flash, redirect, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__, template_folder='templates')
prdict_model = prdict.price_model()
prdict_model.load_dataset()
prdict_model.data_augment()
prdict_model.model_loader('model/priceye_keras.model')


@app.route('/')
def start_event():
    return render_template('index.html')


@app.route('/result', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        input_img = request.files['file']
        print(" --- Prediction Inference ---")
        result = prdict_model.predictor(secure_filename(input_img))
        item = prdict_model.load_vector[list(result).index(max(result))]
        # flash("It's {}".format(item))
        return "It's {}".format(item)


if __name__ == '__main__':
    print(" --- WebApp Online ---")
    # app.run(debug=True)
    # print(" --- Initalising App ---")
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()
