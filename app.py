from flask import Flask, request, jsonify
import tensorflow as tf
from keras import backend as K
import numpy as np
import cv2
import io
	
app = Flask(__name__)


TF_MODEL_FILE_PATH = "./models/modelo_treinado_224x224_ONLY_TRANSFER_LEARNING_MobileNetv2_softmax_4_categorias.h5"

def obter_categoria(valor_procurado):
    dict ={'Armature Exposure':0, 'Concrete Cracks':1, 'Infiltration':2, 'Normal':3}
    for chave, valor in dict.items():
        if valor == valor_procurado:
            return chave
    return None

@app.route('/predict', methods=['POST'])
def predict():
	image = request.files['image'].read()
	image = np.frombuffer(image, np.uint8)
	image = cv2.imdecode(image, cv2.IMREAD_COLOR)
	image = cv2.resize(image, (224, 224))
	image = np.expand_dims(image, axis=0)
	image = tf.convert_to_tensor(image, dtype=tf.float32)
	model = tf.keras.models.load_model(TF_MODEL_FILE_PATH)
	with tf.device('/CPU:0'):
		output = model.predict(image)
	predicted_class = obter_categoria(int(np.argmax(np.round(output)[0])))
	
	#Limit the memory usage
	del model
	del image
	
	response_data = {
	    'prediction': predicted_class,
        'confidence': float(np.max(output))
    }
	return jsonify(response_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

