# Use uma imagem base do TensorFlow. Você pode escolher uma versão específica se necessário.
FROM tensorflow/tensorflow:latest

# Copie seu app para o contêiner
WORKDIR /app
COPY ./requirements.txt /app/
COPY ./app.py /app/
COPY ./models/ /app/models/
COPY ./image_test/ /app/image_test/


# Instale pacotes adicionais necessários
RUN apt-get update && apt-get install -y libsm6 libxext6 libxrender-dev libgl1-mesa-glx
RUN python3 -m pip install --upgrade pip
RUN pip install -r requirements.txt

# Defina a porta padrão para o Flask
EXPOSE 5000

# Comando para rodar o Flask
CMD ["python3", "app.py"]

