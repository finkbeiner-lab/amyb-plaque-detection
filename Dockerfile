FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

CMD git clone https://github.com/finkbeiner-lab/amyb-plaque-detection.git 

CMD cd amyb-plaque-detection

WORKDIR /app
COPY . .
RUN pip install -r requirements.txt

CMD python /app/src/testing/explain.py 
