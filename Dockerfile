FROM python:3.9-slim

WORKDIR /

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

RUN python train_model.py
RUN python predict.py --all

EXPOSE 5000

CMD ["python", "app.py"]
