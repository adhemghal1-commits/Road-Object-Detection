# Variables: override on the command line, e.g. `make train EPOCHS=50 DEVICE=cpu`
SRC_RDD ?= /path/to/RDD2022
DATA    ?= data/rdd2022_yolo/data.yaml
EPOCHS  ?= 100
DEVICE  ?= 0
TUNE_EPOCHS ?= 30
TUNE_ITERS  ?= 50

.PHONY: install data train tune train-tuned evaluate report app test all docker
install:
	pip install -r requirements.txt
data:
	python -m src.prepare_rdd2022 --src $(SRC_RDD) --out data/rdd2022_yolo \
	  --countries Japan India Czech Norway United_States China_MotorBike
train:
	python -m src.train --data $(DATA) --epochs $(EPOCHS) --device $(DEVICE)
tune:
	python -m src.tune --data $(DATA) --epochs $(TUNE_EPOCHS) --iterations $(TUNE_ITERS) --device $(DEVICE)
train-tuned:
	python -m src.train --tuned --data $(DATA) --epochs $(EPOCHS) --device $(DEVICE)
evaluate:
	python -m src.evaluate --data $(DATA) --device $(DEVICE) --onnx
report:
	python -m src.report
app:
	streamlit run app/app.py
test:
	pytest -q
all: train tune train-tuned evaluate report
docker:
	docker build -t road-damage-yolo . && docker run -p 8501:8501 road-damage-yolo
