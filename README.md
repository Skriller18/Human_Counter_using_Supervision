# Human_Counter_using_Supervision

A YoloV8 model designed to run on BeagleBoard BBAI64 for counting and detecting humans using LineZone API of RoboFlow Supervision

## Clone the repository
```bash
git clone https://github.com/Skriller18/Human_Counter_using_Supervision.git
```

## Install the requirements
```bash 
pip install -r requirements.txt
```

## Modify the package folders at supervision by replacing with the folders
```bash
mv annotators classification dataset detection draw geometry ~/site-packages/supervision/
```

## Use the model present in the Model folder :
```bash 
cd od-8220_onnxrt_coco_edgeai-mmdet_yolox_s_lite_640x640_20220221_model_onnx
```
If needed change the models as desired. The small version YOLO is used as of now

## Run the scripts for following Cases:
### Case 1: Counting Humans using Line tracker
```bash
python line_tracker.py
```
### Case 2: Counting Humans using Area Box
```bash
python sv+bt.py
````

```
Versions of CUDA and ONNXRuntime :
CUDA : 11.7 to 12.0
onnxruntime-gpu : 1.5 to 1.6
```

