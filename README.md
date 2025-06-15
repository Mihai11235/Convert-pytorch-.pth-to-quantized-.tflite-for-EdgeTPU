# Convert-pytorch-.pth-to-quantized-.tflite-for-EdgeTPU

Prequisites to convert .pth to .tflite

```
python --version 3.7
```

requirements.txt:
```
PILLOW==9.5.0
torchvision==0.13.1
torch==1.12.1
numpy==1.19.5
tensorflow==2.4.1
urllib3==1.26.20
onnx-tf==1.10.0 //from onnx-tensorflow repo(see below)
```

Clone this repository and install it:

```
git clone https://github.com/onnx/onnx-tensorflow.git && cd onnx-tensorflow
pip install -e .
```

Prequisites for using edgetpu_compiler

```
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -

echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list

sudo apt-get update

sudo apt-get install edgetpu-compiler
```


Run:

```
source .venv/bin/activate
python transform_pth_onnx_1.py
python transform_onnx_tf_2.py
python patch_savedmodel_3.py
```

Use ```convert_tf_tflite_quant.ipynb``` to get quantized .tflite model

For compiling .tflite for Edge Tpu run:
```
edgetpu_compiler smallunet_model_quant.tflite
```