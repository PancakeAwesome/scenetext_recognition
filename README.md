# CRNN-detection
This is a scene text recognition by CRNN model wchich composite by CNN + LSTM + CTC modules.We use CRNN to discern IDcards and printed transfer orders after detecting by CTPN.If you want look at CTPN model,please go to [there](https://github.com/PancakeAwesome/scenetext_detection).The project implemented by Tensorflow.The origin paper can be found [here](https://arxiv.org/pdf/1507.05717v1.pdf7). For more detail about the paper and code, see this [blog][1]

[1]:http://pancakeawesome.ink/%E8%AE%BA%E6%96%87%E7%AC%94%E8%AE%B0%E4%B9%8BCRNN(Convolutional%20Recurrent%20Neural%20Network).html
***

## Prerequisite

1. TensorFlow 1.2

2. Opencv3 Not a must, used to read images.

3. Numpy


## How to run

There are many other parameters with which you can play, have a look at [utils.py](https://github.com/PancakeAwesome/scenetext_recognition/blob/master/utils.py).

**Note** that the num_classes is not added to parameters talked above for clarification.


``` shell
# cd to the your workspace.
# The code will evaluate the accuracy every validation_steps specified in parameters.

ls -R
  .:
  imgs  utils.py  helper.py  main.py  cnn_lstm_otc_ocr.py

  ./imgs:
  train  infer  val  labels.txt
  
  ./imgs/train:
  1.png  2.png  ...  50000.png
  
  ./imgs/val:
  1.png  2.png  ...  50000.png

  ./imgs/infer:
  1.png  2.png  ...  300000.png
   
  
# Train the model.
python ./main.py --train_dir=./imgs/train/ \
                 --val_dir=./imgs/val/ \
                 --image_height=60 \
                 --image_width=180 \
                 --image_channel=1 \
                 --max_stepsize=64 \
                 --num_hidden=128 \
                 --log_dir=./log/train \
                 --num_gpus=0 \
                 --mode=train

# Inference
python ./main.py --infer_dir=./imgs/infer/ \
                 --checkpoint_dir=./checkpoint/ \
                 --num_gpus=0 \
                 --mode=infer

```

