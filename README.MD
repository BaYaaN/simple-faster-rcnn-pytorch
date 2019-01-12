
## 1. Install dependencies

requires PyTorch >=0.4

- install PyTorch >=0.4 with GPU (code are GPU-only), refer to [official website](http://pytorch.org)

- install cupy, you can install via `pip install cupy-cuda80` or(cupy-cuda90,cupy-cuda91, etc).

- install other dependencies:  `pip install -r requirements.txt `

- Optional, but strongly recommended: build cython code `nms_gpu_post`: 

  ```Bash
  cd FasterRcnn/model/utils/nms/
  python build.py build_ext --inplace
  cd -
  ```

- start visdom for visualization

```Bash
nohup python -m visdom.server &
```


## 2. Train

### 2.1 Prepare data

#### Pascal VOC2007

Download pretrained model from [Google Drive](https://drive.google.com/open?id=1cQ27LIn-Rig4-Uayzy_gH5-cW-NRGVzY)
and put to pretrained-model directory

1. Download the training, validation, test data and VOCdevkit

   ```Bash
   wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
   wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
   wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
   ```

2. Extract all of these tars into one directory named `VOCdevkit`

   ```Bash
   tar xvf VOCtrainval_06-Nov-2007.tar
   tar xvf VOCtest_06-Nov-2007.tar
   tar xvf VOCdevkit_08-Jun-2007.tar
   ```

3. It should have this basic structure

   ```Bash
   $VOCdevkit/                           # development kit
   $VOCdevkit/VOCcode/                   # VOC utility code
   $VOCdevkit/VOC2007                    # image sets, annotations, etc.
   # ... and several other directories ...
   ```

4. modify `voc_data_dir` cfg item in `FasterRcnn/utils/config.py`, or pass it to program using argument like `--voc-data-dir=/path/to/VOCdevkit/VOC2007/` .


### 5.3 begin training

```Bash
mkdir checkpoints/ # folder for snapshots
```

```bash
python gan_train.py train --plot_every=30 --load_path='pretrained-model/fasterrcnn_12222105_0.712649824453_caffe_pretrain.pth'
```

If you run visdom result should be visaulize


