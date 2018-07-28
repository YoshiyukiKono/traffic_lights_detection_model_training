# traffic_lights_detection_model_training
For CarND-Capstone


git clone https://github.com/YoshiyukiKono/traffic_lights_detection_model_training.git

git clone https://github.com/tensorflow/models.git

pip install --upgrade pip
pip install absl-py

# API Download
git clone https://github.com/pdollar/coco
# API install
cd coco/PythonAPI
python setup.py install
make

sudo apt install python-pip

python2.7 -m pip install absl-py
python2.7 -m pip install tensorflow
python2.7 -m pip install --upgrade numpy

https://stackoverflow.com/questions/14372706/visual-studio-cant-build-due-to-rc-exe
I succeeded to install cocoapi using the above repository.
To solve the errors that I had faced, I needed to copy rc.exe and rcdll.dll ;
from C:\Program Files (x86)\Windows Kits\8.1\bin\x86
to C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin

https://stackoverflow.com/questions/30418481/error-dict-object-has-no-attribute-iteritems
  File "C:\Work_BigData\tensorflow_models\models\research\object_detection\model_lib.py", line 282, in model_fn
    losses = [loss_tensor for loss_tensor in losses_dict.itervalues()]
AttributeError: 'dict' object has no attribute 'itervalues'

https://github.com/tensorflow/models/issues/3705#issuecomment-375563179

  File "C:\Work_BigData\tensorflow_models\models\research\object_detection\model_lib.py", line 392, in model_fn
    eval_metric_ops = {str(k): v for k, v in eval_metric_ops.iteritems()}
AttributeError: 'dict' object has no attribute 'iteritems'

Just a debug pring on
File C:\Anaconda3\envs\carnd-term1\Lib\site-packages\tensorflow\contrib\data\python\ops\interleave_ops.py




--- Logging error ---
Traceback (most recent call last):
  File "C:\Anaconda3\envs\carnd-term1\lib\logging\__init__.py", line 983, in emit
    stream.write(self.terminator)
OSError: raw write() returned invalid length 194 (should have been between 0 and 97)
Call stack:
  File "C:\Anaconda3\envs\carnd-term1\lib\site-packages\tensorflow\python\ops\script_ops.py", line 158, in __call__
    ret = func(*args)
  File "C:\Work_BigData\tensorflow_models\models\research\object_detection\metrics\coco_evaluation.py", line 274, in update_op
    'groundtruth_is_crowd': gt_is_crowd[:num_gt_box]})
  File "C:\Work_BigData\tensorflow_models\models\research\object_detection\metrics\coco_evaluation.py", line 82, in add_single_ground_truth_image_info
    'previously added', image_id)
  File "C:\Anaconda3\envs\carnd-term1\lib\site-packages\tensorflow\python\platform\tf_logging.py", line 125, in warning
    _get_logger().warning(msg, *args, **kwargs)
Message: 'Ignoring ground truth with image id %s since it was previously added'
Arguments: (1363203192,)
WARNING:tensorflow:Ignoring detection with image id 1363203192 since it was previously added


### Sluth Install
Prerequisites
Sloth is implemented in Python and PyQt4, so it needs both. It further depends on either PIL or okapy for image loading.

conda install pyqt=4
https://stackoverflow.com/questions/21637922/how-to-install-pyqt4-in-anaconda
https://teratail.com/questions/68504

pip install okapi ("conda install PIL" suggests python downgrade to 2.x, so I chose okapi.)

python setup.py install

conda install numpy
pip could not be installed on Python3...
----
for python 2.7
conda install pip
conda install numpy
conda install pyqt=4

python sloth/bin/sloth examples/example1_labels.json