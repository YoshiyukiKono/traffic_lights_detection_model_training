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



@powershell -NoProfile -ExecutionPolicy unrestricted -Command "iex ((new-object net.webclient).DownloadString('https://chocolatey.org/install.ps1'))" && SET PATH=%PATH%;%systemdrive%\chocolatey\bin


Windows 10
Windows キー + X
A キー
Alt + Y
Windows キーと X キーを入力するとメニューが表示される。その一覧にコマンドプロンプトを管理者で実行する項目があるので A キーを入力することで起動できる。ユーザーアカウント制御のウィンドウが表示されるので Alt + Y で「はい(Y)」を選択する。そうすると管理者権限でコマンドプロンプトが起動される。

Windows 10 Creators Update
Windows キー
検索ボックスに cmd と入力
Ctrl + Shift + Enter
Alt + Y
Windows キーを押下するとスタートメニューが開く。そのまま cmd と入力すると、検索ボックスでコマンドプロンプトのプログラム (cmd.exe) が検索されて選択される。コマンドプロンプトのプログラム (cmd.exe) を選択している状態で Ctrl + Shift + Enter で開くと管理者権限で実行できる。ユーザーアカウント制御のウィンドウが表示されるので Alt + Y で「はい(Y)」を選択する。そうすると管理者権限でコマンドプロンプトが起動される。

※ Creators Update 以降の Windows 10 だと、以前の方法では PowerShell が起動してしまう。（コメントしていただいた @ktanakaj さんありがとうございます。）

PS C:\Windows\system32> cmd
Microsoft Windows [Version 10.0.16299.547]
(c) 2017 Microsoft Corporation. All rights reserved.

C:\Windows\system32>@powershell -NoProfile -ExecutionPolicy unrestricted -Command "iex ((new-object net.webclient).DownloadString('https://chocolatey.org/install.ps1'))" && SET PATH=%PATH%;%systemdrive%\chocolatey\bin
Getting latest version of the Chocolatey package for download.
Getting Chocolatey from https://chocolatey.org/api/v2/package/chocolatey/0.10.11.
Extracting C:\Users\yoshi\AppData\Local\Temp\chocolatey\chocInstall\chocolatey.zip to C:\Users\yoshi\AppData\Local\Temp\chocolatey\chocInstall...
Installing chocolatey on this machine
Creating ChocolateyInstall as an environment variable (targeting 'Machine')
  Setting ChocolateyInstall to 'C:\ProgramData\chocolatey'
WARNING: It's very likely you will need to close and reopen your shell
  before you can use choco.
Restricting write permissions to Administrators
We are setting up the Chocolatey package repository.
The packages themselves go to 'C:\ProgramData\chocolatey\lib'
  (i.e. C:\ProgramData\chocolatey\lib\yourPackageName).
A shim file for the command line goes to 'C:\ProgramData\chocolatey\bin'
  and points to an executable in 'C:\ProgramData\chocolatey\lib\yourPackageName'.

Creating Chocolatey folders if they do not already exist.

WARNING: You can safely ignore errors related to missing log files when
  upgrading from a version of Chocolatey less than 0.9.9.
  'Batch file could not be found' is also safe to ignore.
  'The system cannot find the file specified' - also safe.
chocolatey.nupkg file not installed in lib.
 Attempting to locate it from bootstrapper.
PATH environment variable does not have C:\ProgramData\chocolatey\bin in it. Adding...
WARNING: Not setting tab completion: Profile file does not exist at
'C:\Users\yoshi\OneDrive\ドキュメント\WindowsPowerShell\Microsoft.PowerShell_profile.ps1'.
Chocolatey (choco.exe) is now ready.
You can call choco from anywhere, command line or powershell by typing choco.
Run choco /? for a list of functions.
You may need to shut down and restart powershell and/or consoles
 first prior to using choco.
Ensuring chocolatey commands are on the path
Ensuring chocolatey.nupkg is in the lib folder

--

conda create yq
activate yq
pip install yq

(yq) C:\Users\yoshi>yq
usage: yq.exe [-h] [--yaml-output] [--width WIDTH] [--version]
              jq_filter [files [files ...]]
yq.exe: error: the following arguments are required: jq_filter, files