# traffic_lights_detection_model_training
For CarND-Capstone

Where the pth file should be put
```bash
>>> import sys
>>> sys.path
['', 'C:\\Users\\yoshi\\Anaconda3\\envs\\carnd-term3-p3\\python35.zip', 'C:\\Users\\yoshi\\Anaconda3\\envs\\carnd-term3-p3\\DLLs', 'C:\\Users\\yoshi\\Anaconda3\\envs\\carnd-term3-p3\\lib', 'C:\\Users\\yoshi\\Anaconda3\\envs\\carnd-term3-p3', 'C:\\Users\\yoshi\\Anaconda3\\envs\\carnd-term3-p3\\lib\\site-packages']
```

tf_object_detection.pth
```bash
C:/Work_BigData/tensorflow_models/models/research
C:/Work_BigData/tensorflow_models/models/research/slim
C:/Work_BigData/tensorflow_models/models/research/object_detection
```

command
```bash
python -m object_detection.model_main  --pipeline_config_path=models/model/faster_rcnn_inception_v2_coco.config  --model_dir=models/model
```

```bash
tensorboard --logdir=models/model
```

create frozen graph
```bash
python -m object_detection.export_inference_graph --input_type image_tensor --pipeline_config_path faster_rcnn_inception_v2_coco.config --trained_checkpoint_prefix model.ckpt-97 --output_directory exported_graphs
```

```bash
python -m object_detection.export_inference_graph --input_type image_tensor --pipeline_config_path faster_rcnn_inception_v2_coco-full.config --trained_checkpoint_prefix model.ckpt-400 --output_directory exported_graphs
```

```bash
python -m object_detection.export_inference_graph --input_type image_tensor --pipeline_config_path faster_rcnn_inception_v2_coco-sim.config --trained_checkpoint_prefix model.ckpt-866 --output_directory exported_graphs
```



