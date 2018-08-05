# traffic_lights_detection_model_training
For CarND-Capstone

### Environment Preparation
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

### Command: tensorflow/models as of July, 2018 (not for the Capstone project)
```bash
python -m object_detection.model_main  --pipeline_config_path=models/model/faster_rcnn_inception_v2_coco.config  --model_dir=models/model
```

tensorboard

```bash
tensorboard --logdir=models/model
```

### Command: tensorflow/models r1.5 (for the Capstone project)

```bash
python -m object_detection.train --logtostderr --pipeline_config_path=models/model-ssd-sim/ssd_inception_v2_coco_ssd_sim.config  --train_dir=models/model-ssd-sim
```

```bash
python -m object_detection.train --logtostderr --pipeline_config_path=models/model-ssd-real/ssd_inception_v2_coco_ssd_real.config  --train_dir=models/model-ssd-real
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

```bash
cd model\model-ssd-sim-o
python -m object_detection.export_inference_graph --input_type image_tensor --pipeline_config_path ssd_inception_v2_coco_ssd_sim_o.config --trained_checkpoint_prefix model.ckpt-20000 --output_directory exported_graphs

```

```bash
cd model\model-ssd-real
python -m object_detection.export_inference_graph --input_type image_tensor --pipeline_config_path ssd_inception_v2_coco_ssd_real.config --trained_checkpoint_prefix model.ckpt-20000 --output_directory exported_graphs

```
### Relevant Issues

TF1.4 was needed to generate a graph for the project
https://github.com/tensorflow/models/issues/2777

exporter.py must be modified
https://github.com/tensorflow/models/issues/2777
```python
#      rewrite_options = rewriter_config_pb2.RewriterConfig(
#          layout_optimizer=rewriter_config_pb2.RewriterConfig.ON)
# https://github.com/tensorflow/models/issues/2861 (frostell commented on Jan 12)
      rewrite_options = rewriter_config_pb2.RewriterConfig(
          optimize_tensor_layout=rewriter_config_pb2.RewriterConfig.ON)
```
