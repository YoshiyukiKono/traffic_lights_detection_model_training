# traffic_lights_detection_model_training
For CarND-Capstone

command
>python -m object_detection.model_main  --pipeline_config_path=models/model/faster_rcnn_inception_v2_coco.config  --model_dir=models/model

>tensorboard --logdir=models/model

create frozen graph
>python -m object_detection.export_inference_graph --input_type image_tensor --pipeline_config_path faster_rcnn_inception_v2_coco.config --trained_checkpoint_prefix model.ckpt-97 --output_directory exported_graphs

>python -m object_detection.export_inference_graph --input_type image_tensor --pipeline_config_path faster_rcnn_inception_v2_coco-full.config --trained_checkpoint_prefix model.ckpt-400 --output_directory exported_graphs