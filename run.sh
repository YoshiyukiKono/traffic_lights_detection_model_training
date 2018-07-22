# From the tensorflow/models/research/ directory
PIPELINE_CONFIG_PATH={/home/student/github/traffic_lights_detection_model_training/models/model/faster_rcnn.config}
MODEL_DIR={/home/student/github/traffic_lights_detection_model_training/models/model/}
NUM_TRAIN_STEPS=500
NUM_EVAL_STEPS=20
python /home/student/github/models/research/object_detection/model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --num_train_steps=${NUM_TRAIN_STEPS} \
    --num_eval_steps=${NUM_EVAL_STEPS} \
    --alsologtostderr
