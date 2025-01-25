BUCKET="./local_data"
DATA_DIR="${BUCKET}/data"
MODEL_DIR="${BUCKET}/model"


nohup \
 trax train \
  --data_dir="${DATA_DIR}" \
  --trax_usr_dir=anime_chatbot/trainer \
  --problem=anime_chatbot_problem \
  --model=transformer \
  --output_dir="${MODEL_DIR}" \
  --train_steps=1000 \
  --hparams_set=transformer_tiny \
  --eval_steps=10 \
 > train.log 2>&1 &
