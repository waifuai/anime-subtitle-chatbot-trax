BUCKET_NAME="./local_data"  
DATA_DIR="${BUCKET_NAME}/data"
MODEL_DIR="${BUCKET_NAME}/model"
USR_DIR="anime_chatbot/trainer"
COMMON_ARGS=(
  "--data_dir=${DATA_DIR}"
  "--problem=anime_chatbot_problem"
  "--model=transformer"
  "--hparams_set=transformer_tiny"
  "--output_dir=${MODEL_DIR}"
  "--trax_usr_dir=${USR_DIR}" 
  "--decode_hparams=beam_size=4,alpha=0.6"
)

# Batch decoding on CPU
trax decode \ 
  "${COMMON_ARGS[@]}" \
  --decode_from_file=phrases_input.txt

# Interactive decoding on CPU
trax decode \ 
  "${COMMON_ARGS[@]}" \
  --decode_interactive
