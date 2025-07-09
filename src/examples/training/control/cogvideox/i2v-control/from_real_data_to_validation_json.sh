#example use
python src/examples/training/control/cogvideox/i2v-control/from_real_data_to_validation_json.py \
  --images-dir dataset/demo_files/images \
  --sketches-dir dataset/demo_files/sketches \
  --images-txt dataset/demo_files/images.txt \
  --prompts-txt dataset/demo_files/prompts.txt \
  --sketches-txt dataset/demo_files/sketches.txt \
  --num-samples 5 \
  --output src/examples/training/control/cogvideox/i2v-control/example_validation.json