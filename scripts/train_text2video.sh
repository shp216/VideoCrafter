name="base_512_v2"

ckpt='ckpt/model.ckpt'
config='configs/inference_t2v_512_v2.0.yaml'
config2='configs/training_t2v_512_v2.0.yaml'


res_dir="predict_x0"

python3 scripts/evaluation/train_mmg.py \
--seed 123 \
--mode 'base' \
--ckpt_path $ckpt \
--config $config \
--config2 $config2 \
--savedir $res_dir \
--n_samples 1 \
--bs 2 --height 128 --width 128 \
--unconditional_guidance_scale 12.0 \
--ddim_steps 1 \
--ddim_eta 1.0 \
--fps 28 \
--output_dir "outputs" \
--max_train_steps 10 \
--train_batch_size 4 \