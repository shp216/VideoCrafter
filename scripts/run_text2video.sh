name="base_512_v2"

ckpt='ckpt/model.ckpt'
config='configs/inference_t2v_512_v2.0.yaml'

prompt_file="prompts/test_prompts.txt"
res_dir="results"

python3 scripts/evaluation/inference.py \
--seed 123 \
--mode 'base' \
--ckpt_path $ckpt \
--config $config \
--savedir $res_dir/$name \
--n_samples 1 \
--bs 2 --height 128 --width 128 \
--unconditional_guidance_scale 12.0 \
--ddim_steps 1 \
--ddim_eta 1.0 \
--prompt_file $prompt_file \
--fps 28
