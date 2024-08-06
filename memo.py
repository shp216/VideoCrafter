def train_mmg(args, gpu_num, gpu_no, **kwargs):
    ## step 1: 모델 설정
    ## -----------------------------------------------------------------
    config = OmegaConf.load(args.config)
    model_config = config.pop("model", OmegaConf.create())

    mmg_model = instantiate_from_config(model_config)
    teacher_model = instantiate_from_config(model_config)

    # 동적 GPU 할당
    mmg_model = mmg_model.cuda(gpu_no)
    teacher_model = teacher_model.cuda((gpu_no + 1) % gpu_num)

    assert os.path.exists(args.ckpt_path), f"Error: checkpoint [{args.ckpt_path}] Not Found!"
    mmg_model = load_model_checkpoint(mmg_model, args.ckpt_path)
    teacher_model = load_model_checkpoint(teacher_model, args.ckpt_path)

    mmg_model.train()
    teacher_model.eval()
    
    # Logging 폴더 설정
    name = Path(args.config).stem
    folder_name = "debug" if args.is_debug else name + datetime.datetime.now().strftime("-%Y-%m-%dT%H-%M-%S")
    output_dir = os.path.join(args.output_dir, folder_name)
    if args.is_debug and os.path.exists(output_dir):
        os.system(f"rm -rf {output_dir}")

    *_, config = inspect.getargvalues(inspect.currentframe())
    
    # 설정 로깅
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    run = wandb.init(project="MMG_videocrafter", name=folder_name, config=config)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/samples", exist_ok=True)
    os.makedirs(f"{output_dir}/sanity_check", exist_ok=True)
    os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
    
    # 학습 파라미터 설정
    trainable_params = list(filter(lambda p: p.requires_grad, mmg_model.parameters()))

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    
    num_trainable_params = sum(p.numel() for p in trainable_params)
    print(f"Number of trainable parameters: {num_trainable_params}")
    
    num_all_params = sum(p.numel() for p in mmg_model.parameters())
    print(f"Number of mmg_model parameters: {num_all_params}")
    
    train_dataset = WebVid10M(
        csv_path="scripts/evaluation/data/audiocaps.csv",
        video_folder="scripts/evaluation/data/test_trimmed_audiocaps",
        sample_size=256,
        sample_stride=4, sample_n_frames=16,
        is_image=False,
    )    
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
    )
    
    # 스케줄러 설정
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    
    logging.info("***** Running training *****")
    logging.info(f"  Num examples = {len(train_dataset)}")
    logging.info(f"  Num Epochs = {num_train_epochs}")
    logging.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logging.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logging.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0
    
    progress_bar = tqdm(range(global_step, args.max_train_steps))
    progress_bar.set_description("Steps")
    
    for epoch in range(first_epoch, num_train_epochs):
        mmg_model.train()
        for step, batch in enumerate(train_dataloader):
            if args.cfg_random_null_text:
                batch['text'] = [name if random.random() > args.cfg_random_null_text_ratio else "" for name in batch['text']]
                
            # 데이터 배치 체크
            if step == 0:
                pixel_values, texts = batch['pixel_values'], batch['text']
                
                for idx, (pixel_value, text) in enumerate(zip(pixel_values, texts)):
                    pixel_value = pixel_value[None, ...]
                    save_videos_grid(pixel_value, f"{output_dir}/sanity_check/{'-'.join(text.replace('/', '').split()[:10]) if not text == '' else f'{step}-{idx}'}.gif", rescale=True)
            
            ### >>>> Training >>>> ###
                
            # 비디오를 잠재 공간으로 변환
            pixel_values = batch["pixel_values"].cuda(gpu_no)
            video_length = pixel_values.shape[1]
            with torch.no_grad():
                latents = mmg_model.encode_first_stage_2DAE(pixel_values)

            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, 1000, (bsz,)).long()
            noisy_latents, GT_noise = mmg_model.q_sample(latents, timesteps)
            
            with torch.no_grad():
                c_emb_mmg = mmg_model.get_learned_conditioning(batch["text"])

            with torch.no_grad():
                c_emb_teacher = teacher_model.get_learned_conditioning(batch["text"]).cuda((gpu_no + 1) % gpu_num)
            
            mmg_output = mmg_model.apply_model(noisy_latents, timesteps, c_emb_mmg).cuda(gpu_no)
            teacher_output = teacher_model.apply_model(noisy_latents.cuda((gpu_no + 1) % gpu_num), timesteps.cuda((gpu_no + 1) % gpu_num), c_emb_teacher).cuda(gpu_no)

            loss = F.mse_loss(mmg_output.float(), teacher_output.float(), reduction="mean")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mmg_model.parameters(), args.max_grad_norm)
            optimizer.step()
            
            lr_scheduler.step()
            progress_bar.update(1)
            global_step += 1
            
            ### <<<< Training <<<< ###
            
            if args.is_main_process and (not args.is_debug) and args.use_wandb:
                wandb.log({"train_loss": loss.item()}, step=global_step)
                
            if args.is_main_process and (global_step % args.checkpointing_steps == 0 or step == len(train_dataloader) - 1):
                save_path = os.path.join(output_dir, f"checkpoints")
                state_dict = {
                    "epoch": args.epoch,
                    "global_step": global_step,
                    "state_dict": mmg_model.state_dict(),
                }
                if step == len(train_dataloader) - 1:
                    torch.save(state_dict, os.path.join(save_path, f"checkpoint-epoch-{epoch+1}.ckpt"))
                else:
                    torch.save(state_dict, os.path.join(save_path, f"checkpoint.ckpt"))
                logging.info(f"Saved state to {save_path} (global_step: {global_step})")
                
            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            
            if global_step >= args.max_train_steps:
                break
