1.# 修改数据集的加载路径
    # train_path = data_dict['train']
    # test_path = data_dict['val']

    # train_path = [data_dict['train'], data_dict['wider_person_train'], data_dict['crowd_person_train'],
    #               data_dict['local1111_train'], data_dict['exdark_train'], data_dict['background']]

    # train_path = [data_dict['part_train'], data_dict['part_wider_person_train'], data_dict['part_crowd_person_train'],
    #               data_dict['part_local1111_train'], data_dict['background'], data_dict['part_exdark_train']]
    #
    # test_path = [data_dict['val'], data_dict['wider_person_val'], data_dict['crowd_person_val'],
    #              data_dict['local1111_val'], data_dict['exdark_val']]


    train_path = [data_dict['part_wider_person_train'], data_dict['background']]
    test_path = [data_dict['local1111_val'], data_dict['exdark_val']]

    nc = 1 if opt.single_cls else int(data_dict['nc'])  # number of classes
    
2.#将wandb的mode改为offline
    if rank in [-1, 0] and wandb and wandb.run is None:
        opt.hyp = hyp  # add hyperparameters
        wandb_run = wandb.init(config=opt, resume="allow",
                               project='YOLOv5' if opt.project == 'runs/train' else Path(opt.project).stem,
                               name=save_dir.stem,
                               id=ckpt.get('wandb_id') if 'ckpt' in locals() else None, mode="offline")
    loggers = {'wandb': wandb}  # loggers dict
