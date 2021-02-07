import torch
import argparse
import sys
import os
import numpy as np
import time
import cv2

sys.path.append('./')  # to run '$ python *.py' files in subdirectories


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./yolov5s.pt', help='weights path')  # from yolov5/models/
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='image size')  # height, width
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--model_dir', default="./", help='the directory where checkpoints are saved')
    parser.add_argument('--starting_model_id', default=0, type=int, help='the id of the starting checkpoint for averaging, e.g. 1')
    parser.add_argument('--ending_model_id', default=1,  type=int, help='the id of the ending checkpoint for averaging, e.g. 12')
    opt = parser.parse_args()

    opt.model_dir = "/home/liyongjing/Egolee/programs/yolov5-master_2/runs/train/exp6/weights"
    opt.starting_model_id = 0
    opt.ending_model_id = 24

    model_dir = opt.model_dir
    starting_id = int(opt.starting_model_id)
    ending_id = int(opt.ending_model_id)
    model_names = list(range(starting_id, ending_id))
    print("model_names:", model_names)

    model_dirs = [os.path.join(model_dir, 'swa_ema_' + str(i) + '.pt') for i in model_names]
    # models = [attempt_load(model_dir, map_location=torch.device('cpu')) for model_dir in model_dirs]
    models = [torch.load(model_dir, map_location=torch.device('cpu'))['model'].float() for model_dir in model_dirs]
    model_num = len(models)
    print("model_num:", model_num)
    ref_model = models[-1]

    # for k, v in ref_model.state_dict().items():
    #     print(k)
    #     print(v.shape)

    # model_keys = models[-1]['state_dict'].keys()
    model_keys = models[-1].state_dict().keys()

    # state_dict = models[-1]['state_dict']
    state_dict = models[-1].state_dict()

    new_state_dict = state_dict.copy()

    for key in model_keys:
        sum_weight = 0.0
        for m in models:
            # sum_weight += m['state_dict'][key]
            sum_weight += m.state_dict()[key]
        avg_weight = sum_weight / model_num
        new_state_dict[key] = avg_weight

    # for key in model_keys:
    #     b_sum = torch.sum(ref_model.state_dict()[key])
    #     ref_model.state_dict()[key] = new_state_dict[key]
    #     a_sum = torch.sum(ref_model.state_dict()[key])
    #     diff = b_sum - a_sum
    #
    #     print('diff:', diff)
    ref_model.load_state_dict(new_state_dict, strict=True)  # load

    save_model_name = './weights/' + 'swa_ema_average' + str(opt.starting_model_id) + '_' + str(opt.ending_model_id) + '.pt'
    ckpt_swa = {'epoch': 0,
                'best_fitness': 0,
                'training_results': 0,
                'model': ref_model,
                'optimizer': None,
                'wandb_id':  None}

    torch.save(ckpt_swa, save_model_name)
