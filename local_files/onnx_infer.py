# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 10:10:16 

@author: liyongjing
"""
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import cv2
import numpy as np
import os
import onnxruntime as rt
from tqdm import tqdm


def nms(prediction, min_conf=0.25, nms_iou=0.45):
    def xywh2xyxy(x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = np.zeros_like(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

    nc = prediction[0].shape[1] - 5  # number of classes
    xc = prediction[..., 4] > min_conf  # candidates
    output = []
    # batch size = 1
    for xi, x in enumerate(prediction):  # image index, image inference
        x = x[xc[xi]]  # confidence filter
        if not x.shape[0]:
            output.append([])
            continue

        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
        box = xywh2xyxy(x[:, :4])

        # multi class
        if nc > 1:
            i, j = np.where(x[:, 5:] > min_conf)
            x = np.concatenate([box[i], x[i, j + 5, None], j[:, None].astype(np.float32)], 1)
        else:
            # best class only
            j = np.argmax(x[:, 5:], axis=1)
            conf = np.max(x[:, 5:], axis=1).astype(np.float32)
            x = np.concatenate([box, conf[:, None], j[:, None]], axis=1)[conf.reshape(-1) > min_conf]

        conf_sort = np.argsort(-x[:, 4], axis=0)  # sorted by confident
        x = x[conf_sort, :]
        skip = [False] * len(conf_sort)
        mask = [False] * len(conf_sort)
        for i in range(len(conf_sort)):
            if skip[i]:
                continue
            mask[i] = True
            box_high = x[i]
            for j in range(i + 1, len(conf_sort)):
                box_comp = x[j]
                if skip[j] or box_high[5] != box_comp[5]:
                    continue
                xx1 = max(box_high[0], box_comp[0])
                yy1 = max(box_high[1], box_comp[1])
                xx2 = min(box_high[2], box_comp[2])
                yy2 = min(box_high[3], box_comp[3])
                iou_w = xx2 - xx1 + 1
                iou_h = yy2 - yy1 + 1
                if iou_h > 0 and iou_w > 0:
                    area_high = (box_high[2] - box_high[0]) * (box_high[3] - box_high[1])
                    area_comp = (box_comp[2] - box_comp[0]) * (box_comp[3] - box_comp[1])
                    over = iou_h * iou_w / (area_high + area_comp - iou_h * iou_w)
                    if over > nms_iou:
                        skip[j] = True
        x = x[mask, :]
        output.append(x)
    return output


def scale_boxes(boxes, pad_info=[1, 0, 0, 0, 0]):
    scale, pad_top, pad_bottom, pad_left, pad_right = pad_info
    for _box in boxes:
        _box[0] = int((_box[0] - pad_left) / scale)
        _box[2] = int((_box[2] - pad_left) / scale)
        _box[1] = int((_box[1] - pad_top) / scale)
        _box[3] = int((_box[3] - pad_top) / scale)
    return boxes


class YoloDetectionOnnx:
    def __init__(self, onnx_file, batch_size=1):
        print('YoloDetectionOnnx Initial Start...')
        self.onnx_file = onnx_file
        self.sess = rt.InferenceSession(onnx_file)
        self.input_name = self.sess.get_inputs()[0].name

        self.net_w = 640
        self.net_h = 640
        self.batch_size = batch_size

        # 3 stage
        # self.stride = [8., 16., 32.]
        # self.anchor = np.array([[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]], dtype=np.float32)

        # 4 stage
        self.stride = [8., 16., 32., 64.]
        # self.anchor = np.array([[9., 11., 21., 19., 17., 41.], [43., 32., 39., 70., 86., 64.],
        #                         [65., 131., 134., 130., 120., 265.], [282., 180., 247., 354., 512., 387.]], dtype=np.float32)

        self.anchor = np.array([[19, 27, 44, 40, 38, 94], [96, 68, 86, 152,  80, 137],
                                [140, 301, 303, 264, 238, 542 ], [436, 615, 739, 380, 925, 792 ]], dtype=np.float32)

        self.anchor_grid = self.anchor.reshape(len(self.stride), 1, -1, 1, 1, 2)
        self.max_det = 300

        self.min_conf = 0.25
        self.nms_iou = 0.45
        self.remove_focus = True
        print('YoloDetectionOnnx Initial Done...')

    def infer_cv_img(self, cv_img):
        img_input, pad_info = self.img_pre_process(cv_img)
        prediction = self.get_onnx_prediction(img_input)
        pred_boxes = nms(prediction, self.min_conf, self.nms_iou)[0]
        pred_boxes = scale_boxes(pred_boxes, pad_info)
        return pred_boxes

    def infer_cv_img_diff_conf(self, cv_img, diff_conf):
        img_input, pad_info = self.img_pre_process(cv_img)
        prediction = self.get_onnx_prediction(img_input)
        pred_boxes_diff_conf = []
        for confidence in diff_conf:
            pred_boxes = nms(prediction, confidence, self.nms_iou)[0]
            pred_boxes = scale_boxes(pred_boxes, pad_info)
            pred_boxes_diff_conf.append(pred_boxes)
        return pred_boxes_diff_conf

    def infer_batch_cv_imgs(self, cv_imgs):
        assert self.batch_size == len(cv_imgs)
        img_inputs = []
        pad_infos = []
        for cv_img in cv_imgs:
            img_input, pad_info = self.img_pre_process(cv_img)
            img_inputs.append(img_input)
            pad_infos.append(pad_info)

        img_inputs = np.concatenate(img_inputs, axis=0)
        batch_prediction = self.get_onnx_prediction(img_inputs)

        batch_pred_boxes = nms(batch_prediction, self.min_conf, self.nms_iou)
        for i, pred_boxes in enumerate(batch_pred_boxes):
            pad_info = pad_infos[i]
            batch_pred_boxes[i] = scale_boxes(pred_boxes, pad_info)

        return batch_pred_boxes

    def infer_batch_cv_imgs_diff_conf(self, cv_imgs, diff_conf):
        assert self.batch_size == len(cv_imgs)
        img_inputs = []
        pad_infos = []
        for cv_img in cv_imgs:
            img_input, pad_info = self.img_pre_process(cv_img)
            img_inputs.append(img_input)
            pad_infos.append(pad_info)

        img_inputs = np.concatenate(img_inputs, axis=0)
        batch_prediction = self.get_onnx_prediction(img_inputs)
        batch_pred_boxes_diff_conf = []
        for confidence in diff_conf:
            batch_pred_boxes = nms(batch_prediction, confidence, self.nms_iou)
            for i, pred_boxes in enumerate(batch_pred_boxes):
                pad_info = pad_infos[i]
                batch_pred_boxes[i] = scale_boxes(pred_boxes, pad_info)
            batch_pred_boxes_diff_conf.append(batch_pred_boxes)
        return batch_pred_boxes_diff_conf

    def infer(self, img_input):
        img_input = np.concatenate((img_input[..., ::2, ::2], img_input[..., 1::2, ::2], img_input[..., ::2, 1::2],
                                    img_input[..., 1::2, 1::2]), 1).astype(np.float32)

        prediction = self.get_onnx_prediction(img_input)
        pred_boxes = nms(prediction, self.min_conf, self.nms_iou)[0]
        return pred_boxes

    def img_pre_process(self, cv_img):
        assert cv_img is not None
        h_scale = self.net_h/cv_img.shape[0]
        w_scale = self.net_w/cv_img.shape[1]

        scale = min(h_scale, w_scale)
        img_temp = cv2.resize(cv_img, (int(cv_img.shape[1] * scale), int(cv_img.shape[0] * scale)),
                              interpolation=cv2.INTER_LINEAR)

        # cal pad_w, and pad_h
        pad_h = (self.net_h - img_temp.shape[0]) // 2
        pad_w = (self.net_w - img_temp.shape[1]) // 2
        pad_top, pad_bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
        pad_left, pad_right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))

        img_input = np.ones((self.net_h, self.net_w, 3), dtype=np.uint8) * 114
        # img_input[self.pad_h:img_temp.shape[0] + self.pad_h, self.pad_w:img_temp.shape[1] + self.pad_w, :] = img_temp
        img_input[pad_top:img_temp.shape[0] + pad_bottom, pad_left:img_temp.shape[1] + pad_right, :] = img_temp

        # Convert
        img_input = img_input.astype(np.float32)
        img_input = img_input[:, :, ::-1]
        img_input /= 255.0
        img_input = img_input.transpose(2, 0, 1)  # to C, H, W
        img_input = np.ascontiguousarray(img_input)
        img_input = np.expand_dims(img_input, 0)

        # remove foucs
        if self.remove_focus:
            img_input = np.concatenate((img_input[..., ::2, ::2], img_input[..., 1::2, ::2], img_input[..., ::2, 1::2],
                                        img_input[..., 1::2, 1::2]), 1)
        return img_input, [scale, pad_top, pad_bottom, pad_left, pad_right]

    def get_onnx_prediction(self, input_img):
        feats = self.sess.run(None, {self.input_name: input_img})
        z = []
        for i, feat in enumerate(feats):
            bs, na, ny, nx, no = feat.shape
            yv, xv = np.meshgrid(np.arange(ny), np.arange(nx))
            grid = np.stack((xv, yv), 2).reshape((1, 1, ny, nx, 2)).astype(np.float32)
            grid = grid[..., ::-1]

            y = 1.0 / (1.0 + np.exp(-feat))  # sigmoid
            y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + grid) * self.stride[i]  # xy
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
            z.append(y.reshape(bs, -1, no))
        prediction = np.concatenate(z, axis=1)
        prediction = prediction.astype(np.float32)
        return prediction


def onnx_infer_video(onnx_path, video_path):
    # onnx_path = '/home/liyongjing/Egolee/programs/yolov5-master_3/weights/yolov5l6-640_sim.onnx'
    yolo_detector = YoloDetectionOnnx(onnx_path)

    # video_path = "/home/liyongjing/Egolee/data/test_person_car/001.avi"
    video_name = os.path.split(video_path)[-1].split(".")[0]
    output_dir = "/home/liyongjing/Egolee/data/test_person_car/get_diffcult_samples"

    cap = cv2.VideoCapture(video_path)
    rval = True
    cout = 0
    jump_frame = 5

    while rval:
        rval, img = cap.read()
        if not rval:
            break
        if cout % jump_frame == 0:
            det_boxes = yolo_detector.infer_cv_img(img)
            for det_box in det_boxes:
                box = det_box[0:4]
                score = det_box[4:5]
                cls = det_box[5:6]
                show_color = (0, 255, 0)
                if cls != 0:
                    show_color = (0, 0, 255)
                cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), show_color, 2)
                cv2.putText(img, str(round(score[0], 2)), (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            show_color, 1)
        cout += 1

        cv2.namedWindow("img", 0)
        cv2.imshow("img", img)
        wait_key = cv2.waitKey(1)
        if wait_key == 27:
            break
    cap.release()


def onnx_infer_images(onnx_path, image_path, batch_size=1):
    yolo_detector = YoloDetectionOnnx(onnx_path, batch_size)
    confidents = np.arange(1, 10)*0.1

    yolo_detector.batch_size = batch_size
    yolo_detector.min_conf = confidents[0]

    image_names = list(filter(lambda x: x[-3:] == "jpg", os.listdir(image_path)))
    for i in tqdm(range(0, len(image_names), batch_size)):
        img_names = [image_names[j] for j in range(i, i+batch_size) if j < len(image_names)]
        if len(img_names) != batch_size:
            continue

        imgs = [cv2.imread(os.path.join(image_path, img_name)) for img_name in img_names]
        batch_pred_boxes_diff_conf = yolo_detector.infer_batch_cv_imgs_diff_conf(imgs, confidents)
        assert len(batch_pred_boxes_diff_conf) == len(confidents)

        for j, confident in enumerate(confidents):
            print('confident:', confident)
            # batch_pred_boxes = yolo_detector.infer_batch_cv_imgs(imgs)
            batch_pred_boxes = batch_pred_boxes_diff_conf[j]
            assert len(imgs) == len(batch_pred_boxes)

            for k, img in enumerate(imgs):
                img_show = img.copy()
                det_boxes = batch_pred_boxes[k]

                # Do Something...
                for det_box in det_boxes:
                    box = det_box[0:4]
                    score = det_box[4:5]
                    cls = det_box[5:6]

                    if cls == 0:
                        show_color = (0, 255, 0)
                    else:
                        show_color = (0, 0, 255)

                    cv2.rectangle(img_show, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), show_color, 2)
                    cv2.putText(img_show, str(round(score[0], 2)), (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.8, show_color, 1)
                cv2.namedWindow("img_show" + str(k), 0)
                cv2.imshow("img_show" + str(k), img_show)

            wait_key = cv2.waitKey(0)
            if wait_key == 27:
                exit(1)


if __name__=="__main__":
    logger.info("Start Onnx Infer...")
    onnx_path = '/home/liyongjing/Egolee/programs/yolov5-master_3/weights/best_sim.onnx'
    image_path = "/home/liyongjing/Egolee/programs/yolov5-master_1/inference/images"
    batch_size = 2

    onnx_infer_images(onnx_path, image_path, batch_size)
    # onnx_infer_video()
    logger.info("End Onnx Infer...")
