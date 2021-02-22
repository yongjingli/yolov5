"""Exports a YOLOv5 *.pt model to ONNX and TorchScript formats

Usage:
    $ export PYTHONPATH="$PWD" && python models/export.py --weights ./weights/yolov5s.pt --img 640 --batch 1
"""

import argparse
import sys
import time

sys.path.append('./')  # to run '$ python *.py' files in subdirectories

import torch
import torch.nn as nn

import models
from models.experimental import attempt_load
from utils.activations import Hardswish, SiLU
from utils.general import set_logging, check_img_size


import numpy as np
from models.common_self import FocusNew, Upsample
from models.common import Conv
from utils.torch_utils import fuse_conv_and_bn


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./yolov5s.pt', help='weights path')  # from yolov5/models/
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='image size')  # height, width
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--remove-focus', action='store_true', help='remove focus')

    opt = parser.parse_args()
    opt.img_size *= 2 if len(opt.img_size) == 1 else 1  # expand
    print(opt)
    set_logging()
    t = time.time()

    # Load PyTorch model
    model = attempt_load(opt.weights, map_location=torch.device('cpu'))  # load FP32 model
    labels = model.names

    nl = model.model[-1].nl
    anchor_grid = model.model[-1].anchor_grid
    anchors = anchor_grid.view(nl, -1)

    # Checks
    gs = int(max(model.stride))  # grid size (max stride)
    opt.img_size = [check_img_size(x, gs) for x in opt.img_size]  # verify img_size are gs-multiples

    # Input
    img = torch.randn(opt.batch_size, 3, *opt.img_size)  # image size(1,3,320,192) iDetection

    # Focus Remove
    REMOVE_FOCUS = True
    if REMOVE_FOCUS:
        print('Remove Focus...')
        old_focus = model.model[0]

        # new focus
        conv1_out_channels = 0
        for k, v in old_focus.named_parameters():
            if k == 'conv.conv.bias':
                conv1_out_channels = v.shape[0]

        assert conv1_out_channels > 0
        new_focus = FocusNew(3, conv1_out_channels, 3)

        # fuse conv and bn
        for m in new_focus.modules():
            if type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.fuseforward  # update forward

        tt = str(FocusNew)[8:-2].replace('__main__.', '')  # module type
        npp = sum([x.numel() for x in new_focus.parameters()])  # number params
        new_focus.i, new_focus.f, new_focus.type, new_focus.np = 0, -1, tt, npp  # attach index, 'from' index, type, number params

        # copy weights , include bn.moving.var ..
        focus_weights = dict()
        for param_tensor in old_focus.state_dict():
            focus_weights[param_tensor] = old_focus.state_dict()[param_tensor]

        # for k, v in model.named_parameters():
        #     if k == "model.0.conv.conv.weight":
        #         focus_weights["conv.conv.weight"] = v
        #     if k == "model.0.conv.bn.weight":
        #         focus_weights["conv.bn.weight"] = v
        #     if k == "model.0.conv.bn.weight":
        #         focus_weights["conv.bn.weight"] = v

        new_focus.load_state_dict(focus_weights, strict=True)
        model.model[0] = new_focus

        opt.img_size = [x//2 for x in opt.img_size]
        img = torch.zeros((opt.batch_size, 12, *opt.img_size))


    # Update model
    for k, m in model.named_modules():
        m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
        if isinstance(m, models.common.Conv):  # assign export-friendly activations
            if isinstance(m.act, nn.Hardswish):
                m.act = Hardswish()
            elif isinstance(m.act, nn.SiLU):
                m.act = SiLU()

        # friendly upsample
        if isinstance(m, nn.Upsample):
            m = Upsample(None, 2, 'nearest')

        # elif isinstance(m, models.yolo.Detect):
        #     m.forward = m.forward_export  # assign forward (optional)
    model.model[-1].export = True  # set Detect() layer export=True
    y = model(img)  # dry run

    # TorchScript export
    # try:
    #     print('\nStarting TorchScript export with torch %s...' % torch.__version__)
    #     f = opt.weights.replace('.pt', '.torchscript.pt')  # filename
    #     ts = torch.jit.trace(model, img)
    #     ts.save(f)
    #     print('TorchScript export success, saved as %s' % f)
    # except Exception as e:
    #     print('TorchScript export failure: %s' % e)

    # ONNX export
    try:
        import onnx

        print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
        f = opt.weights.replace('.pt', '.onnx')  # filename
        torch.onnx.export(model, img, f, verbose=False, opset_version=9, input_names=['images'],
                          output_names=['classes', 'boxes'] if y is None else ['output'])

        # Checks
        onnx_model = onnx.load(f)  # load onnx model
        onnx.checker.check_model(onnx_model)  # check onnx model
        # print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
        print('====ONNX export success, saved as %s' % f)


        # simpily onnx
        from onnxsim import simplify
        model_simp, check = simplify(onnx_model)
        assert check, "Simplified ONNX model could not be validated"

        f2 = f.replace('.onnx', '_sim.onnx')  # filename
        onnx.save(model_simp, f2)
        print('====ONNX SIM export success, saved as %s' % f2)


        # check output different between pytorch and onnx: y, y_onnx
        import onnxruntime as rt
        input_all = [node.name for node in onnx_model.graph.input]
        input_initializer = [node.name for node in onnx_model.graph.initializer]
        net_feed_input = list(set(input_all) - set(input_initializer))
        assert (len(net_feed_input) == 1)
        sess = rt.InferenceSession(f2)
        y_onnx = sess.run(None, {net_feed_input[0]: img.detach().numpy()})

        for i, (_y, _y_onnx) in enumerate(zip(y, y_onnx)):
            _y_numpy = _y.detach().numpy()
            # all_close = np.allclose(_y_numpy, _y_onnx, rtol=1e-05, atol=1e-06)
            diff = _y_numpy - _y_onnx
            print('output {}:, max diff {}'.format(i, np.max(diff)))
            # assert(np.max(diff) > 1e-5)


        from onnx import shape_inference
        f3 = f2.replace('.onnx', '_shape.onnx')  # filename
        onnx.save(onnx.shape_inference.infer_shapes(onnx.load(f2)), f3)
        print('====ONNX shape inference export success, saved as %s' % f3)

    except Exception as e:
        print('ONNX export failure: %s' % e)
        
    print('anchors"', anchors)
    # CoreML export
    # try:
    #     import coremltools as ct
    #
    #     print('\nStarting CoreML export with coremltools %s...' % ct.__version__)
    #     # convert model from torchscript and apply pixel scaling as per detect.py
    #     model = ct.convert(ts, inputs=[ct.ImageType(name='image', shape=img.shape, scale=1 / 255.0, bias=[0, 0, 0])])
    #     f = opt.weights.replace('.pt', '.mlmodel')  # filename
    #     model.save(f)
    #     print('CoreML export success, saved as %s' % f)
    # except Exception as e:
    #     print('CoreML export failure: %s' % e)

    # Finish
    print('\nExport complete (%.2fs). Visualize with https://github.com/lutzroeder/netron.' % (time.time() - t)) 
