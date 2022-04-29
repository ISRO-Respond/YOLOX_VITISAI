from ctypes import *
from typing import List
import cv2
import numpy as np
import xir
import vart
import os
import math
#import threading
import time
import sys
import queue
#from hashlib import md5
import argparse


DEBUG = False #True
PRINT_IMAGES = False #True

BUF_SIZE = 10
imgQ = queue.Queue(BUF_SIZE)
outQ = queue.Queue(BUF_SIZE)

cifar2_classes = ["automobile", "truck"]

def nms_single(boxes, scores, thresh):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

# Repeatedly call nms_single for each class detected
def nms_batch(boxes, scores, idxs, iou_threshold):
    keep_mask = np.zeros(scores.shape,dtype=np.bool)

    for class_id in np.unique(idxs):
        curr_indices = np.where(idxs == class_id)[0]
        curr_keep_indices = nms_single(boxes[curr_indices], scores[curr_indices], iou_threshold)
        keep_mask[curr_indices[curr_keep_indices]] = True
    keep_indices = np.where(keep_mask)[0]
    # return keep_indices[(scores[keep_indices].sort()[1])[::-1]]
    return keep_indices[np.argsort(scores[keep_indices])[::-1]]

def decode_outputs(self, outputs, dtype,hw,self_strides=[8, 16, 32]):
        grids = []
        strides = []
        for (hsize, wsize), stride in zip(hw, self_strides):
            xv, yv = np.meshgrid(np.linspace(0,hsize-1,hsize), np.linspace(0,wsize-1,wsize))
            grid = np.reshape(np.stack((xv,yv),2),(1,-1,2))
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(np.ones((*shape,1)) * stride)

        grids = np.concatenate(grids, axis=1, dtype=dtype)
        strides = np.concatenate(strides, axis=1, dtype=dtype)

        outputs[..., :2] = (outputs[..., :2] + grids) * strides
        outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * strides
        return outputs

def prepare_postprocess(outputs,decode=True):

    outputs = np.transpose(outputs,(0,2,3,1))
    hw = [x.shape[-2:] for x in outputs]

    outputs = np.transpose(np.concatenate([x.flatten(start_dim=2) for x in outputs], axis=2), (0, 2, 1))
    outputs[...,4:] = Sigmoid1(outputs[..., 4:])

    if(decode):
        return decode_outputs(outputs,dtype=outputs[0].type(),hw=hw)
    else:
        return outputs

def postprocess(prediction, num_classes, conf_thre=0.7, nms_thre=0.45, class_agnostic=False):
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = image_pred[:, 5: 5 + num_classes].max(1, keepdim=True)

        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = np.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
        detections = detections[conf_mask]
        if not detections.size(0):
            continue

        if class_agnostic:
            nms_out_index = nms_single(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                nms_thre,
            )
        else:
            nms_out_index = nms_batch(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                detections[:, 6],
                nms_thre,
            )

        detections = detections[nms_out_index]
        if output[i] is None:
            output[i] = detections
        else:
            output[i] = np.cat((output[i], detections))

    return output

def CPUCalcArgmax(data):
    '''
    returns index of highest value in data
    '''
    val = np.argmax(data)
    return val


def preprocess_fn(image_path):
    '''
    Image pre-processing.
    Opens image as grayscale then normalizes to range 0:1
    input arg: path of image file
    return: numpy array
    '''
    image = cv2.imread(image_path)
    #image = image.reshape(32,32,3)
    data = np.asarray( image, dtype="float32" )
    data = data/255.0
    return data

def Sigmoid1(xx):
    x = np.asarray( xx, dtype="float32")
    t = 1 / (1 + np.exp(-x))
    if DEBUG:
        print("SIGM1 inp shape ", x.shape)
        np.save('sigm1_data_inp.bin', x[0])
        #print("SIGM1 inp: ", x)
        #print("SIGM1 out: ", t)
        np.save('sigm1_data_out.bin', t[0])
    return t

def fix2float(fix_point, value):
    return value.astype(np.float32) * np.exp2(fix_point, dtype=np.float32)


def float2fix(fix_point, value):
    return value.astype(np.float32) / np.exp2(fix_point, dtype=np.float32)


def execute_async(dpu, tensor_buffers_dict):
    input_tensor_buffers = [
        tensor_buffers_dict[t.name] for t in dpu.get_input_tensors()
    ]
    output_tensor_buffers = [
        tensor_buffers_dict[t.name] for t in dpu.get_output_tensors()
    ]
    jid = dpu.execute_async(input_tensor_buffers, output_tensor_buffers)
    return dpu.wait(jid)

def DEBUG_runDPU(dpu_1):
    print("Start DPU DEBUG with 1 input image")
    # get DPU input/output tensors
    inputTensor_1  = dpu_1.get_input_tensors()
    outputTensor_1 = dpu_1.get_output_tensors()
    inputTensor_3  = dpu_3.get_input_tensors()
    outputTensor_3 = dpu_3.get_output_tensors()
    inputTensor_5  = dpu_5.get_input_tensors()
    outputTensor_5 = dpu_5.get_output_tensors()
    input_1_ndim  = tuple(inputTensor_1[0].dims)
    input_3_ndim  = tuple(inputTensor_3[0].dims)
    input_5_ndim  = tuple(inputTensor_5[0].dims)
    output_1_ndim = tuple(outputTensor_1[0].dims)
    output_3_ndim = tuple(outputTensor_3[0].dims)
    output_5_ndim = tuple(outputTensor_5[0].dims)
    batchSize = input_1_ndim[0]

    out1 = np.zeros([batchSize, 32, 32, 16], dtype='float32')
    out3 = np.zeros([batchSize, 16, 16,  8], dtype='float32')
    out5 = np.zeros([batchSize, 32        ], dtype='float32')

    if DEBUG :
        print(" inputTensor1={}\n".format( inputTensor_1[0]))
        print("outputTensor1={}\n".format(outputTensor_1[0]))
        print(" inputTensor3={}\n".format( inputTensor_3[0]))
        print("outputTensor3={}\n".format(outputTensor_3[0]))
        print(" inputTensor5={}\n".format( inputTensor_5[0]))
        print("outputTensor5={}\n".format(outputTensor_5[0]))

    if not imgQ.empty():
        img_org = imgQ.get()
        # run DPU
        execute_async(
            dpu_1, {
                "CNN__input_0_fix": img_org,
                "CNN__CNN_Conv2d_conv1__18_fix": out1
            })
        inp2 = out1.copy()
        out2 = Tanh(inp2)
        print("out2 shape ", out2.shape)
        # run DPU
        execute_async(
            dpu_3, {
                "CNN__CNN_Tanh_act1__19_fix": out2,
                "CNN__CNN_Conv2d_conv2__35_fix": out3
            })
        inp4 = out3.copy()
        out4 = Sigmoid1(inp4)
        print("out4 shape ", out4.shape)
        # run DPU
        execute_async(
            dpu_5, {
                "CNN__CNN_Sigmoid_act2__36_fix": out4,
                "CNN__CNN_Linear_fc1__48_fix":   out5
            })
        inp6 = out5.copy()
        out6 = Sigmoid2(inp6)
        print("out6 shape ", out6.shape)
        cnn_out = Linear(out6)
        prediction_index = CPUCalcArgmax(cnn_out) #(outputData[0][j])
        print("DEBUG DONE")


def runDPU(dpu_1, img):
    # get DPU input/output tensors
    inputTensor_1  = dpu_1.get_input_tensors()
    outputTensor_1 = dpu_1.get_output_tensors() 
    input_1_ndim  = tuple(inputTensor_1[0].dims) # [1, 640, 640, 3]
    output_1_ndim = tuple(outputTensor_1[0].dims) # [1, 80, 80, 50]
    output_2_ndim = tuple(outputTensor_1[1].dims) # [1, 40, 40, 50]
    output_3_ndim = tuple(outputTensor_1[2].dims) # [1, 20, 20, 50]

    batchSize = input_1_ndim[0]

    output1 = np.zeros([batchSize, 80, 80, 50], dtype='float32')
    output2 = np.zeros([batchSize, 40, 40, 50], dtype='float32')
    output3 = np.zeros([batchSize, 20, 20, 50], dtype='float32')

    n_of_images = len(img)
    count = 0
    write_index = 0
    while count < n_of_images:
        if (count+batchSize<=n_of_images):
            runSize = batchSize
        else:
            runSize=n_of_images-count
        '''prepare batch input/output '''
        outputData = []
        inputData = []
        inputData = [np.empty(input_1_ndim, dtype=np.float32, order="C")]
        #outputData = [np.empty(output_5_ndim, dtype=np.float32, order="C")]
        '''init input image to input buffer '''
        for j in range(runSize):
            imageRun = inputData[0]
            imageRun[j, ...] = img[(count + j) % n_of_images].reshape(input_1_ndim[1:])
        '''run with batch '''
        # run DPU
        execute_async(
            dpu_1, {
                "YOLOX__YOLOX_QuantStub_quant_in__input_1_fix": inputData[0],
                "YOLOX__YOLOX_YOLOXHead_head__Cat_cat_list__ModuleList_0__24408_fix": output1,
                "YOLOX__YOLOX_YOLOXHead_head__Cat_cat_list__ModuleList_1__24605_fix": output2,
                "YOLOX__YOLOX_YOLOXHead_head__Cat_cat_list__ModuleList_2__24802_fix": output3
            })
        # inp2 = out1.copy()
        # out2 = Tanh(inp2)
        
        # #print("out6 shape ", out6.shape)
        # cnn_out = Linear(out6)
        '''store output vectors '''
        for j in range(runSize):
            # out_q[write_index] = CPUCalcArgmax(cnn_out[j]) #(outputData[0][j])
            out_q[write_index] = np.cat()
            write_index += 1
        count = count + runSize


#def app(images_dir,threads,model_name):
def app(images_dir,model_name):

    images_list=os.listdir(images_dir)
    runTotal = len(images_list)
    print('Found',len(images_list),'images - processing',runTotal,'of them')

    ''' global list that all threads can write results to '''
    global out_q
    out_q = [None] * runTotal

    ''' get a list of subgraphs from the compiled model file '''
    g = xir.Graph.deserialize(model_name)
    subgraphs = g.get_root_subgraph().toposort_child_subgraph()
    dpu_subgraph0 = subgraphs[0]
    dpu_subgraph1 = subgraphs[1]
    if DEBUG:
        print("dpu_subgraph0 = " + dpu_subgraph0.get_name()) #subgraph_YOLOX__YOLOX_QuantStub_quant_in__input_1
        print("dpu_subgraph1 = " + dpu_subgraph1.get_name()) #subgraph_YOLOX__YOLOX_YOLOPAFPN_backbone__BaseConv_bu_conv1__Conv2d_conv__input_222
        
    dpu_1 = vart.Runner.create_runner(dpu_subgraph1, "run")

    ''' DEBUG with 1 input image '''
    if DEBUG:
        dbg_img = []
        path = "./test/img.png"
        dbg_img.append(preprocess_fn(path))
        imgQ.put(dbg_img[0])
        DEBUG_runDPU(dpu_1)
        return

    ''' Pre Processing images '''
    print("Pre-processing ",runTotal," images")
    img = []
    for i in range(runTotal):
        path = os.path.join(images_dir,images_list[i])
        img.append(preprocess_fn(path))

    ''' DPU execution '''
    print("run DPU")
    start=0
    end = len(img)
    in_q = img[start:end]
    time1 = time.time()
    runDPU(dpu_1,img)
    time2 = time.time()
    timetotal = time2 - time1
    fps = float(runTotal / timetotal)
    print(" ")
    print("FPS=%.2f, total frames = %.0f , time=%.4f seconds" %(fps,runTotal, timetotal))
    print(" ")

    ''' Post Processing '''
    print("Post-processing")
    classes = cifar2_classes
    correct = 0
    wrong = 0
    for i in range(len(out_q)):
        prediction = classes[out_q[i]]
        ground_truth, _ = images_list[i].split("_", 1)
        if PRINT_IMAGES:
            print("image number ", i, ": ", images_list[i])
            inp_img  = in_q[i] * 255.0
            cv2.imshow(images_list[i], np.uint8(inp_img));
            cv2.waitKey(1000);
            print("predicted: ", prediction, " ground Truth ", ground_truth)
        if (ground_truth==prediction):
            correct += 1
        else:
            wrong += 1
    accuracy = correct/len(out_q)
    print("Correct: ",correct," Wrong: ",wrong," Accuracy: ", accuracy)
    return



# only used if script is run as 'main' from command line
def main():

  # construct the argument parser and parse the arguments
  ap = argparse.ArgumentParser()
  ap.add_argument('-d', '--images_dir', type=str, default='../test_images', help='Path to folder of images. Default is images')
  #ap.add_argument('-t', '--threads',    type=int, default=1,        help='Number of threads. Default is 1')
  ap.add_argument('-m', '--model',      type=str, default='./CNN_int_vck190_dw.xmodel', help='Path of xmodel. Default is CNN_zcu102.xmodel')

  args = ap.parse_args()
  print("\n")
  print ('Command line options:')
  print (' --images_dir : ', args.images_dir)
  #print (' --threads    : ', args.threads)
  print (' --model      : ', args.model)
  print("\n")

  #app(args.images_dir,args.threads,args.model)
  app(args.images_dir,args.model)



if __name__ == '__main__':
  main()
