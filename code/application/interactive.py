from cgi import test
from ctypes import *
from multiprocessing.spawn import prepare
from typing import List
import cv2
import numpy as np
import xir
from code.application.subgraph import prepare_postprocess
import vart
import os
import time

from general.preproc import *
from general.exec import *
from general.postproc import *

img_path = 'sign.jpg'
test_size = [640,640]
img = cv2.imread(img_path)
prepd_img, _ = preproc(img,test_size)

model_name = 'YOLOX_zcu104.xmodel'
g = xir.Graph.deserialize(model_name)
subgraphs = g.get_root_subgraph().toposort_child_subgraph()

dpu_subgraph0 = subgraphs[0]
dpu_subgraph1 = subgraphs[1]

print("dpu_subgraph0 = " + dpu_subgraph0.get_name()) 
print("dpu_subgraph1 = " + dpu_subgraph1.get_name())

dpu_1 = vart.Runner.create_runner(dpu_subgraph1, "run")

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

imgs = []
imgs.append(prepd_img)

inputData = [np.empty(input_1_ndim, dtype=np.float32, order="C")]
imageRun = inputData[0]
imageRun[0, ...] = imgs[0].reshape(input_1_ndim[1:])

execute_async(dpu_1, {"YOLOX__YOLOX_QuantStub_quant_in__input_1_fix": inputData[0],"YOLOX__YOLOX_YOLOXHead_head__Cat_cat_list__ModuleList_0__24408_fix": output1,"YOLOX__YOLOX_YOLOXHead_head__Cat_cat_list__ModuleList_1__24605_fix": output2,"YOLOX__YOLOX_YOLOXHead_head__Cat_cat_list__ModuleList_2__24802_fix": output3})

outputs = [output1,output2,output3]

o1 = prepare_postprocess(outputs)
o2 = postprocess(o1,45)