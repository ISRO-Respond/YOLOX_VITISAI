import numpy as np

DEBUG = False #True
PRINT_IMAGES = False #True

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

def decode_outputs(outputs, dtype,hw,self_strides=[8, 16, 32]):
        grids = []
        strides = []
        for (hsize, wsize), stride in zip(hw, self_strides):
            xv, yv = np.meshgrid(np.linspace(0,hsize-1,hsize), np.linspace(0,wsize-1,wsize))
            grid = np.reshape(np.stack((xv,yv),2),(1,-1,2))
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(np.ones((*shape,1)) * stride)

        grids = np.concatenate(grids, axis=1)
        strides = np.concatenate(strides, axis=1)

        outputs[..., :2] = (outputs[..., :2] + grids) * strides
        outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * strides
        return outputs

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

def prepare_postprocess(outputs,decode=True):

    outputs = [np.transpose(o,(0,3,1,2)) for o in outputs]
    hw = [x.shape[-2:] for x in outputs]

    outputs = np.transpose(np.concatenate([x.reshape(-1, x.shape[1], x.shape[2]*x.shape[2]) for x in outputs], axis=2), (0, 2, 1))
    outputs[...,4:] = Sigmoid1(outputs[..., 4:])

    if(decode):
        return decode_outputs(outputs,dtype=outputs[0].dtype,hw=hw)
    else:
        return outputs

def postprocess(prediction, num_classes, conf_thre=0.7, nms_thre=0.45, class_agnostic=False):
    # box_corner = prediction.new(prediction.shape)
    box_corner = prediction.copy()
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):
        # image_pred = np.array(image_pred)
        # If none are remaining => process next image
        image_pred = np.array(image_pred)
        if not image_pred.shape[0]:
            continue
        # Get score and class with highest confidence
        class_conf = image_pred[:, 5: 5 + num_classes].max(1, keepdims=True)
        class_pred =  np.expand_dims(np.argmax(np.array(image_pred[:, 5: 5 + num_classes]), 1), axis=1)

        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = np.concatenate((image_pred[:, :5], class_conf, class_pred), 1)
        detections = detections[conf_mask]
        if not detections.shape[0]:
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
            output[i] = np.concatenate((output[i], detections))

    return output