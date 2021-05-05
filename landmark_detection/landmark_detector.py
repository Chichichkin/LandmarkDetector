import math
from pathlib import Path

import models
import torch
import numpy as np
import cv2 as cv



class LandmarkDetector(object):
    ''' Обёртка для SAN Face Landmark Detector

    Example:
        ```
        from landmark_detection.landmark_detector import LandmarkDetector
        det = LandmarkDetector()
        landmark,err = det.detect(image_path, face)
        ```
    '''

    def __init__(self, model_path="./snapshots/SAN_300W_GTB_itn_cpm_3_50_sigma4_128x128x8/checkpoint_49.pth.tar",
                 benchmark: bool = False):
        '''
        Args:
            module_path: Путь до предобученной модели - если есть необходимость её заменить
            benchmark: включить benchmark для CUDA
        '''
        self.model_path = model_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if benchmark and self.device == 'cuda':
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True

        snapshot = torch.load(self.model_path, map_location=self.device)
        self.param = snapshot['args']

        self.net = models.__dict__[self.param.arch](self.param.modelconfig, None)
        self.net.train(False).to(self.device)

        weights = models.remove_module_dict(snapshot['state_dict'])
        self.net.load_state_dict(weights)

    def detect(self, image, face):
        '''
        Args:
            image: Изображение, либо как путь до изображения, либо как np.ndarray
            face: [x1,y1,x2,y2] - область с лицом
        Returns:
            (
                словарь с полями landmarks ( numpy array размера 68 x 3 -
                координаты x, y и вероятность p для каждой из 68 точек) и
                error_message (строка с текстом ошибки, если не удалось извлечь
                лэндмарки, пустая - в ином случае)
            )
        '''
        # Проверка, в каком формати передаётся изображение
        if isinstance(image, str) or isinstance(image, Path):
            image = cv.imread(image)
        elif isinstance(image, np.ndarray):
            image = image.astype(np.uint8)
        else:
            return None, f'Unsupported input image type {type(image)}'

        tensor, temp_save_wh = self.preprocess(image, face)
        cropped_size = torch.IntTensor([temp_save_wh[1], temp_save_wh[0], temp_save_wh[2], temp_save_wh[3]])

        with torch.no_grad():
            inputs = tensor.unsqueeze(0).to(self.device)
            _, batch_locs, batch_scos, _ = self.net(inputs)

        np_batch_locs, np_batch_scos, cropped_size = batch_locs.cpu().numpy(), batch_scos.cpu().numpy(), cropped_size.numpy()
        locations, scores = np_batch_locs[0, :-1, :], np.expand_dims(np_batch_scos[0, :-1], -1)

        scale_h, scale_w = cropped_size[0] * 1. / inputs.size(-2), cropped_size[1] * 1. / inputs.size(-1)

        locations[:, 0], locations[:, 1] = locations[:, 0] * scale_w + cropped_size[2], locations[:, 1] * scale_h + \
                                           cropped_size[3]
        result = np.c_[locations.round(2), scores]
        return result, ""

    def preprocess(self, image: np.ndarray, face: list):
        '''
            Args:
                image: Изображение,  как np.ndarray
                face: [x1,y1,x2,y2] - область с лицом
            Returns:
                (
                    готовый для подачи в сеть тензор
                    список с начальными шириной и высотой, а также
                )
                '''
        w, h, _ = image.shape
        box = face.copy()
        expand_ratio = self.param.pre_crop_expand
        center = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
        face_ex_w, face_ex_h = (box[2] - box[0]) * expand_ratio, (box[3] - box[1]) * expand_ratio
        x1, y1 = int(max(math.floor(box[0] - face_ex_w), 0)), int(max(math.floor(box[1] - face_ex_h), 0))
        x2, y2 = int(min(math.ceil(box[2] + face_ex_w), w)), int(min(math.ceil(box[3] + face_ex_h), h))

        image = image[y1:y2, x1:x2]
        temp_save_wh = [image.shape[0], image.shape[1], x1, y1, x2, y2]

        center[0] = center[0] - x1
        box[0], box[2] = box[0] - x1, box[2] - x1
        center[1] = center[1] - y1
        box[1], box[3] = box[1] - y1, box[3] - y1

        box[0], box[1] = np.max([box[0], 0]), np.max([box[1], 0])
        box[2], box[3] = np.min([box[2], image.shape[0]]), np.min([box[3], image.shape[1]])

        ow, oh = self.param.crop_width, self.param.crop_height
        scale = [ow * 1. / w, oh * 1. / h]

        center[0] = center[0] * scale[0]
        center[1] = center[1] * scale[1]
        box[0], box[1] = box[0] * scale[0], box[1] * scale[1]
        box[2], box[3] = box[2] * scale[0], box[3] * scale[1]

        image = cv.resize(image, (ow, oh), cv.INTER_LINEAR)

        tensor = torch.from_numpy(image.transpose((2, 0, 1)))
        tensor = tensor.float().div(255)
        for t, m, s in zip(tensor, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]):
            t.sub_(m).div_(s)

        return tensor, temp_save_wh
