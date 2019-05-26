import os
import scipy.io
import numpy as np
from .util import read_image
import pdb


class WIDERBboxDataset:
    def __init__(self, path_to_label, path_to_image, fname):
        self.path_to_label = path_to_label
        self.path_to_image = path_to_image
        self.f = scipy.io.loadmat(os.path.join(path_to_label, fname))
        self.event_list = self.f.get('event_list')
        self.file_list = self.f.get('file_list')
        self.face_bbx_list = self.f.get('face_bbx_list')
        self.label_names = WIDER_BBOX_LABEL_NAMES
        self.im_list, self.bbox_list = self.get_img_list()
        # ipdb.set_trace()
        self.is_difficult = False

    def get_img_list(self):
        im_list = []
        bbox_list = []
        for event_idx, event in enumerate(self.event_list):
            directory = event[0][0]
            for im_idx, im in enumerate(self.file_list[event_idx][0]):
                im_name = im[0][0]
                im_list.append(os.path.join(self.path_to_image, directory, \
                                            im_name + '.jpg'))
                face_bbx = self.face_bbx_list[event_idx][0][im_idx][0]
                bbox_list.append(face_bbx)
        return im_list, bbox_list

    def __len__(self):
        return len(self.im_list)

    def get_example(self, i):
        """Returns the i-th example.
        Returns a color image and bounding boxes. The image is in CHW format.
        The returned image is RGB.

        Args:
            i (int): The index of the example.

        Returns:
            tuple of an image and bounding boxes
        """
        # Load a image
        img_file = self.im_list[i]
        face_bbx = self.bbox_list[i]
        img = read_image(img_file, color=True)
        bboxes = []
        label = []
        difficult = []
        for i in range(face_bbx.shape[0]):
            xmin = int(face_bbx[i][0])
            ymin = int(face_bbx[i][1])
            xmax = int(face_bbx[i][2]) + xmin
            ymax = int(face_bbx[i][3]) + ymin
            bboxes.append((ymin, xmin, ymax, xmax))
            label.append(WIDER_BBOX_LABEL_NAMES.index('Face'))
            difficult.append(self.is_difficult)
        bboxes = np.stack(bboxes).astype(np.float32)
        label = np.stack(label).astype(np.int32)
        difficult = np.array(difficult, dtype=np.bool).astype(np.uint8)  # PyTorch don't support np.bool
        return img, bboxes, label, difficult

    __getitem__ = get_example


WIDER_BBOX_LABEL_NAMES = (
    'Face')
