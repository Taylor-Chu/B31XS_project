# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
#from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data import build_detection_train_loader
from detectron2.engine import DefaultTrainer
from detectron2.engine import SimpleTrainer
from detectron2.engine import HookBase
from typing import Dict, List, Optional
import detectron2.solver as solver
import detectron2.modeling as modeler
import detectron2.data as data
import detectron2.data.transforms as T
import detectron2.checkpoint as checkpointer
from detectron2.data import detection_utils as utils
import weakref
import copy
import torch
import time
from detectron2.structures import BoxMode
from astropy.io import fits
import glob
from detectron2.utils.visualizer import ColorMode
from detectron2.evaluation import inference_on_dataset
from detectron2.data import build_detection_test_loader

from astrodet import astrodet as toolkit

# Prettify the plotting
from astrodet.astrodet import set_mpl_style

def get_astro_dicts(img_dir):
        
    # It's weird to call this img_dir
    set_dirs = sorted(glob.glob('%s/set_*' % img_dir))
    
    dataset_dicts = []
    
    # Loop through each set
    for idx, set_dir in enumerate(set_dirs):
        record = {}
        
        mask_dir = os.path.join(img_dir, set_dir, "masks.fits")
        filename = os.path.join(img_dir, set_dir, "img")
        
        # Open each FITS image
        with fits.open(mask_dir, memmap=False, lazy_load_hdus=False) as hdul:
            sources = len(hdul)
            height, width = hdul[0].data.shape
            data = [hdu.data/np.max(hdu.data) for hdu in hdul]
            category_ids = [hdu.header["CLASS_ID"] for hdu in hdul]
            
        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
        objs = []
        
        # Mask value thresholds per category_id
        thresh = [0.005 if i == 1 else 0.08 for i in category_ids]
        
        # Generate segmentation masks
        for i in range(sources):
            image = data[i]
            mask = np.zeros([height, width], dtype=np.uint8)
            # Create mask from threshold
            mask[:,:][image > thresh[i]] = 1
            # Smooth mask
            mask[:,:] = cv2.GaussianBlur(mask[:,:], (9,9), 2)
            
            # https://github.com/facebookresearch/Detectron/issues/100
            contours, hierarchy = cv2.findContours((mask).astype(np.uint8), cv2.RETR_TREE,
                                                        cv2.CHAIN_APPROX_SIMPLE)
            # mask_new, contours, hierarchy = cv2.findContours((mask).astype(np.uint8), cv2.RETR_TREE,
            segmentation = []
            for contour in contours:
                x,y,w,h = cv2.boundingRect(contour)
                contour = contour.flatten().tolist()
                # segmentation.append(contour)
                if len(contour) > 4:
                    segmentation.append(contour)
            # No valid countors
            if len(segmentation) == 0:
                continue
            
            # Add to dict
            obj = {
                "bbox": [x, y, w, h],
                "area": w*h,
                "bbox_mode": BoxMode.XYWH_ABS,
                "segmentation": segmentation,
                "category_id": category_ids[i] - 1,
            }
            objs.append(obj)
            
        record["annotations"] = objs
        dataset_dicts.append(record)
         
    return dataset_dicts

def read_image(filename, normalize='lupton', stretch=5, Q=10, m=0, ceil_percentile=99.995, dtype=np.uint8):
    
    # Read image
    g = fits.getdata(os.path.join(filename+'_g.fits'), memmap=False)
    r = fits.getdata(os.path.join(filename+'_r.fits'), memmap=False)
    z = fits.getdata(os.path.join(filename+'_z.fits'), memmap=False)
    
    # Contrast scaling / normalization
    I = (z + r + g)/3.0
    
    length, width = g.shape
    image = np.empty([length, width, 3], dtype=dtype)
    
    # Options for contrast scaling
    if normalize.lower() == 'lupton':
        z = z*np.arcsinh(stretch*Q*(I - m))/(Q*I)
        r = r*np.arcsinh(stretch*Q*(I - m))/(Q*I)
        g = g*np.arcsinh(stretch*Q*(I - m))/(Q*I)
    elif normalize.lower() == 'zscore':
        Isigma = I*np.mean([np.nanstd(g), np.nanstd(r), np.nanstd(z)])
        z = (z - np.nanmean(z) - m)/Isigma
        r = (r - np.nanmean(r) - m)/Isigma
        g = (g - np.nanmean(g) - m)/Isigma
    elif normalize.lower() == 'linear':
        z = np.nan_to_num(z/I)
        r = np.nan_to_num(r/I)
        g = np.nan_to_num(g/I)
        # z = (z - m)/I
        # r = (r - m)/I
        # g = (g - m)/I
    else:
        print('Normalize keyword not recognized.')

    max_RGB = np.nanpercentile([z, r, g], ceil_percentile) * 2
    # avoid saturation
    r = r/max_RGB; g = g/max_RGB; z = z/max_RGB

    # Rescale to 0-255 for dtype=np.uint8
    max_dtype = np.iinfo(dtype).max
    r = r*max_dtype
    g = g*max_dtype
    z = z*max_dtype

    # 0-255 RGB image
    image[:,:,0] = z # R
    image[:,:,1] = r # G
    image[:,:,2] = g # B
    
    return image

def train_mapper(dataset_dict, **read_image_args):

    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
    
    image = read_image(dataset_dict["file_name"], **read_image_args)
    augs = T.AugmentationList([
        T.RandomRotation([-90, 90, 180], sample_style='choice'),
        T.RandomFlip(prob=0.5),
        T.Resize((512,512))
    ])
    # Data Augmentation
    auginput = T.AugInput(image)
    # Transformations to model shapes
    transform = augs(auginput)
    image = torch.from_numpy(auginput.image.copy().transpose(2, 0, 1))
    annos = [
        utils.transform_instance_annotations(annotation, [transform], image.shape[1:])
        for annotation in dataset_dict.pop("annotations")
    ]
    return {
       # create the format that the model expects
        "image": image,
        "image_shaped": auginput.image,
        "height": 512,
        "width": 512,
        "image_id": dataset_dict["image_id"],
        "instances": utils.annotations_to_instances(annos, image.shape[1:]),
    }

def test_mapper(dataset_dict, **read_image_args):

    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

    image = read_image(dataset_dict["file_name"], *read_image_args)
    augs = T.AugmentationList([
        #T.RandomRotation([-90, 90, 180], sample_style='choice'),
        #T.RandomFlip(prob=0.5),
        #T.Resize((512,512))
    ])
    # Data Augmentation
    auginput = T.AugInput(image)
    # Transformations to model shapes
    transform = augs(auginput)
    image = torch.from_numpy(auginput.image.copy().transpose(2, 0, 1))
    annos = [
        utils.transform_instance_annotations(annotation, [transform], image.shape[1:])
        for annotation in dataset_dict.pop("annotations")
    ]
    return {
       # create the format that the model expects
        "image": image,
        "image_shaped": auginput.image,
        "height": 512,
        "width": 512,
        "image_id": dataset_dict["image_id"],
        "instances": utils.annotations_to_instances(annos, image.shape[1:]),
        "annotations": annos
    }

if __name__ == '__main__':
    
    dirpath = '/work/sc004/sc004/tc1213/astrodet/dataset/' # Path to dataset, should be changed accordingly
    dataset_names = ['train', 'test', 'val'] 
    
    for i, d in enumerate(dataset_names):
        filenames_dir = os.path.join(dirpath,d)
        DatasetCatalog.register("astro_" + d, lambda: get_astro_dicts(filenames_dir))
        MetadataCatalog.get("astro_" + d).set(thing_classes=["star", "galaxy"], things_colors = ['blue', 'gray'])
    astro_metadata = MetadataCatalog.get("astro_train")
    
    dataset_dicts = {}
    for i, d in enumerate(dataset_names):
        print(f'Loading {d}')
        dataset_dicts[d] = get_astro_dicts(os.path.join(dirpath, d))
    output_dir = '/work/sc004/sc004/tc1213/astrodet/output/JWST_20000/' # Path to output directory
    os.makedirs(output_dir, exist_ok=True)
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x.yaml")) # Get model structure
    cfg.DATASETS.TRAIN = ("astro_train") # Register Metadata
    cfg.DATASETS.TEST = ("astro_val") # Config calls this TEST, but it should be the val dataset
    cfg.TEST.EVAL_PERIOD = 40
    cfg.DATALOADER.NUM_WORKERS = 1
    # if init_coco_weights:
    #     cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x.yaml")  # Initialize from MS COCO
    # else:
    #     cfg.MODEL.WEIGHTS = os.path.join(output_dir, 'model_temp.pth')  # Initialize from a local weights
    # cfg.MODEL.WEIGHTS = os.path.join(output_dir, 'model_temp.pth')  # Initialize from a local weights
    # cfg.MODEL.WEIGHTS = '/work/sc004/sc004/tc1213/astrodet/mask_rcnn_R_50_C4_3x.pkl'  # Initialize from a local weights
    # cfg.MODEL.WEIGHTS = '/work/sc004/sc004/tc1213/astrodet/astro_rcnn_decam.h5'  # Initialize from a local weights
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.001   # pick a good LR -- start from 0.005
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.SOLVER.MAX_ITER = 100    # for DefaultTrainer
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    cfg.OUTPUT_DIR = output_dir
    cfg.TEST.DETECTIONS_PER_IMAGE = 1000

    # Inference should use the config with parameters that are used in training
    # cfg now already contains everything we've set previously. We changed it a little bit for inference:
    cfg.MODEL.WEIGHTS = '/work/sc004/sc004/tc1213/astrodet/output/model_20000.pth' # path to the model trained, can be downloaded from dropbox link in README.md, should be changed accordingly
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold
    predictor = DefaultPredictor(cfg)
    
    # nsample = 3
    # # fig, axs = plt.subplots(1, nsample, figsize=(5*nsample, 5))

    # for i, d in enumerate(random.sample(dataset_dicts['test'], nsample)):
    #     plt.figure(figsize=(10,10))
    #     img = read_image(d["file_name"], normalize="lupton", stretch=5, Q=1, ceil_percentile=99.5)
    #     outputs = predictor(img)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        
    #     print('total instances:', len(d['annotations']))
    #     print('detected instances:', len(outputs['instances'].pred_boxes))
    #     print('')
        
    #     v = Visualizer(img,
    #                 metadata=astro_metadata, 
    #                 scale=1, 
    #                 instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    #     )
    #     out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    #     plt.imshow(out.get_image())
    #     plt.axis('off')
    #     # plt.tight_layout()
    #     plt.savefig(os.path.join(output_dir, f'pred_{i}.png'))
    #     # fig.show()
    d = dataset_dicts['test'][46]
    plt.figure(figsize=(10,10))
    img = read_image(d["file_name"], normalize="linear", stretch=5, Q=1, ceil_percentile=99.5)
    outputs = predictor(img)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    
    print('total instances:', len(d['annotations']))
    print('detected instances:', len(outputs['instances'].pred_boxes))
    print('')
    
    v = Visualizer(img,
                metadata=astro_metadata, 
                scale=1, 
                instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    plt.imshow(out.get_image())
    plt.axis('off')
    # plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'pred_{i}.png'))
    # fig.show()
        
    # NOTE: New version has max_dets_per_image argument in default COCOEvaluator
    # evaluator = toolkit.COCOEvaluatorRecall("astro_val", use_fast_impl=True, output_dir=cfg.OUTPUT_DIR)

    # # First run with train_mapper to generate .json files consistent with training format
    # # Then run with test_mapper to get AP scores (doesn't work with augmentation mapper)
    # train_loader = build_detection_test_loader(dataset_dicts['val'], mapper=train_mapper)
    # test_loader = build_detection_test_loader(dataset_dicts['val'], mapper=test_mapper)
    
    # results = inference_on_dataset(predictor.model, test_loader, evaluator)
    
    # np.save(os.path.join(output_dir, 'results.npy'), results)
    
    # ap_type = 'bbox' # Which type of precision/recall to use? 'segm', or 'bbox'
    # cls_names = ['star', 'galaxy']

    # results_per_category = results[ap_type]['results_per_category']

    # # fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    # # axs = axs.flatten()

    # ious = np.linspace(0.50,0.95,10)
    # colors = plt.cm.viridis(np.linspace(0,1,len(ious)))

    # # Plot precision recall
    # for j, precision_class in enumerate(results_per_category):
    #     precision_shape = np.shape(precision_class)
    #     plt.figure()
    #     for i in range(precision_shape[0]):
    #         # precision has dims (iou, recall, cls, area range, max dets)
    #         # area range index 0: all area ranges
    #         # max dets index -1: typically 100 per image
    #         p_dat = precision_class[i, :, j, 0, -1]
    #         # Hide vanishing precisions
    #         mask = (p_dat > 0)
    #         # Only keep first occurance of 0 value in array
    #         mask[np.cumsum(~mask) == 1] = True
    #         p = p_dat[mask]
    #         # Recall points
    #         r = np.linspace(0, 1, len(p)) # Recall is always defined from 0 to 1 for these plots, I think
    #         dr = np.diff(np.linspace(0, 1, len(p_dat)))[0] # i think
    #         # Plot
    #         iou = np.around(ious[i], 2)
    #         AP = 100*np.sum(p*dr)
    #         plt.plot(r, p, label=r'${\rm{AP}}_{%.2f} = %.1f$' % (iou, AP), color=colors[i], lw=2) # use a viridis color scheme
    #         plt.xlabel('Recall', fontsize=20)
    #         plt.ylabel('Precision', fontsize=20)
    #         plt.xlim(0, 1.1)
    #         plt.ylim(0, 1.1)
    #         plt.legend(fontsize=10, title=f'{cls_names[j]}', bbox_to_anchor=(1.35, 1.0))
    #     plt.savefig(os.path.join(output_dir, f'result_{cls_names[j]}.png'))
            
    # fig.tight_layout()