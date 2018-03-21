import sys
import cv2
import tensorflow as tf
from IPython.display import display, Image
sys.path.append('../../../server/')
from dvalib import detector
from glob import glob
try:
    os.mkdir('boxes')
except:
    pass
	
text_detector = detector.TextBoxDetector(model_path='../../../repos/tf_ctpn_cpu/checkpoints/checkpoint')
text_detector.load()
box_count = 0
for im_name in glob("images/*.jpg"):
    display(Image(im_name,width=300))
    regions = text_detector.detect(im_name)
    im=cv2.imread(im_name)
    for k in regions:
        crop_img = im[int(k['y']):int(k['y'] + k['h']),int(k['x']):int(k['x'] + k['w'])]
        print k['score']
        cv2.imwrite('boxes/box_{}.jpg'.format(box_count),crop_img)
        display(Image('boxes/box_{}.jpg'.format(box_count)))
        box_count += 1