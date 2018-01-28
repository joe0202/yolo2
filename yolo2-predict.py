import tensorflow as tf
import os
import numpy as op
import pickle
from multiprocessing.pool import ThreadPool
import cv2
from darkflow.net.build import TFNet
import numpy as np
import quantize

def findboxes(out):
	options={"model":"cfg/yoloV2-dac.cfg","load":19000,"threshold":0.1,'gpu':0.3}
	tfnet=TFNet(options)
	boxes=tfnet.framework.findboxes(out)
	return boxes
    

def process_box(box,h,w,threshold=0.1):
	options={"model":"cfg/yoloV2-dac.cfg","load":19000,"threshold":0.1,'gpu':0.3}
	tfnet=TFNet(options)
	tempBox= tfnet.framework.process_box(box,h,w,threshold)
	return tempBox



def resize_input(im):
	h=416
	w=416
	c=3
	imsz=cv2.resize(im,(w,h))
	imsz=imsz / 255
	imsz=imsz[:,:,::-1]
	return imsz



imgcv=cv2.imread("test_image/000021.jpg")
ckpt_path="/home/qiaolinjun/qljproject/darkflow/ckpt"#the path of checkpoint file

h,w,_=imgcv.shape
ckpt=tf.train.get_checkpoint_state(ckpt_path)
saver= tf.train.import_meta_graph(ckpt.model_checkpoint_path+'.meta')
with tf.Session() as sess:
	saver.restore(sess,ckpt.model_checkpoint_path)#build the graph from ckpt file
	graph=tf.get_default_graph()
	out_op=graph.get_tensor_by_name("output:0")#get the out_tensor
	im=resize_input(imgcv)
	this_inp = np .expand_dims(im,0)
	feed_dict={graph.get_tensor_by_name("input:0"):this_inp}	
	out=sess.run(out_op,feed_dict)[0]#get the result
       
        #process the output
	boxes=findboxes(out)
	box_info=list()
	for box in boxes:
		tempBox=process_box(box,h,w,0.1)
		if tempBox is None:
	    		continue
		box_info.append({
			"label":tempBox[4],
			"confidence":tempBox[6],
			"topleft":{
        	"x":tempBox[0],
        	"y":tempBox[2]},

        	"bottomright":{
        	"x":tempBox[1],
        	"y":tempBox[3]}
			})
	print(box_info)
	

    
    
   

