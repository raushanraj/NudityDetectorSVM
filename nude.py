import os
import numpy as np
from numpy import array
import cv2
from cv2 import cv
import skimage.feature as sf
from sklearn.svm import SVC
import glob
from urllib2 import urlopen
from cStringIO import StringIO
total_pixels=256.0*256.0

"""normalization is to make image of size 256X256 and convert the format to .jpg"""


def url_image(url, cv2_img_flag=1):
	request = urlopen(url)
	img_array = np.asarray(bytearray(request.read()), dtype=np.uint8)
	return cv2.imdecode(img_array, cv2_img_flag)
	
class image:
    def __init__(self,src,dst,label):
        self.src="C:\\Users\\raj\\Desktop\\mixed\\"+src
        self.dst="C:\\Users\\raj\\Desktop\\resized\\"+dst+".jpg"
        self.normalized=None
        self.label=label
    def resize(self):
        x,y=256,256
        src=cv2.imread(self.src,1)
        src=cv2.resize(src,(x,y))
        cv2.imwrite(self.dst,src)
        dst=cv2.imread(self.dst,1)
        self.normalized=dst
        return self.normalized
 
class imagecheck:
    def __init__(self,src,dst):
        self.src=src
        self.dst=dst+"_n.jpg"
        self.normalized=None
    def resize(self):
        x,y=256,256
        src=cv2.imread(self.src,1)
        src=cv2.resize(src,(x,y))
        cv2.imwrite(self.dst,src)
        dst=cv2.imread(self.dst,1)
        self.normalized=dst
        return self.normalized
         
    
"""Segnebtation module is used to segment out skin pixels in YCrCb color space"""

def segmentation(src):
    img=cv2.cvtColor(src,cv.CV_BGR2YCrCb)
    dst=cv2.inRange(img,(0,133,77),(255,173,127))
    return dst

"""Image Zoning and feature extraction module"""

class features:
	def __init__(self,src):
		self.zone1=src
		self.zone2=src[30:226,30:226]
		self.zone3=src[60:196,60:196]
	def createglcm(self,zone):
		return sf.greycomatrix(zone,[1],[0,np.pi/4,np.pi/2,np.pi*3/4,np.pi*25/12],normed=True)
	def getCorrelation(self,glcm):
		return sf.greycoprops(glcm,'correlation')
	def getHomogeneity(self,glcm):
		return sf.greycoprops(glcm,'homogeneity')
	def getcolorfeatures(self,zone):
		contours, hierarchy = cv2.findContours(zone,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		skin_pixel_connected=0 #area of all contours
		for i in range(len(contours)):
			skin_pixel_connected=skin_pixel_connected+cv2.contourArea(contours[i])
		
		return array([skin_pixel_connected,skin_pixel_connected/total_pixels])
		

def getFeatureVector(src):
	f=features(src)
	zone1=f.zone1
	zone2=f.zone2
	zone3=f.zone3
	zones=[zone1,zone2,zone3]
	feature=[]
	for zone in zones:
		glcm=f.createglcm(zone) 
		colorfeature=f.getcolorfeatures(zone)
		#print colorfeature
		homogeneity=f.getHomogeneity(glcm)
		#print homogeneity[0]
		correlation=f.getCorrelation(glcm)
		#print correlation[0]
		feature=np.concatenate((feature,colorfeature,homogeneity[0],correlation[0]))
		#print feature
	return feature.tolist()

if __name__=="__main__":
	imagelist=[]	
	nudeset=os.listdir("C:\\Users\\raj\\Desktop\\nude\\")
	nudeset.pop()
	for item in nudeset:
		dstpath=os.path.splitext(item)[0]
		#print item,dstpath
		imagelist.append(image(item,dstpath,-1))
	nonnudeset=os.listdir("C:\\Users\\raj\\Desktop\\nonnude\\")
	nonnudeset.pop()
	for item in nonnudeset:
		dstpath=os.path.splitext(item)[0]
		imagelist.append(image(item,dstpath,1))	
	featurespace=[]	
	classes=[]
	for image in imagelist:
		classes.append(image.label)
		image=segmentation(image.resize())
		#cv2.imshow("win",image)
		#cv2.waitKey(0)
		feature=getFeatureVector(image)
		featurespace.append(feature)
	featurespace=np.array(featurespace)
	classifier=SVC(kernel='rbf',C=100.0,gamma=0.07,cache_size=800)
	classifier.fit(np.array(featurespace),np.array(classes))
	path="C:\\Users\\raj\\Desktop\\nn7.jpg"
	testimage=imagecheck(path,os.path.splitext(path)[0])
	print classifier.predict([np.array(getFeatureVector(segmentation(testimage.resize())))])

