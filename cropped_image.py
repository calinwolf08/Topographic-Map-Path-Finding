import cv2
import numpy as np

from generate_masks import MaskGenerator
from extract_contours import ContourExtractor

class CroppedImage:

	def __init__(self, image):
		self.image = image
		self.mask_generator = MaskGenerator(image)
		
		self.mask = None
		self.contours = None
		self.contours_color = None

	def get_contours():
		if self.contours == None:
			self.extract_contours()

		return self.contours

	def extract_contours(self):
		self.mask = self.get_topo_mask()
		# prepare img for extracting contours by dilating, denoising, and finding edges 
		# then apply mask and dilate edges
		(contours_mask, contours_color) = ContourExtractor.prepare_for_contour_extraction(self.image, mask)

		# detect contours and connect extreme points within certain distances
		iters = [(10,10), (20,10), (30,10)]
		(connected_mask, connected_color, contours_img) = ContourExtractor.run_connect_contours(
			self.image, contours_mask, contours_color, iters)

		skel = ContourExtractor.skeletonize_mask(connected_mask)
		skel = ContourExtractor.reduce_image_contours(skel, 1)

		kernel = np.ones((2,2), np.uint8)
		dilated = cv2.dilate(skel, kernel, iterations=1)

		skel_color = cv2.bitwise_and(self.image, self.image, mask=dilated)

		iters += [(40,10), (50,10), (60,10), (70,10)]
		(final, connected_color, final_color) = ContourExtractor.run_connect_contours(
			self.image, dilated, skel_color, iters, 
			lineThickness = 2, minArea = 2, allExtremes = False)

		self.contours = final
		self.contours_color = final_color

		cv2.imshow("cont", self.contours)
		cv2.imshow("cont col", self.contours_color)

	def get_topo_mask(self):
		topo_mask = self.mask_generator.get_topo_mask()

		return topo_mask
