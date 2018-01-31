import cv2
import numpy as np

from generate_masks import MaskGenerator
from extract_contours import ContourExtractor

class CroppedImage:

	def __init__(self, cv_image):
		self.cv_image = cv_image
		self.cv_image_masks = None
		self.contours = None
		self.contours_color = None

	def get_contours(self):
		if self.contours == None:
			self.__get_image_masks()
			self.__extract_contours()

		return self.contours

	def __get_image_masks(self):
		if self.cv_image_masks == None:
			self.cv_image_masks = MaskGenerator(self.cv_image).get_image_masks()

		return self.cv_image_masks

	def __extract_contours(self):
		extractor = ContourExtractor(self)
		extractor.extract_contours()

		self.contours = extractor.extracted_contours
