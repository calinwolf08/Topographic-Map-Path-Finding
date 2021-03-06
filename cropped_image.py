import cv2
import numpy as np

from helper_functions import Point
from generate_masks import MaskGenerator
from extract_contours import ContourExtractor

class CroppedImage:

	def __init__(self, cv_image):
		self.cv_image = cv_image
		self._image_masks = None
		self._contours = None

		self.start = Point(-1, -1)
		self.end = Point(-1, -1)
	
	def generate_masks(self):
		if self._image_masks is None:
			self._image_masks = MaskGenerator(self.cv_image).image_masks 

	@property
	def image_masks(self):
		if self._image_masks is None:
			self._image_masks = MaskGenerator(self.cv_image).image_masks

		return self._image_masks

	@image_masks.setter
	def image_masks(self, value):
		self._image_masks = value

	def generate_contours(self):
		if self._contours is None:
			self.contour_extractor = ContourExtractor(self)
			self._contours = self.contour_extractor.extracted_contours

	@property
	def contours(self):
		if self._contours is None:
			self.contour_extractor = ContourExtractor(self)
			self._contours = self.contour_extractor.extracted_contours

		return self._contours

	@contours.setter
	def contours(self, value):
		self._contours = value
