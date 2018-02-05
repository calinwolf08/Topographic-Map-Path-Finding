import cv2
import numpy as np

from generate_masks import MaskGenerator
from extract_contours import ContourExtractor

class CroppedImage:

	def __init__(self, cv_image):
		self.cv_image = cv_image
		# self.image_masks = MaskGenerator(self.cv_image).image_masks
		# self.contours = ContourExtractor(self).extracted_contours
		self._image_masks = None
		self.contours = None
		self.contours_color = None

	@property
	def image_masks(self):
		if self._image_masks is None:
			self._image_masks = MaskGenerator(self.cv_image).image_masks

		return self._image_masks

	@image_masks.setter
	def image_masks(self, value):
		self._image_masks = value

	@property
	def contours(self):
		if self._contours is None:
			self._contours = ContourExtractor(self).extracted_contours

		return self._contours

	@contours.setter
	def contours(self, value):
		self._contours = value
