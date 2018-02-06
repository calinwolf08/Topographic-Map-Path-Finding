import cv2
import pytesseract
import os
from PIL import Image
import numpy as np

from helper_functions import Helper

class ImageData:
		# multipliers to get portion of image with interval value
	__bottom_thresh = 0.9
	__left_thresh = 0.35
	__right_thresh = 0.65

	# words, offset to interval value
	__words_offsets = [("CONTOUR", 2), ("INTERVAL", 1), ("FEET", -1)]
	__resize_factor = 6

	def __init__(self, image):
		self.image = image

		self.sub_image = self.__get_sub_image()
		self.word_list = self.__get_words()
		self._contour_interval_dist = None
		self._mile_in_pixels = None

	def __get_sub_image(self):
		rows, cols, chan = self.image.shape

		sub_image = self.image[
			int(self.__bottom_thresh*rows):rows, 						# bottom rows
			int(self.__left_thresh*cols):int(self.__right_thresh*cols)	# middle rows
			]

		sub_image = cv2.resize(sub_image, None, fx=self.__resize_factor, fy=self.__resize_factor, 
			interpolation = cv2.INTER_LINEAR)

		sub_image = Helper.convert_image_to_mask(sub_image)
		gray_denoised_image = cv2.fastNlMeansDenoising(sub_image, None, 5, 7, 21)
		threshold_image = cv2.threshold(gray_denoised_image,225,255,cv2.THRESH_BINARY_INV)[1]

		return sub_image

	def __get_countour_interval_dist(self):
		candidates = []
 		
		for word, offset in self.__words_offsets:
			candidates += self.__find_candidates_for_id_and_index(self.word_list, word, offset)

		return candidates[0][1] if len(candidates) > 0 else None 

	def __get_miles_in_pixels(self):
		print(self.word_list)

		# gray = cv2.cvtColor(self.sub_image, cv2.COLOR_BGR2GRAY)
		edges = cv2.Canny(self.sub_image, 50, 115)
		lines = cv2.HoughLines(edges, 1, np.pi/180, 50)

		img = self.sub_image.copy()

		for r,theta in lines[0]:
			# Stores the value of cos(theta) in a
			a = np.cos(theta)

			# Stores the value of sin(theta) in b
			b = np.sin(theta)
			 
			# x0 stores the value rcos(theta)
			x0 = a*r
			 
			# y0 stores the value rsin(theta)
			y0 = b*r
			 
			# x1 stores the rounded off value of (rcos(theta)-1000sin(theta))
			x1 = int(x0 + 1000*(-b))
			 
			# y1 stores the rounded off value of (rsin(theta)+1000cos(theta))
			y1 = int(y0 + 1000*(a))

			# x2 stores the rounded off value of (rcos(theta)+1000sin(theta))
			x2 = int(x0 - 1000*(-b))
			 
			# y2 stores the rounded off value of (rsin(theta)-1000cos(theta))
			y2 = int(y0 - 1000*(a))
			 
			# cv2.line draws a line in img from the point(x1,y1) to (x2,y2).
			# (0,0,255) denotes the colour of the line to be 
			#drawn. In this case, it is red. 
			cv2.line(img,(x1,y1), (x2,y2), (0,0,255),2)

		# cv2.imshow("lines", img)

		return -1

	def __find_candidates_for_id_and_index(self, word_list, id_word, offset):
		candidates = []

		indices = [i for i, x in enumerate(word_list) if x.upper() == id_word]

		for i in indices:
			if word_list[i+offset].isnumeric():
				cand = (i, word_list[i+offset])
				candidates.append(cand)

		return candidates

	def __get_words(self):
		filename = "{}.png".format(os.getpid())
		cv2.imwrite(filename, self.sub_image)

		results = pytesseract.image_to_string(Image.open(filename))

		from pprint import pprint
		results2 = pytesseract.image_to_string(Image.open(filename), boxes=True, config="hocr")

		pprint(results)
		print("----------------------------")

		pprint(results2)

		os.remove(filename)
		results_list = results.split()

		return results_list

	@property
	def contour_interval_dist(self):
		if self._contour_interval_dist is None:
			self._contour_interval_dist = self.__get_countour_interval_dist()

		return self._contour_interval_dist

	@contour_interval_dist.setter
	def contour_interval_dist(self, value):
		self._contour_interval_dist = value

	@property
	def mile_in_pixels(self):
		if self._mile_in_pixels is None:
			self._mile_in_pixels = self.__get_miles_in_pixels()

		return self._mile_in_pixels

	@mile_in_pixels.setter
	def mile_in_pixels(self, value):
		self._mile_in_pixels = value

class TopographicMap:
	def __init__(self, filename):
		self.filename = filename
		self.image = cv2.imread(filename, 1)
		self.image_data = ImageData(self.image)

if __name__ == '__main__':
	img = Topographic_Map("SanLuisObispo.jpg")
	print(img.countour_interval_dist)
