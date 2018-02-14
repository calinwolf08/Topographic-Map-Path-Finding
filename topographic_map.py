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

	# (words, offset) to contour interval value
	__words_offsets = [("CONTOUR", 2), ("INTERVAL", 1), ("FEET", -1)]
	__resize_factor = 6

	def __init__(self, image):
		self.image = image

		self.sub_image = self.__get_sub_image()
		
		word_list, box_list = self.__get_words()
		self.word_list = word_list
		self.box_list = box_list
		
		self._contour_interval_dist = None
		self._feet_per_pixel = None

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

	def __get_feet_per_pixel(self):
		# row_size = 6
		# total = int(len(self.box_list) / 6)
		# idx = 0

		# nums = [(idx, int(char)) for idx, char in enumerate(self.box_list) 
		# if idx % row_size == 0 and char.isdigit() and int(char) > 2 and int(char) < 10]

		# nums.sort(key=lambda val: self.box_list[val[0] + 2])

		# threshold = 3
		# prev_x = -1
		# prev_y = -2 * threshold
		# prev_num = -1

		# img = self.sub_image.copy()

		# lsd = cv2.createLineSegmentDetector(0)
		# lines = lsd.detect(img)[0] 
		# drawn_img = lsd.drawSegments(img,lines)
		# cv2.imshow("LSD",drawn_img )
		
		# # h, w, _ = img.shape

		# # for (idx, num) in nums:
		# # 	cur_x = int(self.box_list[idx + 1])
		# # 	cur_y = int(self.box_list[idx + 2])
		# # 	cur_x2 = int(self.box_list[idx + 3])
		# # 	cur_y2 = int(self.box_list[idx + 4])

		# # 	print(str(num) + ": " + str(cur_x) + ", " + str(cur_y) + " :: " + str(cur_x2) + ", " + str(cur_y2))
		# # 	img = cv2.rectangle(img,(cur_x,h-cur_y),(cur_x2,h-cur_y2),(255,0,0),2)
		# # 	# if abs(cur_y - prev_y) < threshold:
		# # 	# 	dist = abs(cur_x - cur_y)
		# # 	# 	diff = abs(num - prev_num)
		# # 	# 	print("possibility found ^\n--------")

		# # 	# prev_x = cur_x
		# # 	# prev_y = cur_y
		# # 	# prev_num = num
		# img = cv2.resize(img, None, fx=1/6, fy=1/6, 
		# 	interpolation = cv2.INTER_LINEAR)
		# cv2.imshow("blah", img)
		# print(nums)

		return int(1650 / 5280)# hardcoded estimatem, pixel per mile / ft per mile

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

		words = pytesseract.image_to_string(Image.open(filename))

		boxes = pytesseract.image_to_string(Image.open(filename), boxes=True, config="hocr")

		os.remove(filename)
		word_list = words.split()
		box_list = boxes.split()

		return word_list, box_list

	@property
	def contour_interval_dist(self):
		if self._contour_interval_dist is None:
			self._contour_interval_dist = self.__get_countour_interval_dist()

		return self._contour_interval_dist

	@contour_interval_dist.setter
	def contour_interval_dist(self, value):
		self._contour_interval_dist = value

	@property
	def feet_per_pixel(self):
		if self._feet_per_pixel is None:
			self._feet_per_pixel = self.__get_feet_per_pixel()

		return self._feet_per_pixel

	@feet_per_pixel.setter
	def feet_per_pixel(self, value):
		self._feet_per_pixel = value

class TopographicMap:
	def __init__(self, filename):
		self.filename = filename
		self.image = cv2.imread(filename, 1)
		self.image_data = ImageData(self.image)

if __name__ == '__main__':
	img = Topographic_Map("SanLuisObispo.jpg")
	print(img.countour_interval_dist)
