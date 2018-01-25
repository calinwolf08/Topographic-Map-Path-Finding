import cv2
import pytesseract
import os
from PIL import Image

class TopographicMap:
	# multipliers to get portion of image with interval value
	__bottom_thresh = 0.85
	__left_thresh = 0.25
	__right_thresh = 0.75

	# words, offset to interval value
	__words_offsets = [("CONTOUR", 2), ("INTERVAL", 1), ("FEET", -1)]

	def __init__(self, filename):
		self.filename = filename
		self.image = cv2.imread(filename, 1)
		self.__set_countour_interval_dist()

	# find a number, x, in the image in the context: "CONTOUR INTERVAL x FEET" 
	def __set_countour_interval_dist(self):
		results = self.__detect_numbers()
		word_list = results.split()

		candidates = []
 		
		for word, offset in self.__words_offsets:
			candidates += self.__find_candidates_for_id_and_index(word_list, word, offset)

		self.countour_interval_dist = candidates[0][1]

	# look for candidate values by searching for id_word and getting the word at the offset
	def __find_candidates_for_id_and_index(self, word_list, id_word, offset):
		candidates = []

		indices = [i for i, x in enumerate(word_list) if x.upper() == id_word]

		for i in indices:
			if word_list[i+offset].isnumeric():
				cand = (i, word_list[i+offset])
				candidates.append(cand)

		return candidates

	# use pytesseract, find all words in image
	def __detect_numbers(self):
		rows, cols, chan = self.image.shape

		# get portion of image with contour interval text
		img = self.image[
			int(self.__bottom_thresh*rows):rows, 					# bottom rows
			int(self.__left_thresh*cols):int(self.__right_thresh*cols)	# middle rows
			]

		filename = "{}.png".format(os.getpid())
		cv2.imwrite(filename, img)

		results = pytesseract.image_to_string(Image.open(filename))
		os.remove(filename)

		return results

if __name__ == '__main__':
	img = Topographic_Map("SanLuisObispo.jpg")
	print(img.countour_interval_dist)
