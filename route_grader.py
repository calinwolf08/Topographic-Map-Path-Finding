import requests
from bs4 import BeautifulSoup
from selenium import webdriver
import selenium 
import time

login_url = 'https://www.alltrails.com/login?ref=header'
url = 'https://www.alltrails.com/us/california'

driver = webdriver.Chrome()

driver.get(login_url)
driver.find_element_by_id('user_email').send_keys('calinwolf08@gmail.com')
driver.find_element_by_id('user_password').send_keys('password')
driver.find_element_by_css_selector("input[type='submit']").click()
time.sleep(2)
driver.get(url)

test = True

cur_num = 0
prev_num = 0

count = 0

elem = None

while test:
	if driver.current_url != url:
		print("redirected: " + driver.current_url)
		input()
		break

	try:
		start = time.time()
		elem = driver.find_element_by_css_selector("a[data-reactid='474']")
		while 'Load More Trails' not in elem.get_attribute('innerHTML'):
			elem = driver.find_element_by_css_selector("a[data-reactid='474']")

		print(elem.get_attribute('innerHTML'))
		elem.click()
		end = time.time()
		print("clicked elem: " + str(count) + " : " + str(end - start))
	except selenium.common.exceptions.WebDriverException:
		print("not clickable")
	except selenium.common.exceptions.NoSuchElementException:
		print("no element found")

	# time.sleep(1)
	count+=1

	# time.sleep(1)
	# soup = BeautifulSoup(driver.page_source, 'html.parser')

	# ul = [item for item in soup.find_all('ul', attrs={'data-reactid': 4})]
	# li = [item for item in ul[0].find_all('li')]

	# cur_num = len(li)

	# if cur_num != prev_num:
	# 	print(str(count) + ": " + str(cur_num))
	# 	count += 1
	# 	prev_num = cur_num

driver.close()	
