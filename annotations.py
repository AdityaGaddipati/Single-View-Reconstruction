import numpy as np
import cv2

COLORS = [(0, 255, 0), (0, 255, 255), (255, 255, 0), (255, 0, 0), (0, 0, 255)]
def vis_annotations_q2a():
	'''
	annotations: (3, 2, 4) 
	3 groups of parallel lines, each group has 2 lines, each line annotated as (x1, y1, x2, y2)
	'''
	annotations = np.load('data/q2/q2a.npy')			
	img = cv2.imread('data/q2a.png')
	for i in range(annotations.shape[0]):
		COLOR = COLORS[i]
		lines = annotations[i]
		for j in range(lines.shape[0]):
			x1, y1, x2, y2 = lines[j]
			cv2.circle(img, (x1, y1), 3, COLOR, -1)
			cv2.circle(img, (x2, y2), 3, COLOR, -1)
			cv2.line(img, (x1, y1), (x2, y2), COLOR, 2)

	# cv2.imwrite("./output/q2a_1.png", img)
	cv2.imshow('q2a', img)
	cv2.waitKey(0)

def vis_annnotations_q2b():
	'''
	annotations: (3, 4, 2) 
	3 squares, 4 points for each square, each point annotated as (x, y).
		pt0 - pt1
		|		|
		pt3 - pt2
	'''
	annotations = np.load('data/q2/q2b.npy').astype(np.int64)		
	img = cv2.imread('data/q2b.png')

	for i in range(annotations.shape[0]):
		COLOR = COLORS[i]
		square = annotations[i]
		for j in range(square.shape[0]):
			x, y = square[j]		 
			cv2.circle(img, (x, y), 3, COLOR, -1)
			cv2.putText(img, str(j), (x, y), cv2.FONT_HERSHEY_DUPLEX, 1, COLOR, 1, cv2.LINE_AA)
			cv2.line(img, square[j], square[(j+1) % 4], COLOR, 2)

	# cv2.imwrite("./output/q2b_1.png", img)
	cv2.imshow('q2b', img)
	cv2.waitKey(0)

def vis_annnotations_q2c():
	'''
	annotations: (3, 4, 2) 
	3 squares, 4 points for each square, each point annotated as (x, y).
		pt0 - pt1
		|		|
		pt3 - pt2
	'''
	annotations = np.load('data/q2/q2c.npy').astype(np.int64)		
	img = cv2.imread('data/q2c.jpg')

	for i in range(annotations.shape[0]):
		COLOR = COLORS[i]
		square = annotations[i]
		for j in range(square.shape[0]):
			x, y = square[j]		 
			cv2.circle(img, (x, y), 5, COLOR, -1)
			cv2.putText(img, str(j), (x, y), cv2.FONT_HERSHEY_DUPLEX, 2, COLOR, 1, cv2.LINE_AA)
			cv2.line(img, square[j], square[(j+1) % 4], COLOR, 5)
	
	# cv2.imwrite("./output/q2c_1.png", img)
	cv2.imshow('q2c', img)
	cv2.waitKey(0)

def vis_annotations_q3a():
	'''
	annotations: (5, 4, 2)
	5 planes in the scene, 4 points for each plane, each point annotated as (x, y).
		pt0 - pt1
		|		|
		pt3 - pt2
	'''
	annotations = np.load('data/q3/q3.npy').astype(np.int64)		
	img = cv2.imread('data/q3.png')

	for i in range(annotations.shape[0]):
		COLOR = COLORS[i]
		square = annotations[i]
		for j in range(square.shape[0]):
			x, y = square[j]		 
			cv2.circle(img, (x, y), 3, COLOR, -1)
			cv2.putText(img, str(j+i*4), (x, y), cv2.FONT_HERSHEY_DUPLEX, 1, COLOR, 1, cv2.LINE_AA)
			cv2.line(img, square[j], square[(j+1) % 4], COLOR, 2)

		cv2.imshow('q3a', img)
		cv2.waitKey(0)
	# cv2.imwrite("./output/q3a_1.png", img)

def vis_annotations_q3b():
	'''
	annotations: (5, 4, 2)
	5 planes in the scene, 4 points for each plane, each point annotated as (x, y).
		pt0 - pt1
		|		|
		pt3 - pt2
	'''
	# annotations = np.load('data/q3/q3b_1.npy').astype(np.int64)		
	# img = cv2.imread('data/q3b_1.jpeg')

	annotations = np.load('data/q3/q3b_2.npy').astype(np.int64)		
	img = cv2.imread('data/q3b_2.jpeg')

	for i in range(annotations.shape[0]):
		COLOR = COLORS[4-i]
		square = annotations[i]
		for j in range(square.shape[0]):
			x, y = square[j]		 
			cv2.circle(img, (x, y), 3, COLOR, -1)
			cv2.putText(img, str(j+i*4), (x, y), cv2.FONT_HERSHEY_DUPLEX, 1, COLOR, 1, cv2.LINE_AA)
			cv2.line(img, square[j], square[(j+1) % 4], COLOR, 2)

		cv2.imshow('q3b', img)
		cv2.waitKey(0)
	# cv2.imwrite("./output/q3b_2_1.jpeg", img)

if __name__ == '__main__':
	# vis_annotations_q2a()
	# vis_annnotations_q2b()
	# vis_annnotations_q2c()
	# vis_annotations_q3a()
	vis_annotations_q3b()
