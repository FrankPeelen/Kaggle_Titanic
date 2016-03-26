import pandas
import numpy as np

def normalize(column):
	return (column - column.mean()) / (column.max() - column.min())

def genderToBinary(column):
	column[column == "male"] = 0
	column[column == "female"] = 1
	return column

def fillAgeGaps(column):
	return column.fillna(column.mean())
