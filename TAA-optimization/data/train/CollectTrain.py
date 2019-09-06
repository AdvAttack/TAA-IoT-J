import shutil
import os

for folder in range(26):
	ImgList = os.listdir(str(folder))
	for Img in ImgList:
		shutil.copy(os.path.join(str(folder),Img), os.path.join('./AllTrain'))
