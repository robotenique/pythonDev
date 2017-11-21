import re
import sys
from pathlib import Path

"""
Python Markdown image center
====

Centers all images in a markdown file (like github README.md).

Usage: python img_conv.py <file.md>
Or by editing the 'myfile' variable below.
"""

myfile = "pythonDev/README.md"
if len(sys.argv) >= 2:
	if Path(sys.argv[1]).is_file():
		file_2_update = sys.argv[1]
	else:
		print(f"File \'{sys.argv[1]}\' doesn't exist! Using hardcoded file..")
		file_2_update = myfile
else:
	file_2_update = myfile

center_div = lambda img: f"<p align=\"center\"> <img src=\"{img}\"/></p>"
img_l = []
subs = []
with open(file_2_update, "r+") as file:
	text = "".join(file.readlines())
	for s in re.findall("!\[.+\]\(.+\)", text):	
		div = center_div(re.findall("\(.+\)", s)[0].replace("(","").replace(")", ""))
		text = text.replace(s, div)
	file.seek(0)
	file.write(text)
	file.truncate()