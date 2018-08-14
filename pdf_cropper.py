"""
PDF cropper
======
Crop pdf pages to a given size, and save it in an output file
"""
from PyPDF2 import PdfFileWriter,PdfFileReader,PdfFileMerger

pdf_file = PdfFileReader(open("/home/juliano/workspace/books/sample.pdf","rb"))
output_file = PdfFileWriter()

"""
# This prints the original dimensions of the page to help the crop
print(page.cropBox.getLowerLeft())
print(page.cropBox.getLowerRight())
print(page.cropBox.getUpperLeft())
print(page.cropBox.getUpperRight())
"""
for i in range(1, pdf_file.getNumPages()):
    page = pdf_file.getPage(i)
    page.mediaBox.lowerLeft = (130, 110)
    page.mediaBox.lowerRight = (462, 110)
    page.mediaBox.upperLeft = (130, 740)
    page.mediaBox.upperRight = (462, 740)
    output_file.addPage(page)

with open("out.pdf", "wb") as out_f:
    output_file.write(out_f)
