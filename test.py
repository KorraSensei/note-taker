from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

from sumy.parsers.html import HtmlParser
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

from cv2 import cv2
import pytesseract 

import spacy
import pytextrank

pytesseract.pytesseract.tesseract_cmd = 'C://Program Files//Tesseract-OCR//tesseract.exe'

img = cv2.imread("test.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV) 
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))
dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1)
contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,  
                                                 cv2.CHAIN_APPROX_NONE) 
im2 = img.copy()
file = open("recognized.txt", "w+") 
file.write("") 
file.close() 

for cnt in contours: 
    x, y, w, h = cv2.boundingRect(cnt) 
        
    # Drawing a rectangle on copied image 
    rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2) 
        
    # Cropping the text block for giving input to OCR 
    cropped = im2[y:y + h, x:x + w] 
        
    # Open the file in append mode 
    file = open("recognized.txt", "a") 
        
    # Apply OCR on the cropped image 
    text = pytesseract.image_to_string(cropped) 
        
    # Appending the text into file 
    file.write(text) 
    file.write("\n") 
        
    # Close the file 
    file.close 


LANGUAGE = "english"
SENTENCES_COUNT = 10

text = "test"
#url = ""

with open('recognized.txt', 'r') as file:
    data = file.read().replace('\n', '')


#parser = HtmlParser.from_url(url, Tokenizer(LANGUAGE))
# or for plain text files
#parser = PlaintextParser.from_file("recognized.txt", Tokenizer(LANGUAGE))
parser = PlaintextParser.from_string(data, Tokenizer(LANGUAGE))

stemmer = Stemmer(LANGUAGE)

summarizer = Summarizer(stemmer)
summarizer.stop_words = get_stop_words(LANGUAGE)
output = open("notes.txt", "a+")
output.truncate(0)
for sentence in summarizer(parser.document, SENTENCES_COUNT):
    print(sentence)
    output.write(str(sentence))
    output.write('\n')

