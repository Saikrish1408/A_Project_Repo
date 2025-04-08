import easyocr

reader = easyocr.Reader(['en'])  # Specify language
text_results = reader.readtext('img.jpg', detail=0)  # Extract text without bounding box details

extracted_text = ' '.join(text_results)
print(extracted_text)
