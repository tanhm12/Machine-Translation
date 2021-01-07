from tokenizer.preprocess import VnSegmentNLP

san = VnSegmentNLP()

print(san.word_segment("Ông Nguyễn Khắc Chúc  đang làm việc tại Đại học Quốc gia Hà Nội. Bà Lan, vợ ông Chúc, cũng làm việc tại đây."))
print(san.word_segment("Ông Nguyễn Khắc Chúc  đang làm việc tại Đại học Quốc gia Hà Nội."))
