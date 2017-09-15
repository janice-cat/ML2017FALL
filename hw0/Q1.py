import sys
s = sys.stdin.read()

Data = s.split()

Answer = []
SerialNoNew = 0
SerialNoFill = 0
for word in Data:

	flag_in = 0
	for AnswerLine in Answer:
		if word == AnswerLine[0]:
			flag_in = 1
			SerialNoFill = AnswerLine[1]
			break

	if flag_in == 0:
		Answer.append([word,SerialNoNew,1])
		SerialNoNew += 1	
	elif flag_in == 1:
		Answer[SerialNoFill][2] += 1

for AnswerLine in Answer:
	print (AnswerLine[0],AnswerLine[1],AnswerLine[2])