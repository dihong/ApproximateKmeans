#Usage
#	sense_dist.py word
#Author: Dihong
import sys, getopt, os, os.path
import nltk
from nltk.corpus import wordnet as wn
import subprocess
from decimal import Decimal
from numpy import zeros
import time as dt

def main(word):
	#get definition of word from WordNet.
	print 'Word to be disambiguated: %s. The senses in the WordNet are:' % word
	if os.path.isfile(word) == False:
		print 'Error: the input file "'+word+'" does not exist.'
		sys.exit(2)
	ss = wn.synsets(word,pos="n")
	for i in range (0,len(ss)):
		print str(i+1) + ': ' + ss[i].definition()
	print '-----------------------------'
	#Train classifier
	n1=dt.time()
	sys.stdout.write('Generating seeds ... ')
	fd = open(word+'.seed', 'w');
	fd.truncate()
	for i in range (0,len(ss)):
		fd.write(ss[i].definition()+'\n')
	fd.close()
	n2=dt.time()
	print 'done. Elapsed time is %.4f seconds.' % ((n2-n1))
	n1=dt.time()
	sys.stdout.write('Training classifier ... ')
	cmd = './uwsd -train '+word+' '+word+'.seed'+' '+word+'.classifier'+' ' + word
	os.system(cmd)
	n2=dt.time()
	print 'done. Elapsed time is %.4f seconds.' % ((n2-n1))
	n1=dt.time()
	#Classify the text
	sys.stdout.write('Classifying text ... ')
	cmd = './uwsd -test '+word+'.classifier'+' '+word+' ' + word
	out = subprocess.check_output(cmd, shell=True)
	fd = open(word+'.disambiguated', 'w');
	fd.truncate()
	fd.write(out)
	fd.close()
	n2=dt.time()
	print 'done. Elapsed time is %.4f seconds.' % ((n2-n1))
	#Print sense distribution.
	fd_in_text = open(word, 'r');
	in_text = fd_in_text.read().split('\n');
	in_text_label = zeros(len(in_text));
	out = out.split('\n');
	print '-----------------------------'
	hist = zeros(len(ss)+1)
	for i in range(0,len(out)):
		if len(out[i])>1:
			t = Decimal(out[i].split(' ')[0])
			if t==-1:
				hist[0] += 1
				in_text_label[i] = 0;
			else:
				hist[t] += 1
				in_text_label[i] = t;
	print 'Total number of %d lines were processed. The sense distribution is:' % (sum(hist))
	hist = hist/sum(hist);
	for i in range(0,len(hist)):
		if i==0:
			print 'N/A:      %2d%%' % (100*hist[0])
		else:
			print'Sense %2d: %2d%%' % (i,100*hist[i])
		if hist[i]<0:
			cnt = 1
			for k in range(0,len(in_text)):
				if len(in_text[k])>1 and in_text_label[k]==i:
					print in_text[k]
					cnt += 1
				if cnt >3:
					break;

				
				
if __name__ == "__main__":
	if len(sys.argv)  != 2:
		print 'Need to specify the word to be disambiguated.';
		sys.exit(2);
	main(sys.argv[1])
