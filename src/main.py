import numpy

filename = 'data/spambase.data'

raw_data = open(filename, 'rb')

data = numpy.loadtxt(raw_data, delimiter=",")

print data.shape
