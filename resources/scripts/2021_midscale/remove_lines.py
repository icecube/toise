import json

infile='IceCubeHEX_Sunflower_240m_v3_ExtendedDepthRange.GCD.txt'
file = open(infile, 'r')

# load the list of included strings
gsl = open('midscale_geos.json')
options = json.load(gsl)
gsl.close()
gen2_strings = options['corner'] # select the corner
for i, string in enumerate(gen2_strings):
	gen2_strings[i] += 1000

strings_to_keep = list(range(1,87))
strings_to_keep += gen2_strings

# change the name here!
outfile = open('IceCubeHEX_Sunflower_corner_240m_v3_ExtendedDepthRange.GCD.txt', "a")

for i, line in enumerate(file):
	pieces = str.split(line)
	string = int(pieces[0])
	if string in strings_to_keep:
		outfile.write(line)

outfile.close()
