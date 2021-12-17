
all: icecube_8yr.json.gz gen2-inice_10yr.json.gz

gen2-inice_10yr.json.gz:
	toise-figure-data diffuse.flavor.event_counts -d Gen2-InIce 10 -o $@

icecube_8yr.json.gz:
	toise-figure-data diffuse.flavor.event_counts -d IceCube 8 -o $@


