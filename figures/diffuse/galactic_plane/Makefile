
ic_dnn_10y_kra_5.json.gz: Makefile
	toise-figure-data diffuse.galactic.fermi_pi0 -d IceCube-DNNC 10 --galactic-emission kra_5 --angular-smoothing dnn -o ic_dnn_10y_kra_5

ic_monopod_80y_kra_5w.json.gz: Makefile
	toise-figure-data diffuse.galactic.fermi_pi0 -d IceCube-DNNC 80 --galactic-emission kra_5 --angular-smoothing monopod -o ic_monopod_80y_kra_5

ic_dnn_10y_kra_powerlaw.json.gz: Makefile
	toise-figure-data diffuse.galactic.fermi_pi0 -d IceCube-DNNC 10 --galactic-emission kra_powerlaw --null-hypothesis kra_5 --angular-smoothing dnn -o ic_dnn_10y_kra_powerlaw --free-components astro galactic_bg

ic_monopod_80y_kra_powerlaw.json.gz: Makefile
	toise-figure-data diffuse.galactic.fermi_pi0 -d IceCube-DNNC 80 --galactic-emission kra_powerlaw --null-hypothesis kra_5 --angular-smoothing monopod -o ic_monopod_80y_kra_powerlaw --free-components astro galactic_bg
