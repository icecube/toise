
muon_performance.pdf:
	gen2-plot detector.tracks.performance -o $@

surface_geometry.pdf:
	gen2-plot detector.surface_geometry -o $@

cascade_aeff.IceCube.json.gz:
	gen2-figure-data detector.cascades.effective_area -d IceCube 0 -o $@

cascade_aeff.Gen2-InIce.json.gz:
	gen2-figure-data detector.cascades.effective_area -d Gen2-InIce 0 -o $@

cascade_aeff.pdf : cascade_aeff.IceCube.json.gz cascade_aeff.Gen2-InIce.json.gz
	gen2-plot detector.cascades.effective_area $^ -o $@

sparse_veto.pdf :
	gen2-plot detector.cascades.sparse_veto -o $@

.PHONY: all

all: muon_performance.pdf surface_geometry.pdf cascade_aeff.pdf sparse_veto.pdf