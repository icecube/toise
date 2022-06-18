#!/usr/bin/env python

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("gcd_file")
parser.add_argument("text_file")
args = parser.parse_args()

import numpy as np
from icecube import icetray, dataclasses, dataio

from icecube import icetray, dataio, dataclasses

f = dataio.I3File(args.gcd_file)
fr = f.pop_frame(icetray.I3Frame.Geometry)
f.close()

geo = []
for key, omgeo in fr["I3Geometry"].omgeo:
    if omgeo.omtype == omgeo.IceTop:
        continue
    pos = omgeo.position
    geo.append([key.string, key.om, pos.x, pos.y, pos.z])

np.savetxt(args.text_file, geo, fmt="%d %d %.2f %.2f %.2f")
