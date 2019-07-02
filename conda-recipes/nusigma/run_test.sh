
# nusigma signals its displeasure by calling exit(0), so we
# checky success by another route
check=$(
python <<-EOF | tail -n1
import nusigma
import numpy

numpy.testing.assert_allclose(nusigma.nucross(1e5,1,"p","CC",1), 1.80246030548e-34)
print("checkychecky")
EOF
)
if [ "$check" != "checkychecky" ]; then
    exit 1
fi