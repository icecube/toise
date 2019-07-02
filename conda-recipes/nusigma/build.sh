if [ ${target_platform:0:3} = "osx" ]; then
    export LDFLAGS="$LDFLAGS -Wl,-undefined,dynamic_lookup"
fi

FCFLAGS=$FFLAGS ./configure
make libnusigma
cp $RECIPE_DIR/setup.py $RECIPE_DIR/*.pyf $RECIPE_DIR/*.f .
cp -r $RECIPE_DIR/nusigma .
mkdir nusigma/dat
cp -r dat/*.tbl dat/*.pds nusigma/dat
python $RECIPE_DIR/setup.py install