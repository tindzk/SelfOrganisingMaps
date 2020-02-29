# SelfOrganisingMaps

Implementation of GCAL model reported in Stevens et al., (2013) J. Neurosci. paper.

First install [morphologica](https://github.com/ABRG-Models/morphologica) and [Abseil](https://abseil.io/), then build in the usual cmake way:

```shell
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=/usr/local ..
make
cd ..
```

`/usr/local` refers to the path where morphologica and other custom dependencies were installed.

Then run model using e.g.:

```shell
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
build/sim/gcal configs/config.json --seed=1 --mode=1 --input=2
```

The final 3 numbers are:
1. random seed
2. Mode -- 0: displays-off, 1: displays-on, 2: map-only
3. Training pattern type -- 0: oriented Gaussians, 1: preloaded vectors*, 2: videocamera input

*Note that if using preloaded vectors you will need to supply a hdf5 file as the "patterns" parameter in the JSON file. An example can be generated as follows:

```shell
cd configs/
python -m venv venv
source venv/bin/activate
pip install -r  requirements.txt
python genTestPatterns.py
```

This creates the file `configs/testPatterns.h5`.

If the path to file of a saved weightfile is optionally appended, these weights will be used, else initial weights are random.

Enjoy!

## Troubleshooting
### hdf5 not found
If `cmake` fails because it could not find hdf5, the pkg-config file `hdf5.pc` may be missing:

```
-- Have pkg-config, searching for libmorphologica...
-- Checking for module 'libmorphologica'
--   Package 'hdf5', required by 'libmorphologica', not found
```

This can be solved by copying the version-tagged `.pc` file to `hdf5.pc`. Example:

```shell
cp /usr/lib/pkgconfig/hdf5-1.10.5.pc $CMAKE_INSTALL_PREFIX/lib/pkgconfig/hdf5.pc
```
