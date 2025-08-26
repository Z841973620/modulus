# Signed Distance Field (SDF) Library

These folders contain a compiled SDF library, demo application, and the source to re-create the demo.

## Library

The library, `lib/libsdf.so` provides a C interface to the SDF library, with the particular API exposed in `include/sdf.h`. Please see the header's doxygen comments for precise details on usage.

## Demo executable

The SDF demo executable `bin/demo` takes an OBJ triangulated mesh as input, and outputs a grid of points with the SDF as values in VTK format. This can be opened in Paraview for visualization.

### Demo source

The source and example `cmake` build scripts are located in `demo_src/`. `demo_src/main.cpp` includes the necessary functions to read the OBJ input, interface with `libsdf` and output the SDF field grid as a VTK file. Building follows a standard cmake project, however a recent version of CMake is required (3.17+)

```
cd demo_src
mkdir build && cd build
cmake ../
make
./demo ../bunny.obj
```
