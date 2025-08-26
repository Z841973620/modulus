# GeQueL - Geometry Query Library

Some test kernels for geometry queries - closest point, 
closest surface point on triangle, kNN, etc


# Targeted Backends

currently envisioing the following backends:

- cpu via "manual" data strcutures (eg, balanced kd-tree)

- cpu via embree bvh

- gpu via cuda

- gpu on optix bvh

based on backend we probably have to differentiate between

- 'host-only' - dataset and queires all live on host

- 'gpu-offload' - dataset and/or qeueries live on host, get copied to device as required

- 'gpu-native - both dataset and queries live on device, get exeuted on device


# Currently Targeted Kernels

kernels currently on radar:

- fcp: find "the" closest point (in set of points) to gtiven vec3f query location.

	- return ID of point, and distance?

- kNN: find k nearest points (wihtin set of points) to given vec3f
  query location (restricted by max query radius)

	- return the N point IDs
	
	- may need variants with hardcoded fixed k?


- point range query: find all points (within provided maximum number Nmax) within point data set

    - might also want triangles range query? (ie, data set is triangles, not poitns)
	
	- if query radius contains more than Nmax points we will NOT
      guaranteee that those will be the Nmax closest; return only
      Nmax, withour any guarantees as to which ones will be reutrned.

- cpat: closest point on any triangle: 

	- input is set of triangels, queries are 3D points
	
	- computes for each query the ID of the triangle that conaits the
      closest surface point, and possibly the point itself, and/or
      barycentric coordinates of that point

