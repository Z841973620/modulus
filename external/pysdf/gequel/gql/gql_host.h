


namespace gql {
  
  // ==================================================================
  // POINT set geometries
  // ==================================================================
  GQLPoints createPointSet(const vec3f *points, size_t count);

  /*! given a data set of points, find, for each query point, the
      closest point in that point set (within a given max query
      radius); and return, for each query point, the index of the
      point in the point data set */
  void closestPoint(GQLPoints points,
                    int   *queryResult,
                    vec3f *queryPoints,
                    size_t queryCount,
                    float  maxRadius);
  
  // ==================================================================
  // TRIANGLE soup geometries
  // ==================================================================
  GQLTriangles createTriangles(const vec3f *vertexArray,
                               size_t vertexArrayCount,
                               const vec3i *indexArray,
                               size_t indexArrayCount);

  struct ClosestSurfacePointResult {
    vec3f P;
    int   triID;
  };

  /*! given a set of query points, find, for each one, the closest
      point on any triangle of a given trianel surface; returning, for
      each, both the triangle index (within that surface) and the 3D
      point itself */
  void closestSurfacePoint(GQLTriangles geometry,
                           ClosestSurfacePointResult *queryResults,
                           vec3f                     *queryPoints,
                           size_t                     queryCount,
                           bool need_offload
                           );
  
} // ::gql
