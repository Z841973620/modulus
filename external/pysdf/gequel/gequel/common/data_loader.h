

#include "./p4.h"
#include "./tiny_obj_loader.h"

#include <thrust/device_vector.h>

#include <sys/stat.h>

namespace gequel {
  box3f makeAABBForOBJ(const std::string& filename) {
    box3f bounds;
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string error;
    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &error, filename.c_str());
    if (!error.empty()) throw std::runtime_error(error);

    for (int i = 1; i < attrib.vertices.size() / 3; i++) {
      bounds.extend(vec3f(attrib.vertices[3 * i + 0],
                          attrib.vertices[3 * i + 1],
                          attrib.vertices[3 * i + 2]));
    }
    return bounds;
  }

  // Rescale data from 2x2x2 box centered at origin to box in mins, maxs
  inline void rescaleData(std::vector<vec3f>& points, const box3f& bounds) {
    for (vec3f& p : points) {
      vec3f frac = (p + vec3f(1.f, 1.f, 1.f)) * 0.5f;
      p = bounds.lower + frac * bounds.span();
    }
  }

  inline std::vector<vec3f> createPoints_uniform(int N = 100000)
  {
    std::mt19937 gen(0x12345); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    
    std::vector<vec3f> points;
    for (int i=0;i<N;i++)
      points.push_back(vec3f(dis(gen),dis(gen),dis(gen)));
    return points;
  }


  inline std::vector<vec3f> createPoints_clusters(int N = 100000)
  {
#if 1
    static std::mt19937 gen(0x12345); //Standard mersenne_twister_engine seeded with rd()
#else
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
#endif
    std::uniform_real_distribution<> dis(-1.f, +1.f);

    int numSpheres = int(sqrtf(N))+1;
    int numPointsPerSphere = numSpheres;
    
    std::vector<vec3f> points;

    for (int sphereID=0;sphereID<numSpheres;sphereID++) {
      vec3f spherePos = vec3f(dis(gen),dis(gen),dis(gen));
      float sphereRad = 1e-3f + fabs(dis(gen)) * .1f;
      for (int i=0;i<numPointsPerSphere;i++) {
        vec3f v;
        while (1) {
          v = vec3f(dis(gen),dis(gen),dis(gen));
          if (dot(v,v) <= 1.f) break;
        }
        v = normalize(v);
        v = spherePos + v*sphereRad;
        points.push_back(v);
      }
    }
    return points;
  }

  inline std::vector<vec3f> createPoints_triangles(const std::string &filename,
                                                   int N = 100000) {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string err;
    bool ret =
        tinyobj::LoadObj(&attrib, &shapes, &materials, &err, filename.c_str());
    if (!err.empty())
      throw std::runtime_error(err);

    std::vector<int32_t> vertex_indices;
    for (const auto &s : shapes)
      for (const auto &i : s.mesh.indices)
        vertex_indices.push_back(i.vertex_index);

    std::vector<float> cumulativeAreas;
    float totalArea = 0;
    size_t num_triangles = vertex_indices.size() / 3;
    cumulativeAreas.reserve(num_triangles);
    for (size_t i = 0; i < num_triangles; i++) {
      vec3f a = vec3f(attrib.vertices[3 * vertex_indices[3 * i + 0] + 0],
                      attrib.vertices[3 * vertex_indices[3 * i + 0] + 1],
                      attrib.vertices[3 * vertex_indices[3 * i + 0] + 2]);
      vec3f b = vec3f(attrib.vertices[3 * vertex_indices[3 * i + 1] + 0],
                      attrib.vertices[3 * vertex_indices[3 * i + 1] + 1],
                      attrib.vertices[3 * vertex_indices[3 * i + 1] + 2]);
      vec3f c = vec3f(attrib.vertices[3 * vertex_indices[3 * i + 2] + 0],
                      attrib.vertices[3 * vertex_indices[3 * i + 2] + 1],
                      attrib.vertices[3 * vertex_indices[3 * i + 2] + 2]);
      vec3f areaNormal = cross(b - a, c - a);
      totalArea += sqrtf(dot(areaNormal, areaNormal)) / 2.f;
      cumulativeAreas.push_back(totalArea);
    }

    std::mt19937 gen(0x12345); // Standard mersenne_twister_engine
    std::uniform_real_distribution<> dis(0.0, 1.0);
    std::vector<vec3f> points;
    points.reserve(N);
    for (int i = 0; i < N; i++) {
      int pos = std::lower_bound(cumulativeAreas.begin(), cumulativeAreas.end(),
                                 dis(gen) * totalArea) -
                cumulativeAreas.begin();
      float u, v;
      while (true) {
        u = dis(gen);
        v = dis(gen);
        if (u + v <= 1.f) {
          break;
        }
      }
      vec3f a = vec3f(attrib.vertices[3 * vertex_indices[3 * pos + 0] + 0],
                      attrib.vertices[3 * vertex_indices[3 * pos + 0] + 1],
                      attrib.vertices[3 * vertex_indices[3 * pos + 0] + 2]);
      vec3f b = vec3f(attrib.vertices[3 * vertex_indices[3 * pos + 1] + 0],
                      attrib.vertices[3 * vertex_indices[3 * pos + 1] + 1],
                      attrib.vertices[3 * vertex_indices[3 * pos + 1] + 2]);
      vec3f c = vec3f(attrib.vertices[3 * vertex_indices[3 * pos + 2] + 0],
                      attrib.vertices[3 * vertex_indices[3 * pos + 2] + 1],
                      attrib.vertices[3 * vertex_indices[3 * pos + 2] + 2]);

      points.push_back(a + u * (b - a) + v * (c - a));
    }

    return points;
  }

  inline std::vector<vec3f> createPoints_grid(int N = 100000) {
    int dimSize = powf(N, 1.f / 3) + 1;

    std::vector<vec3f> points;
    points.reserve(dimSize * dimSize * dimSize);

    for (int z = 0; z < dimSize; z++) {
      float gz = float(z) / (dimSize - 1) * 2.f - 1;
      for (int y = 0; y < dimSize; y++) {
        float gy = float(y) / (dimSize - 1) * 2.f - 1;
        for (int x = 0; x < dimSize; x++) {
          float gx = float(x) / (dimSize - 1) * 2.f - 1;
          points.emplace_back(gx, gy, gz);
        }
      }
    }
    return points;
  }

  // Create points on the triangle, with some jitter along the normal direction
  inline std::vector<vec3f> createPoints_triangle(vec3f p1, vec3f p2, vec3f p3, int N = 100000)
  {
    static std::mt19937 gen(0x12345); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(0.f, 1.f);

    vec3f normal = normalize(cross(p2 - p1, p3 - p1));
    std::vector<vec3f> points;
    for (int i = 0; i < N; i++) {
      while (true) {
        float u = dis(gen);
        float v = dis(gen);
        if (u + v <= 1.f) {
          float dist = 0.5 * dis(gen) - 0.25;
          points.push_back(p1 + u*(p2-p1) + v*(p3-p1) + dist*normal);
          break;
        }
      }
    }
    return points;
  }

  inline std::pair<std::vector<vec3f>, std::vector<vec3i>> createPoints_obj(const std::string& objPath)
  {
    std::vector<vec3f> points;
    std::vector<vec3i> faces;

    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string err;

    tinyobj::attrib_t attrib;
    if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &err, objPath.c_str()))
      throw std::runtime_error( std::string("Error: Unable to load " + objPath));

    uint64_t numPts = attrib.vertices.size() / 3;
    points.resize(numPts);
    for (uint64_t i = 0; i < numPts; ++i) {
      points[i] = vec3f(
        attrib.vertices[i * 3 + 0],
        attrib.vertices[i * 3 + 1],
        attrib.vertices[i * 3 + 2]
      );
    }

    size_t numTris = shapes[0].mesh.indices.size() / 3;
    faces.resize(numTris);
    for (size_t i = 0; i < numTris; i++) {
      faces[i] = vec3i(
        shapes[0].mesh.indices[3 * i + 0].vertex_index,
        shapes[0].mesh.indices[3 * i + 1].vertex_index,
        shapes[0].mesh.indices[3 * i + 2].vertex_index
      );
    }
  
    return std::make_pair(std::move(points), std::move(faces));
  }

  inline std::vector<vec3f> createPoints_p4(const std::string& p4Path)
  {
    return readP4(p4Path.c_str());
  }
}
