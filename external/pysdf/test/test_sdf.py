# test

import pysdf.sdf as sdf
import numpy as np


def tet_verts(flip_x=1):
    tet = np.array([flip_x*0, 0, 0, # bottom
                    flip_x*0, 1, 0,
                    flip_x*1, 0, 0,
                    flip_x*0, 0, 0, # front
                    flip_x*1, 0, 0,
                    flip_x*0, 0, 1,
                    flip_x*0, 0, 0, # left
                    flip_x*0, 0, 1,
                    flip_x*0, 1, 0,
                    flip_x*1, 0, 0, # "top"
                    flip_x*0, 1, 0,
                    flip_x*0, 0, 1], dtype=np.float64)

    return tet

def test_sdf():

    tet = tet_verts()
                        
    (sdf_tet, sdf_hit_pts) = sdf.signed_distance_field(tet, np.array([1, 1, 1, 0.1, 0.1, 0.1], dtype=np.float64), include_hit_points=True)
    print("results:", sdf_tet, sdf_hit_pts)
    np.testing.assert_allclose(sdf_tet, [ 1.15470052, -0.1       ], atol=1e-7)

    (sdf_tet, sdf_hit_pts) = sdf.signed_distance_field(tet, np.array([1, 1, 1, 0.12, 0.11, 0.1], dtype=np.float64),
                                                       include_hit_points=True)
    print("sdf_hit_pts=", sdf_hit_pts)
    np.testing.assert_allclose(sdf_hit_pts,
                               [0.33333328, 0.33333334, 0.33333334, 0.12,       0.11,       0.,        ],
                               atol=1e-7)

def test_df():

    tet = tet_verts()

    (df_tet, df_hit) = sdf.distance_field(tet, np.array([1, 1, 1, 0.1, 0.1, 0.1], dtype=np.float64), include_hit_points=True)
    print("results:", df_tet, df_hit)
    np.testing.assert_allclose(df_tet, [ 1.15470052, 0.1       ], atol=1e-7)


def test_inside_outside():
    tet = tet_verts()

    sf_tet = sdf.inside_outside(tet, np.array([1, 1, 1, 0.1, 0.1, 0.1], dtype=np.float64), nr_sphere_points=100)
    print("results:", sf_tet)
    np.testing.assert_allclose(sf_tet, np.array([0, 1]))


def test_intersection():

    # create set of co-planar test triangles
    # triangles = np.array([0, 1, 0,
    #                       -1, -1, 0,
    #                       1, -1, 0,
    #                       0, 1, 0,
    #                       -1, -1, 0,
    #                       1, -1, 0,
    #                       0, 1, 0,
    #                       -1, -1, 0,
    #                       1, -1, 0], dtype=np.float64)

    tet = tet_verts()
    tetflipped = tet_verts(flip_x=-1)
    tet_both = np.concatenate([tet, tetflipped])

    sf_tets = sdf.inside_outside(tet_both, np.array([1.0, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float64), nr_sphere_points=10)
    print("sf_tets=", sf_tets)
    np.testing.assert_allclose(sf_tets, np.array([0, 1]))

if __name__ == "__main__":
    test_sdf()
    test_df()
    test_inside_outside()
    test_intersection()
