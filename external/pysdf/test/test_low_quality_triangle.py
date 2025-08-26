import pysdf.sdf as pysdf
import numpy as np
from pyevtk.hl import polyLinesToVTK

def test_lqt(eps=1e-2):

    lqt = np.array([0, 0, 0, # bottom
                    eps, -3.14159*eps, 0,
                    0, 1, 0], dtype=np.float64)

    # tests a long and skinny triangle.
    
    lqt = np.array([-5, -0.5, 0.5625,
                    -0.75, -0.5, 0.4375,
                    -0.1, -0.5, 0.4375])

    # points is opposite from triangle normal, which tests a fixed bug for points inside object.
    points = np.array([-0.984955, -0.493648, 0.453559])

    sdf_lqt, hit_lqt = pysdf.distance_field(lqt, points, nr_sphere_points=1000, include_hit_points=True)

    x = np.zeros(2 * len(points)//3)
    y = np.zeros(2 * len(points)//3)
    z = np.zeros(2 * len(points)//3)
    points_x = np.array([points[0]])
    points_y = np.array([points[1]])
    points_z = np.array([points[2]])
    xh = hit_lqt[0::3]
    yh = hit_lqt[1::3]
    zh = hit_lqt[2::3]
    x[0::2] = points_x
    x[1::2] = xh
    y[0::2] = points_y
    y[1::2] = yh
    z[0::2] = points_z
    z[1::2] = zh
    polyLinesToVTK("sdf_hit_single", x, y, z, 2*np.ones(x.size//2))
    tx = lqt[::3].tolist()
    tx.append(lqt[0])
    ty = lqt[1::3].tolist()
    ty.append(lqt[1])
    tz = lqt[2::3].tolist()
    tz.append(lqt[2])
    polyLinesToVTK("lqt",
                   np.array(tx),
                   np.array(ty),
                   np.array(tz),
                   np.array([4]))
    sdf_error = abs(sdf_lqt[0] + (-0.5 - points[1]))

    np.testing.assert_allclose(sdf_error, 0.0, atol=1e-7)

    return sdf_lqt, hit_lqt

if __name__ == "__main__":
    test_lqt()
