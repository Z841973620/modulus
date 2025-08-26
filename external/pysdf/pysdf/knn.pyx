# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 

import numpy as np
cimport numpy as np

import ctypes

cdef extern from "knn.h":
    void findKnn(int num_all_points, double* all_points,
                 int num_query_points, double* query_points,
                 int k,
                 double* result_points_all,
                 int* num_result_points_per_query)


def find_knn(np.ndarray[double, ndim=1, mode="c"] all_points,
             np.ndarray[double, ndim=1, mode="c"] query_points,
             k):


    cdef int num_all_points = len(all_points)/3
    cdef int num_query_points = len(query_points)/3

    cdef np.ndarray[double] tmp_results = np.zeros(num_query_points * 3 * k)
    cdef np.ndarray[int] num_results = np.zeros(num_query_points, dtype=np.int32)

    findKnn(num_all_points, &all_points[0], num_query_points, &query_points[0], k,
            &tmp_results[0], &num_results[0])

    final_results = []
    total_points = 0
    for i in range(num_query_points):
        query_result = np.zeros(3*num_results[i])
        for j in range(num_results[i]):
            query_result[3*j+0] = tmp_results[3*total_points+0]
            query_result[3*j+1] = tmp_results[3*total_points+1]
            query_result[3*j+2] = tmp_results[3*total_points+2]
            total_points += 1
        
        final_results.append(query_result)

    return final_results

