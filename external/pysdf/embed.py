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

# This very short program embeds a PTX input into a "const char[]" in a specified header file.

import argparse
import string
import sys

def main(argv=None):
    if argv is None:
        argv = sys.argv

    parser = argparse.ArgumentParser()

    parser.add_argument("-o", "--output", type=str)
    parser.add_argument("variable", type=str)
    parser.add_argument("ptx_input", type=str)

    args = parser.parse_args(argv[1:])

    f = open(args.ptx_input,"r")
    ptx_file = f.read()
    f.close()
    f = open(args.output, "w")
    f.write("#pragma once\n")
    f.write("// WARNING: Auto-generated file created by build-system. Will be updated whenever PTX source ('{}') is updated\n".format(args.ptx_input))
    # write the source into the char-array using C++11 multiline raw string literal support.
    f.write("const char {}[] = \nR\"({}\n)\";\n".format(args.variable, ptx_file))

    f.close()

if __name__ == '__main__':
    main()
