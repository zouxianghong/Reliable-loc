from setuptools import setup
#import sys
#print(sys.path)
#sys.path = ['', '/opt/software/anaconda3/lib/python37.zip', '/opt/software/anaconda3/lib/python3.7', '/opt/software/anaconda3/lib/python3.7/lib-dynload','/opt/software/anaconda3/lib/python3.7/site-packages','/home/TrueC/.local/lib/python3.7/site-packages', '/home/TrueC/Git/VideoSuperResolution']

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='emd',
    ext_modules=[
        CUDAExtension('emd', [
            'emd.cpp',
            'emd_cuda.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })