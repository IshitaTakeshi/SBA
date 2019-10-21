# Sparse Bundle Adjustment (SBA)

[![Build Status](https://travis-ci.org/IshitaTakeshi/SBA.svg?branch=master)](https://travis-ci.org/IshitaTakeshi/SBA)
[![codecov](https://codecov.io/gh/IshitaTakeshi/SBA/branch/master/graph/badge.svg)](https://codecov.io/gh/IshitaTakeshi/SBA)

[Documentation](https://sparsebundleadjustment.readthedocs.io/en/master/)

Python implementation of Sparse Bundle Adjustment.

Bundle Adjustment is a problem of minimizing the reprojection error concerning all 3D points and camera pose parameters, which is usually solved by the Gauss-Newton or Levenberg-Marquardt method.  
These update methods are suffered from the computation cost because inverting the approximated Hessian is highly expensive.  
Sparse Bundle Adjustment is an algorithm that can calculate the update efficiently by employing the sparse structure of the Jacobian.


```
@article{lourakis2009sba,
  title={SBA: A software package for generic sparse bundle adjustment},
  author={Lourakis, Manolis IA and Argyros, Antonis A},
  journal={ACM Transactions on Mathematical Software (TOMS)},
  volume={36},
  number={1},
  pages={2},
  year={2009},
  publisher={ACM}
}
```
