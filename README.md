# Shallow water wave model under location uncertainty
PyTorch implementation of stochastic shallow water wave model on 2D periodic domain.
This code can be used to reproduce the numerical results in the paper 
[@Mémin et al. (2023)](https://arxiv.org/abs/2304.10183).

Copyright 2023 Long Li.

## Description

In this project, we investigate the wave solutions of a stochastic rotating shallow water model. 
This approximate model provides an interesting simple description of the interplay between waves and 
random forcing ensuing either from the wind or coming as the feedback of the ocean on the atmosphere 
and leading in a very fast way to the selection of some wave-length. This interwoven, yet simple, 
mechanism explains the emergence of typical wavelength associated to near inertial waves. 
Ensemble-mean waves that are not in phase with the random forcing are damped at an exponential rate, 
whose magnitude depends on the random forcing variance. This codes allows to illustrate the plane wave
solutions associated to the linearized system under some specific noises.

## Getting Started

### Dependencies

* Prerequisites: Pytorch, Numpy, Matplotlib.

* Tested with Intel CPUs, NVIDIA RTX 2080Ti GPU and Tesla P100 GPU.

### Installing

```
git clone https://github.com/matlong/sw-wave-lu.git
```

### Executing program

* To run the deterministic wave model:
```
python3 run_det.py
```

* To run the stochastic wave model:
```
python3 run_sto.py
```

* To compare the damping effect due to different noises:
```
python3 compare_damping.py
```

<!---
## Help

Any advise for common problems or issues.
```
command to run if program contains helper info
```
-->

## Authors

Etienne Mémin, Long Li, Noé Lahaye, Gilles Tissot and Bertrand Chapron.

Contact: long.li@inria.fr

<!---
ex. Dominique Pizzie  
ex. [@DomPizzie](https://twitter.com/dompizzie)
-->

<!---
## Version History

* 0.2
    * Various bug fixes and optimizations
    * See [commit change]() or See [release history]()
* 0.1
    * Initial Release
-->

## Citation

```
@misc{mémin2023linear,
      title={Linear waves solutions of a stochastic shallow water model},
      author={Etienne Mémin and Long Li and Noé Lahaye and Gilles Tissot and Bertrand Chapron},
      year={2023},
      eprint={2304.10183},
      archivePrefix={arXiv},
      primaryClass={physics.flu-dyn},
      doi = {10.48550/ARXIV.2304.10183},
      url = {https://arxiv.org/abs/2304.10183}
}
```

## Acknowledgments

The authors acknowledge the support of the ERC EU project 856408-STUOD.

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.

<!---
Inspiration, code snippets, etc.
* [awesome-readme](https://github.com/matiassingers/awesome-readme)
* [PurpleBooth](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2)
* [dbader](https://github.com/dbader/readme-template)
* [zenorocha](https://gist.github.com/zenorocha/4526327)
* [fvcproductions](https://gist.github.com/fvcproductions/1bfc2d4aecb01a834b46)
-->
