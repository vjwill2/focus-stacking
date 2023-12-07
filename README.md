# CS 445 Final Project - Focus Stacking

An implemenation of laplacian focus stacking and 3d depth mapping from an image stack. 

## Getting Started

### Prerequisites

* Ensure you have a Python 3 environment (Tested with 3.11) set up with the following packages:
  * matplotlib
  * numpy
  * opencv
  * sciki-image
* An `environment.yaml` config file is provided for conda environments

### Usage

To generate all project results, run:

```bash
python generate_results.py
```

* Results can then be found in `./results_naive` for naive approach, `./results_laplacian` for laplacian, and `./results_stack` for 3d depth mapping
* Exact arguments used to generate results for each method are defined in the `generate_results.py` script

### About

* `depth_mapping.py` - Script containing 3D depth mapping implementation
* `naive_approach.py` - Script containing naive focus stacking implementation
* `focus_stacking.py` - Script containing laplacian focus stacking implementation
* `generate_results.py` - Script containing arguments 

## Authors

* Bennett Wu
* Deeya Bodas
* Victor Will

## Acknowledgments

A list of resources and references used in the development of the project

* Laplacian Focus Stacking: Adelson, Edward & Anderson, Charles & Bergen, James & Burt, Peter & Ogden, Joan. (1983). Pyramid Methods in Image Processing. RCA Eng.. 29. 

* Depth Mapping: J. Wlodek, K. J. Gofron, Y. Q. Cai; Achieving 3D imaging through focus stacking. AIP Conf. Proc. 15 January 2019; 2054 (1): 050001. https://doi.org/10.1063/1.5084619

* ECC Image Alignment: https://learnopencv.com/image-alignment-ecc-in-opencv-c-python/

* Bug Images: https://github.com/bznick98/Focus_Stacking

* Ant Images: http://grail.cs.washington.edu/projects/photomontage/

* Depth Map Visualization: https://depthmapviewer.ugocapeto.com/
  
