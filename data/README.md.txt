## Data Availability and Description

This study uses the PeMSD4 and PeMSD8 traffic datasets provided by the
California Department of Transportation Performance Measurement System (PeMS).

### Data Source
The datasets are publicly available from the official PeMS platform:
http://pems.dot.ca.gov/

PeMS provides large-scale, real-world traffic measurements collected from
loop detectors deployed on California freeways.

### Dataset Description
The PeMSD4 and PeMSD8 datasets contain traffic measurements sampled at
5-minute intervals. Each record includes the following attributes:

- Traffic flow
- Traffic occupancy
- Traffic speed

The data are organized by sensor (detector) and timestamp, forming
multivariate spatiotemporal sequences suitable for traffic flow prediction.

### Data Preprocessing
Raw PeMS data are provided in structured files (e.g., `.dyna` format).
Preprocessing includes temporal alignment, normalization, and construction
of sliding windows for supervised learning.

All preprocessing steps are implemented in the provided codebase.

### Data Usage and Redistribution
Due to data size and licensing constraints, the raw datasets are not
redistributed in this repository.

Researchers can obtain the original data directly from the PeMS website
and reproduce all experiments by following the instructions in the
main `README.md` file.

### Reproducibility Statement
All scripts required for data preprocessing, training, and evaluation
are publicly available in this repository. The absence of raw data does
not affect the reproducibility of the proposed methodology.
