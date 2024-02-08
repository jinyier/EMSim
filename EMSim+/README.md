- EMSim+ accelerates the prediction of EM emanations by introducing an generative adversarial network (GAN) based on EMSim.
- Version 1.0
- Contacts: 
    - Haocheng Ma : hc_ma@tju.edu.cn
    - Yier Jin : jinyier@gmail.com

<table>
  <tr>
    <td  align="center"><img src="../doc/EMSim+ vs. EMSim.jpg" ></td>
  </tr>
</table>
	
	
# Table of contents
- [Prerequisites](#prerequisites)
- [Running EMSim+](#running-emsim)
    - [Current Analysis](#current-analysis)
    - [Electromagnetic Computation](#electromagnetic-computation)
    - [Feature Extraction](#feature-extraction)	
    - [GAN Model Training](#GAN-model-training)
    - [EM Prediction](#EM-prediction)		
- [Contributors](#contributors)
- [Copyright](#copyright)

# Prerequisites
At a minimum:

- Python 3.8+ with PIP
- TensorFlow2.4 with GPU
- Linux or Windows

# Running EMSim
EMSim+ consists of three main steps: feature extraction, GAN model training and EM prediction.

<table>
  <tr>
    <td  align="center"><img src="../doc/EMSim+ Flow.jpg" ></td>
  </tr>
</table>


## Current Analysis and Electromagnetic Computation

These two steps are from EMSim and are intended to subsequently produce training data for GANs.

## Feature Extraction

Feature extraction aims to extract cell current, power grid and EM information from the database of the chip physical layout.
Then, we convert them into feature maps.

### Cell Current Map

```
create_current_map.py
optional arguments:
  [ --def_path ]                   path to the def file, should end in .def
  [ --current_path ]               path to the simulated logic cell currents
  [ --num_input_stimuli ]          number of plaintexts
  [ --target_area_x ]              target simulated area in x axial direction  
  [ --target_area_y ]              target simulated area in y axial direction
  [ --num_probe_x_tiles ]          number of point grid in x axial direction
  [ --num_probe_y_tiles ]          number of point grid in y axial direction
  [ --layout_min_x ]               Reference coordinate in x axial direction
  [ --layout_min_y ]               Reference coordinate in y axial direction
  [ --start_points ]               Start of sample point for each current trace
  [ --sample_points ]              End of sample point for each current trace
  [ --current_map_train ]          generate cell current map for GAN training
  [ --current_map_test ]           generate cell current map for EM prediction
  
```

### Power Grid Map

```
create_grid_map(2-pad).py
optional arguments:
  [ --metal_layers ]               target metal layers
  [ --target_area_x ]              target simulated area in x axial direction  
  [ --target_area_y ]              target simulated area in y axial direction
  [ --num_probe_x_tiles ]          number of point grid in x axial direction
  [ --num_probe_y_tiles ]          number of point grid in y axial direction
  [ --layout_min_x ]               Reference coordinate in x axial direction
  [ --layout_min_y ]               Reference coordinate in y axial direction
  [ --power_grid_map ]             generate power grid map for GAN training and EM prediction
```

```
create_grid_map(4-pad).py
optional arguments:
  [ --metal_layers ]               target metal layers
  [ --target_area_x ]              target simulated area in x axial direction  
  [ --target_area_y ]              target simulated area in y axial direction
  [ --num_probe_x_tiles ]          number of point grid in x axial direction
  [ --num_probe_y_tiles ]          number of point grid in y axial direction
  [ --layout_min_x ]               Reference coordinate in x axial direction
  [ --layout_min_y ]               Reference coordinate in y axial direction

```
There is only one power grid map of one layout design, which is used in both the model training and EM prediction phases.
- Note:
    - You need to get the coordinates of VDD and VSS in the layout design.
    - You should choose different scripts for 2-pad and 4-pad power supply designs.



### EM Map

```
create_em_map.py
optional arguments:
  [ --EM_path ]                    path to the simulated EM data
  [ --num_input_stimuli ]          number of plaintexts
  [ --num_probe_x_tiles ]          number of point grid in x axial direction
  [ --num_probe_y_tiles ]          number of point grid in y axial direction
  [ --time_steps ]                 simulation time for each EM trace
  [ --em_map_train ]               generate em map for GAN training
  [ --em_map_test ]                generate em current map for EM prediction
```



## GAN Model Training


We aim to design and train a GAN for EM prediction.
- Note:
    - The generator G accepts cell current maps, power grid maps and time sequence.
    - Both the EM maps predicted by G and the real EM maps, together with the input maps of G, are alternatively fed to the discriminator D for determination.
    - The results of D are further fed back to G to enhance the quality of the predicted EM maps


```
GAN4EM_training.py
optional arguments:
  [ --num_probe_x_tiles ]          number of point grid in x axial direction
  [ --num_probe_y_tiles ]          number of point grid in Y axial direction
  [ --time_steps ]                 simulation time for each EM trace
  [ --percent_valid_split ]        10% of training points is used for validation during training
  [ --learning_rate_val ]          learning rate decays exponentially from 0.0005
  [ --num_input_stimuli ]          number of plaintexts
  [ --decay_rate_val ]             learning rate with the discount factor 0.98
  [ --epochs ]                     number of times the model worked on the entire training dataset
  [ --batch_size ]                 number of samples processed per training epoch
  [ --trained_model_path ]         save the trained weight parameters
  [ --training_current_map ]       input cell current map for model training
  [ --training_gird_map ]          input power grid map for model training
  [ --training_em_map ]            input EM map for model training 

```
## EM Prediction

The generator G is preserved and serves as an inference model for EM prediction.

```
GAN4EM_prediction.py
optional arguments:
  [ --num_probe_x_tiles ]          number of point grid in x axial direction
  [ --num_probe_y_tiles ]          number of point grid in Y axial direction
  [ --time_steps ]                 simulation time for each EM trace
  [ --percent_valid_split ]        10% of training points is used for validation during training
  [ --learning_rate_val ]          learning rate decays exponentially from 0.0005
  [ --num_input_stimuli ]          number of plaintexts
  [ --decay_rate_val ]             learning rate with the discount factor 0.98
  [ --trained_model_path ]         trained weight parameters
  [ --test_current_map ]           input cell current map for EM prediction
  [ --test_gird_map ]              input power grid map for EM prediction
  [ --test_em_map ]                input EM map for evaluate GAN model 


```
# Contributors

| Name         | Affiliation           | email                                                     |
| ------------ | --------------------- | --------------------------------------------------------- |
| Haocheng Ma  | Tianjin University    | [hc_ma@tju.edu.cn](mailto:hc_ma@tju.edu.cn)               |
| Yier Jin     | University of Florida | [jinyier@gmail.com](mailto:jinyier@gmail.com)             |
| Max Panoff   | University of Florida | [m.panoff@ufl.edu](mailto:m.panoff@ufl.edu)               |
| Jiaji He     | Tianjin University    | [dochejj@tju.edu.cn](mailto:dochejj@tju.edu.cn)           |
| Ya Gao       | Tianjin University    | [gaoyaya@tju.edu.cn](mailto:gaoyaya@tju.edu.cn)           |

# Copyright

It is mainly intended for non-commercial use, such as academic research.

# Citation

If you utilize EMSim+ in your research, we kindly request citation of the respective publication：https://ieeexplore.ieee.org/abstract/document/10323883

```
@INPROCEEDINGS{10323883,
  author={Gao, Ya and Ma, Haocheng and Kong, Jindi and He, Jiaji and Zhao, Yiqiang and Jin, Yier},
  booktitle={2023 IEEE/ACM International Conference on Computer Aided Design (ICCAD)}, 
  title={EMSim+: Accelerating Electromagnetic Security Evaluation with Generative Adversarial Network}, 
  year={2023},
  volume={},
  number={},
  pages={1-8},
  keywords={Solid modeling;Semiconductor device measurement;Design automation;Simulation;Predictive models;Generative adversarial networks;Silicon;CAD for Security;Side-Channel Analysis;Generative Adversarial Network},
  doi={10.1109/ICCAD57390.2023.10323883}}
```

