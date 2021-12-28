![](https://github.com/jinyier/EMSim/blob/main/doc/EMSim_LOGO.png)

- EMSim is designed to predict the EM emanations from ICs at the layout level.
- Version 1.0
- Contacts: 
    - Haocheng Ma : hc_ma@tju.edu.cn
    - Yier Jin : jinyier@gmail.com

# Table of contents
- [Prerequisites](#prerequisites)
- [Running EMSim](#running-EMSim)
    - [Data Preparation](#data-preparation)
    - [Current Analysis](#current-analysis)
    - [Electromagnetic Computation](#electromagnetic-computation)
- [Contributing](#contributing)
- [License](#license)

# Prerequisites
At a minimum:

- Python 3.8+ with PIP
- VCS, Calibre xRC, Primetime PX, HSpice
- Linux or Windows

# Running EMSim
EMSim consists of three main steps: data preparation, current analysis and EM computation.

<table>
  <tr>
    <td  align="center"><img src="./doc/Flow.jpg" ></td>
  </tr>
</table>

## Data Preparation

A RTL-to-GDS flow is a prerequisite to creating a layout database, which provides input data for EMSIM.

```
design.gds                # GDSII data of physical layout
design.v                  # layout-level netlists in Verilog
design.def                # physcial design data in DEF
design.sdc                # timing constraints
design.sdf                # timing data to specify interconnect delay and cell dealys
design.spef               # parasitic data in SPEF
```

## Current Analysis

Current analysis aims to simulate transient currents flowing in power grids of ICs.

### Extract Detailed Parasitics

```
generate_lvs_rule.py
optional arguments:
  [ --help ]                   see help
  [ --def_path ]               path to the def file, should end in .def
  [ --hcell_path ]             path to the output hcell file
  [ --xcell_path ]             path to the output xcell file
  [ --lvs_rule_path ]          path to the output lvs_rule file
```

### Analyze Logic Power

### Hybridize Spice Model

## Electromagnetic Computation


# Contributing


# License
