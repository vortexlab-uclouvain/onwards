Welcome to OnWaRDS!
===================

*Online Wake Rapid Dynamic Simulator* - Maxime Lejeune - UCLouvain 2022

![image](docs/source/ONWARDS.svg)

Quick start
-----------
``install.sh`` will guide you through the installation process (set up environment variables and compile sources).

```bash:
git clone git@github.com:vortexlab-uclouvain/onwards.git
cd onwards
source install.sh
```

OnWaRDS was developed using Python 3.9. The following modules are required:
* matplotlib 3.4.3
* numpy      1.21.2
* scipy      1.6.1

``templates`` provides a few OnWaRDS configurations examples and will guide you through the different steps required to perform a farm simulation. 

```bash:
cd $ONWARDS_PATH/templates
python 00_sandbox.py
```

Documentation
-------------
Documentation available [here](https://onwards.git-page.immc.ucl.ac.be/onwards).

Reference data
--------------
Data available [here](https://doi.org/10.14428/DVN/AUVUI6)

Publications
------------

[\[1\]](https://doi.org/10.3389/fenrg.2022.884068) M. Lejeune, M. Moens, and P. Chatelain. *A meandering-capturing wake model coupled to rotor-based flow-sensing for operational wind farm flow prediction*. Frontiers in Energy Research, 10, jul 2022.

[\[2\]](https://doi.org/10.1088/1742-6596/2265/2/022018) M. Lejeune, M. Moens, and P. Chatelain. *Extension and validation of an operational dynamic wake model to yawed configurations*. Journal of Physics: Conference Series, 2265(2):022018, may 2022.

[\[3\]](https://doi.org/10.1088/1742-6596/1618/6/062055) M. Lejeune, M. Moens, M. Coquelet, N. Coudou, and P. Chatelain. *Data assimilation for the prediction of wake trajectories within wind farms*. Journal of Physics: Conference Series, 1618:062055, sep 2020.

[\[3\]](http://hdl.handle.net/2078.1/275067) M. Lejeune. *A meandering-capturing wake model coupled to rotor-based flow-sensing for operational wind farm flow estimation*. PhD thesis, UCLouvain, 2023.

How to cite
-----------
If you use OnWaRDS in a scientific publication, please use the following citation:
```
@article{Lejeune22,
	author = {Maxime Lejeune and Maud Moens and Philippe Chatelain},
	journal = {Frontiers in Energy Research},
	month = {jul},
	title = {A Meandering-Capturing Wake Model Coupled to Rotor-Based Flow-Sensing for Operational Wind Farm Flow Prediction},
	volume = {10},
	year = {2022}
}
```

License
-------

This program (OnWaRDS) is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program (see COPYING file).  If not, see <https://www.gnu.org/licenses/>.

*************************************************************************
The documentation for this program is under a creative commons attribution share-alike 4.0 license.  http://creativecommons.org/licenses/by-sa/4.0/

*************************************************************************
A dataset for testing this program is provided under a Creative Commons attribution-NonCommercial-NoDerivatives 4.0 International License.
https://creativecommons.org/licenses/by-nc-nd/4.0/

Acknowledgements
----------------

This project has received funding from the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation program (grant agreement no. 725627). Simulations were performed using computational resources provided by the Consortium des Équipements de Calcul intensif (CÉCI), funded by the Fonds de la Recherche Scientifique de Belgique (F.R.S.- FNRS) under Grant No. 2.5020.11, and computational resources made available on the Tier-1 supercomputer of the Fédération Wallonie-Bruxelles, infrastructure funded by the Walloon Region under the Grant Agreement No. 1117545.

✉️ [Contact Us](mailto:maxime.lejeune@uclouvain.be)
