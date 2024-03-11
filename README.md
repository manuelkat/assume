<!--
SPDX-FileCopyrightText: ASSUME Developers

SPDX-License-Identifier: AGPL-3.0-or-later
-->

# SPARKLE - Simulation Platform for Agent-based Research on energy marKet dynamics for Lower Emissions


**SPARKLE** is an open-source toolbox for agent-based simulations of European electricity markets, with a primary focus on the German market setup. Developed as an open-source model, its primary objectives are to ensure usability and customizability for a wide range of users and use cases in the energy system modeling community.

## Relation to ASSUME

This is a fork of ASSUME, created to focus on the results of my PhD thesis. It is published under a different name, so that my thesis is not too tightly coupled to the ASSUME research project, as I do not receive funding from it.

I plan to add all features to the upstream assume project, where I am also one of the core maintainers and developers.
After finishing my PhD, SPARKLE will not be maintained any longer, so you should surely just use the ASSUME Framework in the first place.

For convenience reasons, the package name and cli name stays `assume` to reduce the unneeded changes towards the maintained version.


## Documentation

- [User Documentation](https://assume.readthedocs.io/en/latest/)
- [Installation Guide](https://assume.readthedocs.io/en/latest/installation.html)

## Installation

You can install SPARKLE using pip. Choose the appropriate installation method based on your needs:

### Using pip

To install the core package:

```bash
pip install -e .
```

To install with additional capabilities:

```bash
pip install -e .[full]
```

### Timescale Database and Grafana Dashboards

If you want to benefit from a supported database and integrated Grafana dashboards for scenario analysis, you can use the provided Docker Compose file.

Follow these steps:

1. Clone the repository and navigate to its directory:

```bash
git clone https://github.com/maurerle/sparkle.git
cd sparkle
```

2. Start the database and Grafana using the following command:

```bash
docker-compose up -d
```

This will launch a container for TimescaleDB and Grafana with preconfigured dashboards for analysis. You can access the Grafana dashboards at `http://localhost:3000`.

### Using Learning Capabilities

If you intend to use the reinforcement learning capabilities of ASSUME and train your agents, make sure to install Torch. Detailed installation instructions can be found [here](https://pytorch.org/get-started/locally/).



## Trying out ASSUME and the provided Examples

To ease your way into ASSUME we provided some examples and tutorials. The former are helpful if you would like to get an impression of how ASSUME works and the latter introduce you into the development of ASSUME.

### Usage

There are two ways to run simulations using sparkle

- Using the provided Docker setup:

If you have installed Docker and set up the Docker Compose file previously, you can select 'timescale' in `examples.py` before running the simulation. This will save the simulation results in a Timescale database, and you can access the Dashboard at `http://localhost:3000`.

- Using the CLI to run simulations:

```bash
assume -s example_01b -db "postgresql://assume:assume@localhost:5432/assume"
```

For additional CLI options, run `assume -h`.

## License

Copyright 2022-2024 [ASSUME developers](https://assume.readthedocs.io/en/latest/developers.html).

ASSUME is licensed under the [GNU Affero General Public License v3.0](./LICENSES/AGPL-3.0-or-later.txt). This license is a strong copyleft license that requires that any derivative work be licensed under the same terms as the original work. It is approved by the [Open Source Initiative](https://opensource.org/licenses/AGPL-3.0).
