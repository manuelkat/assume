# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging

import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import (
    SolverFactory,
    SolverStatus,
    TerminationCondition,
    check_available_solvers,
)

from assume.common.base import SupportsMinMax
from assume.common.market_objects import MarketConfig, Orderbook
from assume.common.utils import get_products_index
from assume.units.dsm_load_shift import DSMFlex
from assume.units.dst_components import (
    create_boiler,
    create_ev,
    create_generic_storage,
    create_heatpump,
    create_pv_plant,
)

SOLVERS = ["gurobi", "glpk", "cbc", "cplex"]

logger = logging.getLogger(__name__)

# Mapping of component type identifiers to their respective classes
building_components = {
    "heatpump": create_heatpump,
    "boiler": create_boiler,
    "thermal_storage": create_generic_storage,
    "ev": create_ev,
    "battery_storage": create_generic_storage,
    "pv_plant": create_pv_plant,
}


class Building(SupportsMinMax, DSMFlex):
    """
    The Building class represents a building unit in the energy system.

    This class models the energy consumption, production, and storage capabilities of a building, allowing for
    optimization of energy flows. The building can include various components such as heat pumps, electric boilers,
    thermal storage, battery storage, photovoltaic (PV) systems, and electric vehicles (EVs). The optimization
    objective can be configured to minimize expenses, maximize revenue, or achieve other custom goals.

    Supported Components:
    - Heat Pump: Uses electricity to provide heating with a coefficient of performance (COP).
    - Electric Boiler: Provides heating either using electricity or natural gas.
    - Thermal Storage: Stores thermal energy to provide flexibility in heating demand.
    - Battery Storage: Stores electrical energy for later use, supporting both charging and discharging operations.
    - Photovoltaic (PV) System: Generates electricity from solar energy, which can be used on-site or sold to the grid.
    - Electric Vehicle (EV) Charging: Represents the charging of electric vehicles, contributing to electricity demand.

    Args:
        id (str): The unique identifier of the unit.
        unit_operator (str): The operator of the unit.
        technology (str): The technology of the unit, typically "building".
        bidding_strategies (dict): The bidding strategies of the unit.
        index (pd.DatetimeIndex): The index for the time series data of the unit.
        components (dict[str, dict]): The components of the unit such as heat pump, electric boiler, thermal storage, etc.
        node (str, optional): The node of the unit in the energy network. Defaults to "node0".
        location (tuple[float, float], optional): The geographical coordinates (latitude, longitude) of the unit. Defaults to (0.0, 0.0).
        flexibility_measure (str, optional): The flexibility measure used for the unit, e.g., maximum load shift. Defaults to "max_load_shift".
        objective (str, optional): The objective of the unit, e.g., minimize expenses ("minimize_expenses"). Defaults to "minimize_expenses".
    """

    def __init__(
        self,
        id: str,
        unit_operator: str,
        bidding_strategies: dict,
        index: pd.DatetimeIndex,
        components: dict[str, dict],
        node: str = "node0",
        location: tuple[float, float] = (0.0, 0.0),
        flexibility_measure: str = "max_load_shift",
        objective: str = "minimize_expenses",
        **kwargs,
    ):
        super().__init__(
            id=id,
            unit_operator=unit_operator,
            technology="building",
            bidding_strategies=bidding_strategies,
            index=index,
            node=node,
            location=location,
            **kwargs,
        )

        # Retrieve prices
        forecaster = self.forecaster
        self.electricity_price_buy = forecaster["price_electricity_buy"]
        self.electricity_price_sell = forecaster["price_electricity_sell"]
        self.natural_gas_price = forecaster["fuel_price_natural_gas"]

        # Retrieve load profiles
        self.electricity_demand = forecaster[f"{self.id}_load_profile"]
        self.heat_demand = forecaster[f"{self.id}_heat_demand"]
        self.ev_charging_profile = forecaster[f"{self.id}_ev_load_profile"]
        self.ev_availability_profile = forecaster[f"{self.id}_ev_availability"]
        self.pv_power_profile = forecaster[f"{self.id}_pv_power_profile"]
        self.pv_availability_profile = forecaster[f"{self.id}_pv_availability_profile"]

        self.flexibility_measure = flexibility_measure
        self.objective = objective

        self.components = self.check_components(components)

        # Main Model part
        self.model = pyo.ConcreteModel()

        self.define_sets()
        self.define_parameters()
        self.define_variables()

        self.initialize_components(components)
        self.define_constraints()
        self.define_objective()

        solvers = check_available_solvers(*SOLVERS)
        if len(solvers) < 1:
            raise Exception(f"None of {SOLVERS} are available")
        self.solver = SolverFactory(solvers[0])

        self.opt_power_requirement = None
        self.variable_cost_series = None

    def check_components(self, components: dict[str, dict]):
        """
        Checks and sets the building components based on the provided configuration.

        Args:
            components (dict[str, dict]): The components configuration for the building.
        """
        # Component existence checks
        self.has_heatpump = "heatpump" in components
        self.has_boiler = "boiler" in components
        self.has_thermal_storage = "thermal_storage" in components
        self.has_ev = "ev" in components
        self.has_battery_storage = "battery_storage" in components
        self.has_pv = "pv_plant" in components

        # Check fuel type of boiler
        self.is_boiler_electric = (
            self.has_boiler
            and components.get("boiler", {}).get("fuel_type") == "electric"
        )

        # Availability checks and profile assignments
        if self.has_ev:
            if self.ev_availability_profile is None:
                raise KeyError(
                    f"Missing 'availability' of the EV for building {self.id}."
                )
            components["ev"]["availability_profile"] = self.ev_availability_profile
            if self.ev_charging_profile is not None:
                components["ev"]["charging_profile"] = self.ev_charging_profile

        if self.has_pv:
            if self.pv_availability_profile is None or self.pv_power_profile is None:
                raise KeyError(
                    f"Missing 'availability' or 'power profile' of the PV for building {self.id}."
                )
            if (
                self.pv_power_profile is not None
                and self.pv_availability_profile is not None
            ):
                raise ValueError(
                    f"Both 'availability' and 'power profile' of the PV are provided for building {self.id}. Only one is allowed."
                )
            if self.pv_power_profile is not None:
                components["pv_plant"]["power_profile"] = self.pv_power_profile
            if self.pv_availability_profile is not None:
                components["pv_plant"]["availability_profile"] = (
                    self.pv_availability_profile
                )

        return components

    def initialize_components(self):
        """
        Initializes the components of the building.

        """
        self.model.dsm_blocks = pyo.Block(list(self.components.keys()))
        for technology, component_data in self.components.items():
            if technology in building_components:
                factory_method = building_components[technology]
                self.model.dsm_blocks[technology].transfer_attributes_from(
                    factory_method(
                        self.model, time_steps=self.model.time_steps, **component_data
                    )
                )

    def define_sets(self) -> None:
        """
        Defines the sets for the Pyomo model.
        """
        self.model.time_steps = pyo.Set(
            initialize=[idx for idx, _ in enumerate(self.index)]
        )

    def define_parameters(self):
        """
        Defines the parameters for the Pyomo model.
        """
        self.model.electricity_price = pyo.Param(
            self.model.time_steps,
            initialize={t: value for t, value in enumerate(self.electricity_price)},
        )
        self.model.natural_gas_price = pyo.Param(
            self.model.time_steps,
            initialize={t: value for t, value in enumerate(self.natural_gas_price)},
        )
        self.model.heat_demand = pyo.Param(
            self.model.time_steps,
            initialize={t: value for t, value in enumerate(self.heat_demand)},
        )
        self.model.electricity_demand = pyo.Param(
            self.model.time_steps,
            initialize={t: value for t, value in enumerate(self.electricity_demand)},
        )

    def define_variables(self):
        """
        Defines the variables for the Pyomo model.
        """
        self.model.P_from_grid = pyo.Var(
            self.model.time_steps, within=pyo.NonNegativeReals
        )
        self.model.P_to_grid = pyo.Var(
            self.model.time_steps, within=pyo.NonNegativeReals
        )

        self.model.cost = pyo.Var(self.model.time_steps, within=pyo.NonNegativeReals)
        self.model.revenue = pyo.Var(self.model.time_steps, within=pyo.NonNegativeReals)

    def define_constraints(self):
        @self.model.Constraint(self.model.time_steps)
        def energy_balance_constraint(m, t):
            """
            Ensures the total energy balance, considering all electric power flows in and out of the system.
            """
            # Energy produced from different sources
            pv_power = m.pv_power[t] if self.has_pv else 0
            battery_discharge = (
                m.dsm_blocks["battery_storage"].discharge[t]
                if self.has_battery_storage
                else 0
            )
            p_from_grid = m.P_from_grid[t]

            # Energy demand components
            electricity_demand = m.electricity_demand[t]
            heatpump_demand = (
                m.dsm_blocks["heatpump"].power_in[t] if self.has_heatpump else 0
            )
            boiler_demand = (
                m.dsm_blocks["boiler"].power_in[t]
                if self.has_boiler and self.is_boiler_electric
                else 0
            )
            battery_charge = (
                m.dsm_blocks["battery_storage"].charge[t]
                if self.has_battery_storage
                else 0
            )
            ev_demand = m.dsm_blocks["ev"].charge[t] if self.has_ev else 0
            p_to_grid = m.P_to_grid[t]

            # Energy balance equation
            return (pv_power + battery_discharge + p_from_grid) == (
                electricity_demand
                + heatpump_demand
                + boiler_demand
                + battery_charge
                + ev_demand
                + p_to_grid
            )

        @self.model.Constraint(self.model.time_steps)
        def variable_revenue_constraint(m, t):
            """
            Calculates the variable revenue per time step from selling electricity to the grid.
            """
            return m.revenue[t] == m.P_to_grid[t] * self.electricity_price_sell[t]

        @self.model.Constraint(self.model.time_steps)
        def variable_cost_constraint(m, t):
            """
            Calculates the variable cost per time step from buying electricity from the grid.
            """
            return m.cost[t] == m.P_from_grid[t] * self.electricity_price_buy[t]

        if self.has_heatpump or self.has_boiler or self.has_thermal_storage:

            @self.model.Constraint(self.model.time_steps)
            def heat_flow_constraint(m, t):
                """
                Ensures the heat flow from the heat pump, electric boiler, and thermal storage
                satisfies the heat demand.
                """
                heatpump_output = (
                    m.dsm_blocks["heatpump"].heat_out[t] if self.has_heatpump else 0
                )
                boiler_output = (
                    m.dsm_blocks["boiler"].heat_out[t] if self.has_boiler else 0
                )
                thermal_storage_charge = (
                    m.dsm_blocks["thermal_storage"].charge[t]
                    if self.has_thermal_storage
                    else 0
                )
                thermal_storage_discharge = (
                    m.dsm_blocks["thermal_storage"].discharge[t]
                    if self.has_thermal_storage
                    else 0
                )

                return (
                    heatpump_output + boiler_output + thermal_storage_charge
                    == m.heat_demand[t] + thermal_storage_discharge
                )

    def define_objective(self):
        """
        Defines the objective for the optimization model.
        """

        @self.model.Objective(sense=pyo.minimize)
        def obj_rule(m):
            """
            Minimizes the net expenses over all time steps.
            """
            net_expenses = sum(m.cost[t] - m.revenue[t] for t in self.model.time_steps)
            return net_expenses

    def calculate_optimal_operation_if_needed(self):
        """
        Checks if an optimal operation needs to be calculated based on the power requirements and the defined objective.
        If the power requirement has not been calculated and the objective is "minimize_expenses", it proceeds to calculate
        the optimal operation.
        Raises an error if the objective is not supported.
        """
        if self.opt_power_requirement is None and self.objective == "minimize_expenses":
            self.calculate_optimal_operation()
        elif self.objective != "minimize_expenses":
            raise NotImplementedError(
                f"Objective {self.objective} is not implemented for building {self.id}."
            )

    def calculate_optimal_operation(self):
        """
        Determines the optimal operation of the building based on the defined model and objective.
        Solves the model instance, processes the results, and updates power requirements and cost series.
        """
        # create an instance of the model
        instance = self.model.create_instance()
        # solve the instance
        results = self.solver.solve(instance, tee=False)

        # Check solver status and termination condition
        if (results.solver.status == SolverStatus.ok) and (
            results.solver.termination_condition == TerminationCondition.optimal
        ):
            logger.debug("The model was solved optimally.")

            # Display the Objective Function Value
            objective_value = pyo.value(instance.obj_rule)
            logger.debug(f"The value of the objective function is {objective_value}.")
        elif results.solver.termination_condition == TerminationCondition.infeasible:
            logger.debug("The model is infeasible.")
        else:
            logger.debug(f"Solver Status: {results.solver.status}")
            logger.debug(
                f"Termination Condition: {results.solver.termination_condition}"
            )

        # Extract power input, output, and calculate the needed power
        power_input = instance.P_from_grid.get_values()
        power_output = instance.P_to_grid.get_values()
        needed_power = {
            key: power_input[key] - power_output.get(key, 0)
            for key in power_input.keys()
        }
        self.opt_power_requirement = pd.Series(data=needed_power, index=self.index)

        # Variable cost and revenue series
        variable_costs = instance.cost.get_values()
        variable_revenues = instance.revenue.get_values()
        net_costs = {
            key: variable_costs[key] - variable_revenues.get(key, 0)
            for key in variable_costs.keys()
        }
        self.variable_cost_series = pd.Series(data=net_costs, index=self.index)

        self.write_additional_outputs(instance)

    def calculate_marginal_cost(self, start: pd.Timestamp, power: float) -> float:
        """
        Calculate the marginal cost of the unit based on the provided time and power.

        Args:
            start (pandas.Timestamp): The start time of the dispatch.
            power (float): The power output of the unit.

        Returns:
            float: the marginal cost of the unit for the given power.
        """
        # Initialize marginal cost
        marginal_cost = 0

        if self.opt_power_requirement[start] > 0:
            marginal_cost = (
                self.variable_cost_series[start] / self.opt_power_requirement[start]
            )
        return marginal_cost

    def set_dispatch_plan(
        self,
        marketconfig: MarketConfig,
        orderbook: Orderbook,
    ) -> None:
        """
        Adds the dispatch plan from the current market result to the total dispatch plan and calculates the cashflow.

        Args:
            marketconfig (MarketConfig): The market configuration.
            orderbook (Orderbook): The orderbook.
        """
        products_index = get_products_index(orderbook)

        for order in orderbook:
            start = order["start_time"]
            end = order["end_time"]
            end_excl = end - self.index.freq
            if isinstance(order["accepted_volume"], dict):
                self.outputs["energy"].loc[start:end_excl] += [
                    order["accepted_volume"][key]
                    for key in order["accepted_volume"].keys()
                ]
            else:
                self.outputs["energy"].loc[start:end_excl] += order["accepted_volume"]

        # TODO: Need to check with Nick how we formulate the bidding strategy and market (Energy Provider)

        self.calculate_cashflow("energy", orderbook)

        for start in products_index:
            # TODO: For what is this needed??
            current_power = self.outputs["energy"][start]
            self.outputs["energy"][start] = current_power

        self.bidding_strategies[marketconfig.market_id].calculate_reward(
            unit=self,
            marketconfig=marketconfig,
            orderbook=orderbook,
        )

    def write_additional_outputs(self, instance):
        if self.has_battery_storage:
            soc = pd.Series(
                data=instance.dsm_blocks["battery_storage"].soc.get_values(),
                dtype=float,
            )
            soc.index = self.index
            self.outputs["soc"] = soc
        if self.has_ev:
            ev_soc = pd.Series(
                data=instance.dsm_blocks["ev"].ev_battery_soc.get_values(),
                index=self.index,
                dtype=object,
            )
            ev_soc.index = self.index
            self.outputs["ev_soc"] = ev_soc

    def as_dict(self) -> dict:
        """
        Returns the attributes of the unit as a dictionary, including specific attributes.

        Returns:
            dict: The attributes of the unit as a dictionary.
        """
        components_list = [component for component in self.model.dsm_blocks.keys()]
        components_string = ",".join(components_list)

        unit_dict = super().as_dict()
        unit_dict.update(
            {
                "unit_type": "demand",
                "components": components_string,
            }
        )

        return unit_dict
