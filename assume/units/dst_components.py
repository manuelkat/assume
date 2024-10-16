# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pandas as pd
import pyomo.environ as pyo


def create_heatpump(
    model: pyo.ConcreteModel,
    time_steps: list[int],
    rated_power: float,
    cop: float,
    min_power: float = 0.0,
    ramp_up: float | None = None,
    ramp_down: float | None = None,
    min_operating_steps: int = 0,
    min_down_steps: int = 0,
    **kwargs,
) -> pyo.Block:
    """
    Creates a generic heat pump unit within an energy system model.

    Args:
        model (ConcreteModel): A Pyomo ConcreteModel object representing the optimization model.
        time_steps (List[int]): Time steps over which the model operates.
        rated_power (float): Maximum power input to the heat pump.
        cop (float): Coefficient of performance of the heat pump.
        min_power (float, optional): Minimum power input to the heat pump. Defaults to 0.0.
        ramp_up (Optional[float], optional): Maximum allowed increase in power per time step.
            If None, allows full ramp-up to rated_power. Defaults to None.
        ramp_down (Optional[float], optional): Maximum allowed decrease in power per time step.
            If None, allows full ramp-down to rated_power. Defaults to None.
        min_operating_steps (int, optional): Minimum number of consecutive time steps the heat pump must operate.
            Defaults to 0.
        min_down_steps (int, optional): Minimum number of consecutive time steps the heat pump must remain off.
            Defaults to 0.
        **kwargs: Additional keyword arguments for custom behavior or constraints.

    Returns:
        model_part (Block): A Pyomo block representing the heat pump with variables and constraints.
    """

    # Set default ramp limits to rated_power if not provided
    ramp_up = rated_power if ramp_up is None else ramp_up
    ramp_down = rated_power if ramp_down is None else ramp_down

    # Define block
    model_part = pyo.Block()

    # Define parameters
    model_part.rated_power = pyo.Param(initialize=rated_power)
    model_part.min_power = pyo.Param(initialize=min_power)
    model_part.cop = pyo.Param(initialize=cop)
    model_part.ramp_up = pyo.Param(initialize=ramp_up)
    model_part.ramp_down = pyo.Param(initialize=ramp_down)
    model_part.min_operating_steps = pyo.Param(initialize=min_operating_steps)
    model_part.min_down_steps = pyo.Param(initialize=min_down_steps)

    # Define variables
    model_part.power_in = pyo.Var(
        time_steps, within=pyo.NonNegativeReals, bounds=(0, rated_power)
    )
    model_part.heat_out = pyo.Var(time_steps, within=pyo.NonNegativeReals)
    model_part.operating_cost_hp = pyo.Var(time_steps, within=pyo.NonNegativeReals)

    # Define operational status variable if min operating time, downtime, or min_power is required
    if min_operating_steps > 0 or min_down_steps > 0 or min_power > 0:
        model_part.operational_status = pyo.Var(time_steps, within=pyo.Binary)

        # Link power_in and operational_status
        @model_part.Constraint(time_steps)
        def min_power_constraint(b, t):
            return b.power_in[t] >= b.min_power * b.operational_status[t]

        @model_part.Constraint(time_steps)
        def max_power_constraint(b, t):
            return b.power_in[t] <= b.rated_power * b.operational_status[t]

    # Coefficient of performance (COP) constraint
    @model_part.Constraint(time_steps)
    def cop_constraint(b, t):
        return b.heat_out[t] == b.power_in[t] * b.cop

    # Ramp-up constraint
    @model_part.Constraint(time_steps)
    def ramp_up_constraint(b, t):
        if t == time_steps[0]:
            return pyo.Constraint.Skip
        return b.power_in[t] - b.power_in[t - 1] <= b.ramp_up

    # Ramp-down constraint
    @model_part.Constraint(time_steps)
    def ramp_down_constraint(b, t):
        if t == time_steps[0]:
            return pyo.Constraint.Skip
        return b.power_in[t - 1] - b.power_in[t] <= b.ramp_down

    # Minimum operating time constraint
    if min_operating_steps > 0:

        @model_part.Constraint(time_steps)
        def min_operating_time_constraint(b, t):
            if t < min_operating_steps - 1:
                return pyo.Constraint.Skip

            relevant_time_steps = range(t - min_operating_steps + 1, t + 1)
            return (
                sum(b.operational_status[i] for i in relevant_time_steps)
                >= min_operating_steps * b.operational_status[t]
            )

    # Minimum downtime constraint
    if min_down_steps > 0:

        @model_part.Constraint(time_steps)
        def min_downtime_constraint(b, t):
            if t < min_down_steps - 1:
                return pyo.Constraint.Skip

            relevant_time_steps = range(t - min_down_steps + 1, t + 1)
            return sum(
                1 - b.operational_status[i] for i in relevant_time_steps
            ) >= min_down_steps * (1 - b.operational_status[t])

    return model_part


def create_boiler(
    model: pyo.ConcreteModel,
    time_steps: list[int],
    rated_power: float,
    efficiency: float = 1.0,
    min_power: float = 0.0,
    ramp_up: float | None = None,
    ramp_down: float | None = None,
    min_operating_steps: int = 0,
    min_down_steps: int = 0,
    fuel_type: str = "electric",  # 'electric' or 'natural_gas'
    **kwargs,
) -> pyo.Block:
    """
    Creates a generic boiler unit within an energy system model.

    Args:
        model (ConcreteModel): A Pyomo ConcreteModel object representing the optimization model.
        time_steps (List[int]): Time steps over which the model operates.
        rated_power (float): Maximum power input to the boiler.
        efficiency (float, optional): Efficiency of the boiler. Defaults to 1.0.
        min_power (float, optional): Minimum power input to the boiler. Defaults to 0.0.
        ramp_up (Optional[float], optional): Maximum allowed increase in power per time step.
            If None, allows full ramp-up to rated_power. Defaults to None.
        ramp_down (Optional[float], optional): Maximum allowed decrease in power per time step.
            If None, allows full ramp-down to rated_power. Defaults to None.
        min_operating_steps (int, optional): Minimum number of consecutive time steps the boiler must operate.
            Defaults to 0.
        min_down_steps (int, optional): Minimum number of consecutive time steps the boiler must remain off.
            Defaults to 0.
        fuel_type (str, optional): Type of fuel used by the boiler ('electric' or 'natural_gas'). Defaults to 'electric'.
        **kwargs: Additional keyword arguments for custom behavior or constraints.

    Returns:
        model_part (Block): A Pyomo block representing the boiler with variables and constraints.
    """

    # Set default ramp limits to rated_power if not provided
    ramp_up = rated_power if ramp_up is None else ramp_up
    ramp_down = rated_power if ramp_down is None else ramp_down

    # Define block
    model_part = pyo.Block()

    # Define parameters
    model_part.rated_power = pyo.Param(initialize=rated_power)
    model_part.min_power = pyo.Param(initialize=min_power)
    model_part.efficiency = pyo.Param(initialize=efficiency)
    model_part.ramp_up = pyo.Param(initialize=ramp_up)
    model_part.ramp_down = pyo.Param(initialize=ramp_down)

    # Define variables
    model_part.power_in = pyo.Var(
        time_steps, within=pyo.NonNegativeReals, bounds=(0, rated_power)
    )
    if fuel_type == "natural_gas":
        model_part.natural_gas_in = pyo.Var(time_steps, within=pyo.NonNegativeReals)
    model_part.heat_out = pyo.Var(time_steps, within=pyo.NonNegativeReals)

    # Define operational status variable if min operating time, downtime, or min_power is required
    if min_operating_steps > 0 or min_down_steps > 0 or min_power > 0:
        model_part.operational_status = pyo.Var(time_steps, within=pyo.Binary)

        # Link power_in and operational_status
        @model_part.Constraint(time_steps)
        def min_power_constraint(b, t):
            return b.power_in[t] >= b.min_power * b.operational_status[t]

        @model_part.Constraint(time_steps)
        def max_power_constraint(b, t):
            return b.power_in[t] <= b.rated_power * b.operational_status[t]

    # Efficiency constraint based on fuel type
    @model_part.Constraint(time_steps)
    def efficiency_constraint(b, t):
        if fuel_type == "electric":
            return b.heat_out[t] == b.power_in[t] * b.efficiency
        elif fuel_type == "natural_gas":
            return b.heat_out[t] == b.natural_gas_in[t] * b.efficiency
        else:
            raise ValueError(
                "Unsupported fuel_type. Choose 'electric' or 'natural_gas'."
            )

    # Ramp-up constraint
    @model_part.Constraint(time_steps)
    def ramp_up_constraint(b, t):
        if t == time_steps[0]:
            return pyo.Constraint.Skip
        return b.power_in[t] - b.power_in[t - 1] <= b.ramp_up

    # Ramp-down constraint
    @model_part.Constraint(time_steps)
    def ramp_down_constraint(b, t):
        if t == time_steps[0]:
            return pyo.Constraint.Skip
        return b.power_in[t - 1] - b.power_in[t] <= b.ramp_down

    # Minimum operating time constraint
    if min_operating_steps > 0:

        @model_part.Constraint(time_steps)
        def min_operating_time_constraint(b, t):
            if t < min_operating_steps - 1:
                return pyo.Constraint.Skip

            relevant_time_steps = range(t - min_operating_steps + 1, t + 1)
            return (
                sum(b.operational_status[i] for i in relevant_time_steps)
                >= min_operating_steps * b.operational_status[t]
            )

    # Minimum downtime constraint
    if min_down_steps > 0:

        @model_part.Constraint(time_steps)
        def min_downtime_constraint(b, t):
            if t < min_down_steps - 1:
                return pyo.Constraint.Skip

            relevant_time_steps = range(t - min_down_steps + 1, t + 1)
            return sum(
                1 - b.operational_status[i] for i in relevant_time_steps
            ) >= min_down_steps * (1 - b.operational_status[t])

    return model_part


def create_ev(
    model: pyo.ConcreteModel,
    time_steps: list[int],
    max_capacity: float,
    min_capacity: float,
    max_power_charge: float,
    max_power_discharge: float | None = None,
    efficiency_charge: float = 1.0,
    efficiency_discharge: float = 1.0,
    initial_soc: float = 1.0,
    ramp_up: float | None = None,
    ramp_down: float | None = None,
    availability_profile: pd.Series | None = None,
    charging_profile: pd.Series | None = None,
    **kwargs,
) -> pyo.Block:
    """
    Creates an Electric Vehicle (EV) unit by extending a generic storage model with EV-specific constraints.

    Args:
        model (pyo.ConcreteModel): A Pyomo ConcreteModel object representing the optimization model.
        time_steps (list[int]): Time steps in the optimization model.
        max_capacity (float): The maximum capacity of the EV battery (MWh).
        min_capacity (float): The minimum capacity of the EV battery (MWh).
        max_power_charge (float): The maximum charging power of the EV (MW).
        max_power_discharge (float | None, optional): The maximum discharging power of the EV (MW).
            If None, defaults to `max_power_charge`. Defaults to None.
        efficiency_charge (float, optional): Charging efficiency of the EV. Defaults to 1.0.
        efficiency_discharge (float, optional): Discharging efficiency of the EV. Defaults to 1.0.
        initial_soc (float, optional): Initial state of charge (SOC) as a fraction of `max_capacity`. Defaults to 1.0.
        ramp_up (float | None, optional): Maximum increase in charging power per time step (MW).
            If None, no ramp-up constraint is applied. Defaults to None.
        ramp_down (float | None, optional): Maximum decrease in charging power per time step (MW).
            If None, no ramp-down constraint is applied. Defaults to None.
        availability_profile (pd.Series | None, optional): Series indicating EV availability with time_steps as indices
            and binary values (1 available, 0 unavailable). Defaults to None.
        charging_profile (pd.Series | None, optional): Series indicating predefined charging profile.
            If given, the EV will follow this profile instead of optimizing charging. Defaults to None.
        **kwargs: Additional keyword arguments for custom behavior or constraints.

    Returns:
        pyo.Block: A Pyomo block representing the EV unit with variables and constraints.

    Constraints:
        - Ensures SOC stays between `min_capacity` and `max_capacity`.
        - Limits charging and discharging power to specified maxima.
        - Tracks SOC changes over time based on charging, discharging, and storage loss.
        - Limits ramp-up and ramp-down rates if specified.
        - Enforces charging profiles if specified.
        - Respects EV availability periods if specified.
    """

    # Set default power limits to max_power_charge if not provided
    max_power_discharge = (
        max_power_charge if max_power_discharge is None else max_power_discharge
    )

    # Initialize the generic storage with EV-specific parameters
    model_part = create_generic_storage(
        model=model,
        time_steps=time_steps,
        max_capacity=max_capacity,
        min_capacity=min_capacity,
        max_power_charge=max_power_charge,
        max_power_discharge=max_power_discharge,
        efficiency_charge=efficiency_charge,
        efficiency_discharge=efficiency_discharge,
        initial_soc=initial_soc,
        ramp_up=ramp_up,
        ramp_down=ramp_down,
        **kwargs,
    )

    # Apply availability profile constraints if provided
    if availability_profile is not None:
        # Ensure that availability_profile is indexed by time_steps
        if not isinstance(availability_profile, pd.Series):
            raise TypeError("`availability_profile` must be a pandas Series.")
        if not all(t in availability_profile.index for t in time_steps):
            raise ValueError(
                "All `time_steps` must be present in `availability_profile` index."
            )

        @model_part.Constraint(time_steps)
        def discharge_availability_constraint(b, t):
            """
            Ensures the EV is only discharged during available periods.
            """
            availability = availability_profile[t]
            return b.discharge[t] <= availability * b.max_power_discharge

        @model_part.Constraint(time_steps)
        def charge_availability_constraint(b, t):
            """
            Ensures the EV is only charged during available periods.
            """
            availability = availability_profile[t]
            return b.charge[t] <= availability * b.max_power_charge

    # Apply predefined charging profile constraints if provided
    if charging_profile is not None:
        # Ensure that charging_profile is indexed by time_steps
        if not isinstance(charging_profile, pd.Series):
            raise TypeError("`charging_profile` must be a pandas Series.")
        if not all(t in charging_profile.index for t in time_steps):
            raise ValueError(
                "All `time_steps` must be present in `charging_profile` index."
            )

        @model_part.Constraint(time_steps)
        def charging_profile_constraint(b, t):
            """
            Enforces the predefined charging profile.
            """
            return b.charge[t] == charging_profile[t]

    return model_part


def create_generic_storage(
    model: pyo.ConcreteModel,
    time_steps: list[int],
    max_capacity: float,
    min_capacity: float = 0.0,
    max_power_charge: float | None = None,
    max_power_discharge: float | None = None,
    efficiency_charge: float = 1.0,
    efficiency_discharge: float = 1.0,
    initial_soc: float = 1.0,
    ramp_up: float | None = None,
    ramp_down: float | None = None,
    storage_loss_rate: float = 0.0,
    **kwargs,
) -> pyo.Block:
    """
    Creates a generic storage unit (e.g., battery) within an energy system model.

    Args:
        model (pyo.ConcreteModel): A Pyomo ConcreteModel object representing the optimization model.
        time_steps (list[int]): Time steps over which the model operates.
        max_capacity (float): Maximum energy storage capacity of the storage unit.
        min_capacity (float, optional): Minimum allowable state of charge (SOC). Defaults to 0.0.
        max_power_charge (float | None, optional): Maximum charging power of the storage unit.
            If None, defaults to max_capacity. Defaults to None.
        max_power_discharge (float | None, optional): Maximum discharging power of the storage unit.
            If None, defaults to max_capacity. Defaults to None.
        efficiency_charge (float, optional): Efficiency of the charging process. Defaults to 1.0.
        efficiency_discharge (float, optional): Efficiency of the discharging process. Defaults to 1.0.
        initial_soc (float, optional): Initial state of charge as a fraction of max_capacity.
            Defaults to 1.0.
        ramp_up (float | None, optional): Maximum increase in charging/discharging power per time step.
            If None, no ramp-up constraint is applied. Defaults to None.
        ramp_down (float | None, optional): Maximum decrease in charging/discharging power per time step.
            If None, no ramp-down constraint is applied. Defaults to None.
        storage_loss_rate (float, optional): Fraction of energy lost per time step due to storage inefficiencies.
            Defaults to 0.0.
        **kwargs: Additional keyword arguments for custom behavior or constraints.

    Returns:
        pyo.Block: A Pyomo block representing the storage system with variables and constraints.

    Constraints:
        - Ensures SOC stays between min_capacity and max_capacity.
        - Limits charging and discharging power to specified maxima.
        - Tracks SOC changes over time based on charging, discharging, and storage loss.
    """

    # Set default power limits to max_capacity if not provided
    max_power_charge = max_capacity if max_power_charge is None else max_power_charge
    max_power_discharge = (
        max_capacity if max_power_discharge is None else max_power_discharge
    )

    # Define block
    model_part = pyo.Block()

    # Define parameters
    model_part.max_capacity = pyo.Param(initialize=max_capacity)
    model_part.min_capacity = pyo.Param(initialize=min_capacity)
    model_part.max_power_charge = pyo.Param(initialize=max_power_charge)
    model_part.max_power_discharge = pyo.Param(initialize=max_power_discharge)
    model_part.efficiency_charge = pyo.Param(initialize=efficiency_charge)
    model_part.efficiency_discharge = pyo.Param(initialize=efficiency_discharge)
    model_part.initial_soc = pyo.Param(initialize=initial_soc * max_capacity)
    model_part.ramp_up = pyo.Param(initialize=ramp_up)
    model_part.ramp_down = pyo.Param(initialize=ramp_down)
    model_part.storage_loss_rate = pyo.Param(initialize=storage_loss_rate)

    # Define variables
    model_part.soc = pyo.Var(
        time_steps,
        within=pyo.NonNegativeReals,
        bounds=(min_capacity, max_capacity),
        doc="State of Charge at each time step",
    )
    model_part.charge = pyo.Var(
        time_steps,
        within=pyo.NonNegativeReals,
        bounds=(0, max_power_charge),
        doc="Charging power at each time step",
    )
    model_part.discharge = pyo.Var(
        time_steps,
        within=pyo.NonNegativeReals,
        bounds=(0, max_power_discharge),
        doc="Discharging power at each time step",
    )

    # Define SOC dynamics with energy loss and efficiency
    @model_part.Constraint(time_steps)
    def soc_balance_rule(b, t):
        if t == time_steps[0]:
            prev_soc = b.initial_soc
        else:
            prev_soc = b.soc[t - 1]
        return b.soc[t] == (
            prev_soc
            + b.efficiency_charge * b.charge[t]
            - (1 / b.efficiency_discharge) * b.discharge[t]
            - b.storage_loss_rate * prev_soc
        )

    # Apply ramp-up constraints if ramp_up is specified
    if ramp_up is not None:

        @model_part.Constraint(time_steps)
        def charge_ramp_up_constraint(b, t):
            """
            Limits the ramp-up rate of EV charging.
            """
            if t == time_steps[0]:
                return pyo.Constraint.Skip
            return b.charge[t] - b.charge[t - 1] <= ramp_up

        @model_part.Constraint(time_steps)
        def discharge_ramp_up_constraint(b, t):
            """
            Limits the ramp-up rate of EV discharging.
            """
            if t == time_steps[0]:
                return pyo.Constraint.Skip
            return b.discharge[t] - b.discharge[t - 1] <= ramp_up

    # Apply ramp-down constraints if ramp_down is specified
    if ramp_down is not None:

        @model_part.Constraint(time_steps)
        def charge_ramp_down_constraint(b, t):
            """
            Limits the ramp-down rate of EV charging.
            """
            if t == time_steps[0]:
                return pyo.Constraint.Skip
            return b.charge[t - 1] - b.charge[t] <= ramp_down

        @model_part.Constraint(time_steps)
        def discharge_ramp_down_constraint(b, t):
            """
            Limits the ramp-down rate of EV discharging.
            """
            if t == time_steps[0]:
                return pyo.Constraint.Skip
            return b.discharge[t - 1] - b.discharge[t] <= ramp_down

    return model_part


def create_pv_plant(
    model: pyo.ConcreteModel,
    time_steps: list[int],
    capacity: float,
    availability_profile: pd.Series | None = None,
    power_profile: pd.Series | None = None,
    **kwargs,
) -> pyo.Block:
    """
    Creates a Photovoltaic (PV) power plant unit within an energy system model.

    Args:
        model (pyo.ConcreteModel): A Pyomo ConcreteModel object representing the optimization model.
        time_steps (list[int]): Time steps over which the model operates.
        capacity (float): The maximum power output capacity of the PV unit (MW).
        availability_profile (pd.Series | None, optional): Series indicating PV availability with time_steps as indices
            and binary values (1 available, 0 unavailable). Defaults to None.
        power_profile (pd.Series | None, optional): Series indicating a predefined power output profile.
            If provided, the PV unit will follow this profile instead of optimizing power output. Defaults to None.
        **kwargs: Additional keyword arguments for custom behavior or constraints.

    Returns:
        pyo.Block: A Pyomo block representing the PV plant with variables and constraints.

    Constraints:
        - power_profile_constraint: Ensures the PV follows a predefined power profile if provided.
        - availability_pv_constraint: Ensures the PV operates only during available periods.
        - max_power_pv_constraint: Ensures the power output of the PV unit does not exceed the maximum power limit.
    """

    # Set capacity as the maximum power output
    max_power = capacity

    # Raise error if both availability_profile and power_profile are provided
    # or if neither is provided
    if availability_profile is not None and power_profile is not None:
        raise ValueError(
            "Provide either `availability_profile` or `power_profile` for the PV plant, not both."
        )
    elif availability_profile is None and power_profile is None:
        raise ValueError(
            "Provide `availability_profile` or `power_profile` for the PV plant."
        )

    # Validate availability_profile if provided
    if availability_profile is not None:
        if not isinstance(availability_profile, pd.Series):
            raise TypeError("`availability_profile` must be a pandas Series.")
        if not all(t in availability_profile.index for t in time_steps):
            raise ValueError(
                "All `time_steps` must be present in `availability_profile` index."
            )

    # Validate power_profile if provided
    if power_profile is not None:
        if not isinstance(power_profile, pd.Series):
            raise TypeError("`power_profile` must be a pandas Series.")
        if not all(t in power_profile.index for t in time_steps):
            raise ValueError(
                "All `time_steps` must be present in `power_profile` index."
            )

    # Define block
    model_part = pyo.Block()

    # Define parameters
    model_part.capacity = pyo.Param(initialize=capacity, within=pyo.NonNegativeReals)

    # Define variables
    model_part.power = pyo.Var(
        time_steps,
        within=pyo.NonNegativeReals,
        bounds=(0, max_power),
    )

    # Define constraints

    # Predefined power profile constraint
    if power_profile is not None:

        @model_part.Constraint(time_steps)
        def power_profile_constraint(b, t):
            """
            Ensures the PV follows the predefined power profile.
            """
            return b.power[t] == power_profile[t]

    # Availability profile constraints
    if availability_profile is not None:

        @model_part.Constraint(time_steps)
        def availability_pv_constraint(b, t):
            """
            Ensures the PV operates only during available periods.
            """
            availability = availability_profile[t]
            return b.power[t] <= availability * b.capacity

    # Maximum power constraint (redundant due to variable bounds, included for clarity)
    @model_part.Constraint(time_steps)
    def max_power_pv_constraint(b, t):
        """
        Ensures the power output of the PV unit does not exceed the maximum power limit.
        """
        return b.power[t] <= b.capacity

    return model_part


def create_electrolyser(
    model: pyo.ConcreteModel,
    time_steps: list[int],
    rated_power: float,
    efficiency: float,
    min_power: float = 0.0,
    ramp_up: float | None = None,
    ramp_down: float | None = None,
    min_operating_steps: int = 0,
    min_down_steps: int = 0,
    **kwargs,
) -> pyo.Block:
    """
    Represents an electrolyser unit used for hydrogen production through electrolysis.

    Args:
        model (pyomo.ConcreteModel): The Pyomo model where the electrolyser unit will be added.
        time_steps (list[int]): The list of time steps for the model.
        rated_power (float): The rated power capacity of the electrolyser (in MW).
        efficiency (float): The efficiency of the electrolysis process (0-1).
        min_power (float): The minimum power required for operation (in MW).
        ramp_up (float | None): The maximum rate at which the electrolyser can increase its power output (in MW/hr).
        ramp_down (float | None): The maximum rate at which the electrolyser can decrease its power output (in MW/hr).
        min_operating_steps (int): The minimum number of steps the electrolyser must operate continuously.
        min_down_steps (int): The minimum number of downtime steps required between operating cycles.

    Returns:
        pyo.Block: The electrolyser block.
    """
    # Set default ramp limits to rated_power if not provided
    ramp_up = rated_power if ramp_up is None else ramp_up
    ramp_down = rated_power if ramp_down is None else ramp_down

    # Define block
    model_part = pyo.Block()

    # Define parameters
    model_part.rated_power = pyo.Param(initialize=rated_power)
    model_part.efficiency = pyo.Param(initialize=efficiency)
    model_part.min_power = pyo.Param(initialize=min_power)
    model_part.ramp_up = pyo.Param(initialize=ramp_up)
    model_part.ramp_down = pyo.Param(initialize=ramp_down)
    model_part.min_operating_steps = pyo.Param(initialize=min_operating_steps)
    model_part.min_down_steps = pyo.Param(initialize=min_down_steps)

    # Define variables
    model_part.power_in = pyo.Var(
        time_steps, within=pyo.NonNegativeReals, bounds=(0, rated_power)
    )
    model_part.hydrogen_out = pyo.Var(time_steps, within=pyo.NonNegativeReals)
    model_part.electrolyser_operating_cost = pyo.Var(
        time_steps, within=pyo.NonNegativeReals
    )

    # Define operational status variable if needed
    if min_operating_steps > 0 or min_down_steps > 0 or min_power > 0:
        model_part.operational_status = pyo.Var(time_steps, within=pyo.Binary)

        # Link power_in and operational_status
        @model_part.Constraint(time_steps)
        def min_power_constraint(b, t):
            return b.power_in[t] >= b.min_power * b.operational_status[t]

        @model_part.Constraint(time_steps)
        def max_power_constraint(b, t):
            return b.power_in[t] <= b.rated_power * b.operational_status[t]

    # Efficiency constraint
    @model_part.Constraint(time_steps)
    def hydrogen_production_constraint(b, t):
        return b.power_in[t] == b.hydrogen_out[t] / b.efficiency

    # Ramp-up constraint
    @model_part.Constraint(time_steps)
    def ramp_up_constraint(b, t):
        if t == time_steps[0]:
            return pyo.Constraint.Skip
        return (
            b.power_in[t] - b.power_in[time_steps[time_steps.index(t) - 1]] <= b.ramp_up
        )

    # Ramp-down constraint
    @model_part.Constraint(time_steps)
    def ramp_down_constraint(b, t):
        if t == time_steps[0]:
            return pyo.Constraint.Skip
        return (
            b.power_in[time_steps[time_steps.index(t) - 1]] - b.power_in[t]
            <= b.ramp_down
        )

    # Minimum operating time constraint
    if min_operating_steps > 0:

        @model_part.Constraint(time_steps)
        def min_operating_time_constraint(b, t):
            if t < min_operating_steps - 1:
                return pyo.Constraint.Skip
            relevant_time_steps = range(t - min_operating_steps + 1, t + 1)
            return (
                sum(b.operational_status[i] for i in relevant_time_steps)
                >= min_operating_steps * b.operational_status[t]
            )

    # Minimum downtime constraint
    if min_down_steps > 0:

        @model_part.Constraint(time_steps)
        def min_downtime_constraint(b, t):
            if t < min_down_steps - 1:
                return pyo.Constraint.Skip
            relevant_time_steps = range(t - min_down_steps + 1, t + 1)
            return sum(
                1 - b.operational_status[i] for i in relevant_time_steps
            ) >= min_down_steps * (1 - b.operational_status[t])

    # Operating cost constraint
    @model_part.Constraint(time_steps)
    def operating_cost_with_el_price(b, t):
        return (
            b.electrolyser_operating_cost[t]
            == b.power_in[t] * model.electricity_price[t]
        )

    return model_part


def create_dri_plant(
    model: pyo.ConcreteModel,
    time_steps: list[int],
    specific_hydrogen_consumption: float,
    specific_natural_gas_consumption: float,
    specific_electricity_consumption: float,
    specific_iron_ore_consumption: float,
    rated_power: float,
    min_power: float,
    fuel_type: str,
    ramp_up: float | None = None,
    ramp_down: float | None = None,
    min_operating_steps: int = 0,
    min_down_steps: int = 0,
    **kwargs,
) -> pyo.Block:
    """
    Represents a DRI (Direct Reduced Iron) plant.

    Args:
        model (pyomo.ConcreteModel): The Pyomo model where the DRI plant will be added.
        time_steps (list[int]): The list of time steps for the model.
        specific_hydrogen_consumption (float): The specific hydrogen consumption of the DRI plant (in MWh per ton of DRI).
        specific_natural_gas_consumption (float): The specific natural gas consumption of the DRI plant (in MWh per ton of DRI).
        specific_electricity_consumption (float): The specific electricity consumption of the DRI plant (in MWh per ton of DRI).
        specific_iron_ore_consumption (float): The specific iron ore consumption of the DRI plant (in ton per ton of DRI).
        rated_power (float): The rated power capacity of the DRI plant (in MW).
        min_power (float): The minimum power required for operation (in MW).
        fuel_type (str): The type of fuel used by the DRI plant ("hydrogen", "natural_gas", "both").
        ramp_up (float | None): The maximum rate at which the DRI plant can increase its power output (in MW/hr).
        ramp_down (float | None): The maximum rate at which the DRI plant can decrease its power output (in MW/hr).
        min_operating_steps (int): The minimum number of steps the DRI plant must operate continuously.
        min_down_steps (int): The minimum number of downtime steps required between operating cycles.

    Returns:
        pyo.Block: The DRI plant block.
    """
    # Set default ramp limits to rated_power if not provided
    ramp_up = rated_power if ramp_up is None else ramp_up
    ramp_down = rated_power if ramp_down is None else ramp_down

    # Define block
    model_part = pyo.Block()

    # Define parameters
    model_part.specific_hydrogen_consumption = pyo.Param(
        initialize=specific_hydrogen_consumption
    )
    model_part.specific_natural_gas_consumption = pyo.Param(
        initialize=specific_natural_gas_consumption
    )
    model_part.specific_electricity_consumption = pyo.Param(
        initialize=specific_electricity_consumption
    )
    model_part.specific_iron_ore_consumption = pyo.Param(
        initialize=specific_iron_ore_consumption
    )
    model_part.rated_power = pyo.Param(initialize=rated_power)
    model_part.min_power = pyo.Param(initialize=min_power)
    model_part.ramp_up = pyo.Param(initialize=ramp_up)
    model_part.ramp_down = pyo.Param(initialize=ramp_down)
    model_part.min_operating_steps = pyo.Param(initialize=min_operating_steps)
    model_part.min_down_steps = pyo.Param(initialize=min_down_steps)

    # Define variables
    model_part.power_dri = pyo.Var(
        time_steps, within=pyo.NonNegativeReals, bounds=(0, rated_power)
    )
    model_part.iron_ore_in = pyo.Var(time_steps, within=pyo.NonNegativeReals)
    model_part.natural_gas_in = pyo.Var(time_steps, within=pyo.NonNegativeReals)
    model_part.hydrogen_in = pyo.Var(time_steps, within=pyo.NonNegativeReals)
    model_part.dri_output = pyo.Var(time_steps, within=pyo.NonNegativeReals)
    model_part.dri_operating_cost = pyo.Var(time_steps, within=pyo.NonNegativeReals)

    # Define operational status variable if needed
    if min_operating_steps > 0 or min_down_steps > 0 or min_power > 0:
        model_part.operational_status = pyo.Var(time_steps, within=pyo.Binary)

        # Link power_dri and operational_status
        @model_part.Constraint(time_steps)
        def min_power_constraint(b, t):
            return b.power_dri[t] >= b.min_power * b.operational_status[t]

        @model_part.Constraint(time_steps)
        def max_power_constraint(b, t):
            return b.power_dri[t] <= b.rated_power * b.operational_status[t]

    # Fuel consumption constraint
    @model_part.Constraint(time_steps)
    def dri_output_constraint(b, t):
        if fuel_type == "hydrogen":
            return b.dri_output[t] == b.hydrogen_in[t] / b.specific_hydrogen_consumption
        elif fuel_type == "natural_gas":
            return (
                b.dri_output[t]
                == b.natural_gas_in[t] / b.specific_natural_gas_consumption
            )
        elif fuel_type == "both":
            return b.dri_output[t] == (
                b.hydrogen_in[t] / b.specific_hydrogen_consumption
            ) + (b.natural_gas_in[t] / b.specific_natural_gas_consumption)

    # Electricity consumption constraint
    @model_part.Constraint(time_steps)
    def electricity_consumption_constraint(b, t):
        return b.power_dri[t] == b.dri_output[t] * b.specific_electricity_consumption

    # Iron ore consumption constraint
    @model_part.Constraint(time_steps)
    def iron_ore_constraint(b, t):
        return b.iron_ore_in[t] == b.dri_output[t] * b.specific_iron_ore_consumption

    # Ramp-up constraint
    @model_part.Constraint(time_steps)
    def ramp_up_constraint(b, t):
        if t == 0:
            return pyo.Constraint.Skip
        return b.power_dri[t] - b.power_dri[t - 1] <= b.ramp_up

    # Ramp-down constraint
    @model_part.Constraint(time_steps)
    def ramp_down_constraint(b, t):
        if t == 0:
            return pyo.Constraint.Skip
        return b.power_dri[t - 1] - b.power_dri[t] <= b.ramp_down

    # Minimum operating time constraint
    if min_operating_steps > 0:

        @model_part.Constraint(time_steps)
        def min_operating_time_constraint(b, t):
            if t < min_operating_steps - 1:
                return pyo.Constraint.Skip
            relevant_time_steps = range(t - min_operating_steps + 1, t + 1)
            return (
                sum(b.operational_status[i] for i in relevant_time_steps)
                >= min_operating_steps * b.operational_status[t]
            )

    # Minimum downtime constraint
    if min_down_steps > 0:

        @model_part.Constraint(time_steps)
        def min_downtime_constraint(b, t):
            if t < min_down_steps - 1:
                return pyo.Constraint.Skip
            relevant_time_steps = range(t - min_down_steps + 1, t + 1)
            return sum(
                1 - b.operational_status[i] for i in relevant_time_steps
            ) >= min_down_steps * (1 - b.operational_status[t])

    # Operating cost constraint
    @model_part.Constraint(time_steps)
    def dri_operating_cost_constraint(b, t):
        return (
            b.dri_operating_cost[t]
            == b.natural_gas_in[t] * model.natural_gas_price[t]
            + b.power_dri[t] * model.electricity_price[t]
            + b.iron_ore_in[t] * model.iron_ore_price
        )

    return model_part


def create_electric_arc_furnace(
    model: pyo.ConcreteModel,
    time_steps: list[int],
    rated_power: float,
    min_power: float,
    specific_electricity_consumption: float,
    specific_dri_demand: float,
    specific_lime_demand: float,
    ramp_up: float | None = None,
    ramp_down: float | None = None,
    min_operating_steps: int = 0,
    min_down_steps: int = 0,
    **kwargs,
) -> pyo.Block:
    """
    Adds the Electric Arc Furnace (EAF) to the Pyomo model.

    Args:
        model (pyomo.ConcreteModel): The Pyomo model where the EAF will be added.
        time_steps (list[int]): The list of time steps for the model.
        rated_power (float): The rated power capacity of the electric arc furnace (in MW).
        min_power (float): The minimum power requirement of the electric arc furnace (in MW).
        specific_electricity_consumption (float): The specific electricity consumption of the electric arc furnace (in MWh per ton of steel produced).
        specific_dri_demand (float): The specific demand for Direct Reduced Iron (DRI) in the electric arc furnace (in tons per ton of steel produced).
        specific_lime_demand (float): The specific demand for lime in the electric arc furnace (in tons per ton of steel produced).
        ramp_up (float | None): The ramp-up rate of the electric arc furnace (in MW per hour).
        ramp_down (float | None): The ramp-down rate of the electric arc furnace (in MW per hour).
        min_operating_steps (int): The minimum number of steps the EAF must operate continuously.
        min_down_steps (int): The minimum number of downtime steps required between operating cycles.

    Returns:
        pyo.Block: The EAF block.
    """
    # Set default ramp limits to rated_power if not provided
    ramp_up = rated_power if ramp_up is None else ramp_up
    ramp_down = rated_power if ramp_down is None else ramp_down

    # Define block
    model_part = pyo.Block()

    # Define parameters
    model_part.rated_power = pyo.Param(initialize=rated_power)
    model_part.min_power = pyo.Param(initialize=min_power)
    model_part.specific_electricity_consumption = pyo.Param(
        initialize=specific_electricity_consumption
    )
    model_part.specific_dri_demand = pyo.Param(initialize=specific_dri_demand)
    model_part.specific_lime_demand = pyo.Param(initialize=specific_lime_demand)
    model_part.ramp_up = pyo.Param(initialize=ramp_up)
    model_part.ramp_down = pyo.Param(initialize=ramp_down)
    model_part.min_operating_steps = pyo.Param(initialize=min_operating_steps)
    model_part.min_down_steps = pyo.Param(initialize=min_down_steps)

    # Define variables
    model_part.power_eaf = pyo.Var(
        time_steps, within=pyo.NonNegativeReals, bounds=(0, rated_power)
    )
    model_part.dri_input = pyo.Var(time_steps, within=pyo.NonNegativeReals)
    model_part.steel_output = pyo.Var(time_steps, within=pyo.NonNegativeReals)
    model_part.eaf_operating_cost = pyo.Var(time_steps, within=pyo.NonNegativeReals)
    model_part.emission_eaf = pyo.Var(time_steps, within=pyo.NonNegativeReals)
    model_part.lime_demand = pyo.Var(time_steps, within=pyo.NonNegativeReals)

    # Define operational status variable if needed
    if min_operating_steps > 0 or min_down_steps > 0 or min_power > 0:
        model_part.operational_status = pyo.Var(time_steps, within=pyo.Binary)

        # Link power_eaf and operational_status
        @model_part.Constraint(time_steps)
        def min_power_constraint(b, t):
            return b.power_eaf[t] >= b.min_power * b.operational_status[t]

        @model_part.Constraint(time_steps)
        def max_power_constraint(b, t):
            return b.power_eaf[t] <= b.rated_power * b.operational_status[t]

    # Steel output based on DRI input
    @model_part.Constraint(time_steps)
    def steel_output_dri_relation(b, t):
        return b.steel_output[t] == b.dri_input[t] / b.specific_dri_demand

    # Steel output based on power consumption
    @model_part.Constraint(time_steps)
    def steel_output_power_relation(b, t):
        return b.power_eaf[t] == b.steel_output[t] * b.specific_electricity_consumption

    # Lime demand based on steel output
    @model_part.Constraint(time_steps)
    def eaf_lime_demand(b, t):
        return b.lime_demand[t] == b.steel_output[t] * b.specific_lime_demand

    # CO2 emissions based on lime demand
    @model_part.Constraint(time_steps)
    def eaf_co2_emission(b, t):
        return b.emission_eaf[t] == b.lime_demand[t] * model.lime_co2_factor

    # Ramp-up constraint
    @model_part.Constraint(time_steps)
    def ramp_up_eaf_constraint(b, t):
        if t == 0:
            return pyo.Constraint.Skip
        return b.power_eaf[t] - b.power_eaf[t - 1] <= b.ramp_up

    # Ramp-down constraint
    @model_part.Constraint(time_steps)
    def ramp_down_eaf_constraint(b, t):
        if t == 0:
            return pyo.Constraint.Skip
        return b.power_eaf[t - 1] - b.power_eaf[t] <= b.ramp_down

    # Minimum operating time constraint
    if min_operating_steps > 0:

        @model_part.Constraint(time_steps)
        def min_operating_time_constraint(b, t):
            if t < min_operating_steps - 1:
                return pyo.Constraint.Skip
            relevant_time_steps = range(t - min_operating_steps + 1, t + 1)
            return (
                sum(b.operational_status[i] for i in relevant_time_steps)
                >= min_operating_steps * b.operational_status[t]
            )

    # Minimum downtime constraint
    if min_down_steps > 0:

        @model_part.Constraint(time_steps)
        def min_down_time_constraint(b, t):
            if t < min_down_steps - 1:
                return pyo.Constraint.Skip
            relevant_time_steps = range(t - min_down_steps + 1, t + 1)
            return sum(
                1 - b.operational_status[i] for i in relevant_time_steps
            ) >= min_down_steps * (1 - b.operational_status[t])

    # Operating cost constraint
    @model_part.Constraint(time_steps)
    def eaf_operating_cost_constraint(b, t):
        return (
            b.eaf_operating_cost[t]
            == b.power_eaf[t] * model.electricity_price[t]
            + b.emission_eaf[t] * model.co2_price
            + b.lime_demand[t] * model.lime_price
        )

    return model_part
