# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import importlib
from pathlib import Path

# must be installed for import path to work
import calliope
import pandas as pd
import yaml

from assume.common.base import BaseStrategy
from assume.common.forecasts import NaiveForecast
from assume.world import World

# model = calliope.Model("path/to/model.yaml", override_dict=override_dict)
model = calliope.examples.urban_scale()


def read_csv(base_path, filename):
    return pd.read_csv(
        base_path + "/" + filename,
        date_format="%Y-%m-%d_%H:%M:%S",
        sep=";",
        header=None,
        names=["time", "load"],
        index_col="time",
    )["load"]


input_path = Path(importlib.resources.files("calliope") / "example_models")
name = "urban_scale"
# name = "national_scale"
model_path = input_path / name
# model = calliope.Model("path/to/model.yaml", override_dict=override_dict)

model = calliope.examples.urban_scale()

with open(model_path) as f:
    # TODO also load imported files in yaml
    model_data = yaml.safe_load(f)
    # TODO replace data_sources with actual data

    time_subset = model_data["config"]["init"]["time_subset"]

    start, end = time_subset


override_dict = {"config": {"init": {"time_subset": ["2005-07-07", "2005-07-08"]}}}


def update_dict_keys(initial_dict: dict, override_dict: dict):
    for key, value in override_dict.items():
        if isinstance(value, dict):
            if key in initial_dict:
                update_dict_keys(initial_dict[key], value)
            else:
                initial_dict[key] = value
        else:
            initial_dict[key] = value


async def load_calliope_async(
    world: World,
    name: str,
    input_path: Path,
    marketdesign: list[MarketConfig],
    bidding_strategies: dict[str, BaseStrategy],
    override_dict: dict = {},
):
    model_path = input_path / name
    with open(model_path / "model.yaml") as f:
        model_data = yaml.safe_load(f)

    for path in model_data["import"]:
        with open(model_path / path) as f:
            import_dict = yaml.safe_load(f)
            update_dict_keys(model_data, import_dict)

    update_dict_keys(model_data, override_dict)

    if model_data["config"]["build"]["mode"] == "plan":
        print(
            "ERROR: Plan not supported. Only operate is currently supported in ASSUME"
        )

    time_subset = model_data["config"]["init"]["time_subset"]

    start, end = time_subset

    save_interval = 48

    index = pd.date_range(start=start, end=end, freq="1h")

    # TODO replace data_sources with actual data

    await world.setup(
        start=start,
        end=end,
        save_frequency_hours=save_interval,
        simulation_id=name,
        index=index,
    )

    mo_id = "market_operator"
    world.add_market_operator(id=mo_id)

    for market_config in marketdesign:
        grid = None
        # TODO build grid from model_data["nodes"]
        market_config.param_dict["grid_data"] = grid
        world.add_market(mo_id, market_config)

    agents = []
    dfs = {}

    for tech in model_data["techs"]:
        print(tech)
    for node in model_data["nodes"]:
        print(node)
    for ds in model_data["data_sources"]:
        print(ds)

    # source_name = "demand"
    # data_source = model_data["data_sources"][source_name]
    for source_name, data_source in model_data["data_sources"].items():
        if source_name == "time_varying_parameters":
            continue
        if data_source["source"]:
            source_path = data_source["source"]
            if data_source.get("drop"):
                skip = 1
                header = 0
            else:
                skip = 0
                if isinstance(data_source["columns"], list):
                    header = list(range(len(data_source["columns"])))
                else:
                    header = 0
            df = pd.read_csv(
                model_path / source_path,
                header=header,
                skiprows=skip,
                parse_dates=True,
                index_col=0,
            )
        else:
            df = pd.DataFrame(index=index)
        dfs[source_name] = df
        # TODO don't know what this is for
        # data_source now has the actual tech
        tech = data_source["add_dimensions"].get("techs")

        translate_dict = {
            "demand_electricity": ("demand", "energy"),
            "demand_heat": ("demand", "heat"),
            "pv": ("solar", "energy"),
        }

        for column in df.columns:
            if isinstance(column, tuple):
                tech, node = column
            else:
                node = column

            tech_dict = model_data["techs"][tech]

            match tech_dict["base_tech"]:
                case "supply":
                    actual_tech, product_type = translate_dict[tech]
                    # pv_profile is availability
                    availability = df[tech_dict["source_unit"]]
                    max_power = tech_dict["flow_cap_max"]
                    area_use_max = tech_dict["area_use_max"]
                    world.add_unit_operator(source_name)
                    world.add_unit(
                        f"{source_name}1",
                        "power_plant",
                        source_name,  # units_operator name
                        # the unit_params have no hints
                        {
                            "min_power": 0,
                            "max_power": max_power,
                            "bidding_strategies": {"EOM": "naive_eom"},
                            "technology": actual_tech,
                            "node": node,
                        },
                        NaiveForecast(
                            index, availability=availability
                        ),  # hier zeitreihen hinterlegen
                    )

                case "conversion":
                    max_power = tech_dict["flow_cap_max"]["data"]
                    efficiency = tech_dict["flow_out_eff"]["data"]
                    heat_to_power_ratio = tech_dict["heat_to_power_ratio"]
                    tech_dict["carrier_in"]
                    tech_dict["carrier_out"]
                    tech_dict["cost_flow_cap"]["data"]  # maximaler verdienst?
                    tech_dict["cost_flow_out"]["data"]  # einspeisevergütung
                    {
                        "agent": tech + node,
                        "node": node,
                        "technology": "conversion",
                    }
                    print("conversion is currently not supported")
                    print("no strategy to bid on two markets yet")
                case "demand":
                    max_power = df[tech, node].max()
                    # normalize profile to values from 0 to 1
                    availability = df[tech, node] / max_power
                    min_power = 0
                    actual_tech, product_type = translate_dict.get(tech)
                    # TODO check if product type is demand or actual tech
                    world.add_unit_operator(source_name)
                    world.add_unit(
                        f"{source_name}1",
                        "demand",
                        source_name,  # units_operator name
                        # the unit_params have no hints
                        {
                            "min_power": min_power,
                            "max_power": max_power,
                            "bidding_strategies": {"EOM": "naive_eom"},
                            "technology": actual_tech,
                            "node": node,
                        },
                        NaiveForecast(
                            index,
                            demand=df[tech, node],
                            availability=availability,
                        ),  # hier sind die zeitreihen hinterlegt
                    )


if __name__ == "__main__":
    db_uri = "postgresql://assume:assume@localhost:5432/assume"
    world = World(database_uri=db_uri)
    scenario = "urban_scale"

    base_path = Path(importlib.resources.files("calliope") / "example_models")
    world.loop.run_until_complete(
        load_calliope_async(
            world,
            scenario,
            base_path,
        )
    )
    print(f"did load {scenario} - now simulating")
    world.run()
