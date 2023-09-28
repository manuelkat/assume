from datetime import datetime, timedelta
from itertools import product

import numpy as np
import pandas as pd

from assume.common.market_objects import Order


def create_orderbook(order: Order = None, node_ids=[0], count=100, seed=30):
    if not order:
        start = datetime.today()
        end = datetime.today() + timedelta(hours=1)
        order: Order = {
            "start_time": start,
            "end_time": end,
            "agent_id": "dem1",
            "bid_id": "bid1",
            "volume": 0,
            "price": 0,
            "only_hours": None,
            "node_id": 0,
        }
    orders = []
    np.random.seed(seed)

    for node_id, i in product(node_ids, range(count)):
        new_order = order.copy()
        new_order["price"] = np.random.randint(100)
        new_order["volume"] = np.random.randint(-10, 10)
        if new_order["volume"] > 0:
            agent_id = f"gen_{i}"
        else:
            agent_id = f"dem_{i}"
        new_order["agent_id"] = agent_id
        new_order["bid_id"] = f"bid_{i}"
        new_order["node_id"] = node_id
        orders.append(new_order)
    return orders


def extend_orderbook(
    products,
    volume,
    price,
    orderbook=None,
    bid_type="SB",
    min_acceptance_ratio=None,
):
    """
    Creates constant bids over the time span of all products
    with specified values for price and volume
    and appends the orderbook
    """
    if not orderbook:
        orderbook = []
    if volume == 0:
        return orderbook

    if bid_type == "BB":
        if volume < 0:
            agent_id = f"block_dem{len(orderbook)+1}"
        else:
            agent_id = f"block_gen{len(orderbook)+1}"

        order: Order = {
            "start_time": products[0][0],
            "end_time": products[0][1],
            "agent_id": agent_id,
            "bid_id": f"bid_{len(orderbook)+1}",
            "volume": {product[0]: volume for product in products},
            "accepted_volume": {},
            "price": price,
            "accepted_price": {},
            "only_hours": None,
            "bid_type": bid_type,
        }

        if min_acceptance_ratio is not None:
            order.update({"min_acceptance_ratio": min_acceptance_ratio})
        else:
            order.update({"min_acceptance_ratio": 0})

        orderbook.append(order)

    else:
        if volume < 0:
            agent_id = f"dem{len(orderbook)+1}"
        else:
            agent_id = f"gen{len(orderbook)+1}"

        for product in products:
            order: Order = {
                "start_time": product[0],
                "end_time": product[1],
                "agent_id": agent_id,
                "bid_id": f"bid_{len(orderbook)+1}",
                "volume": volume,
                "accepted_volume": 0,
                "price": price,
                "accepted_price": None,
                "only_hours": None,
                "bid_type": bid_type,
            }

            if min_acceptance_ratio is not None:
                order.update({"min_acceptance_ratio": min_acceptance_ratio})
            else:
                order.update({"min_acceptance_ratio": 0})

            orderbook.append(order)

    return orderbook


def get_test_prices(num: int = 24):
    power_price = 50 * np.ones(num)
    # power_price[18:24] = 0
    # power_price[24:] = 20
    co2 = np.ones(num) * 23.8
    # * np.random.uniform(0.95, 1.05, 48)     # -- Emission Price     [€/t]
    gas = np.ones(num) * 0.03
    # * np.random.uniform(0.95, 1.05, 48)    # -- Gas Price          [€/kWh]
    lignite = np.ones(num) * 0.015
    # * np.random.uniform(0.95, 1.05)   # -- Lignite Price      [€/kWh]
    coal = np.ones(num) * 0.02
    # * np.random.uniform(0.95, 1.05)       # -- Hard Coal Price    [€/kWh]
    nuclear = np.ones(num) * 0.01
    # * np.random.uniform(0.95, 1.05)        # -- nuclear Price      [€/kWh]

    prices = dict(
        power=power_price, gas=gas, co2=co2, lignite=lignite, coal=coal, nuclear=nuclear
    )
    prices = pd.DataFrame(
        data=prices, index=pd.date_range(start="2018-01-01", freq="h", periods=num)
    )

    return prices
