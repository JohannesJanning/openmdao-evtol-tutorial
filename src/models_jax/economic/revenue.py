import jax.numpy as jnp


def revenue_per_flight(fare_km, distance_trip_km, N_s, LF):
    return fare_km * distance_trip_km * N_s * LF


def ticket_price_per_passenger(revenue_flight, N_s, LF):
    return revenue_flight / (N_s * LF)


def profit_per_flight(revenue_flight, TOC_flight):
    return revenue_flight - TOC_flight


def annual_profit(revenue_flight, TOC_flight, FC_a):
    return (revenue_flight - TOC_flight) * FC_a


def revenue_per_flight_pm(toc, pm):
    return toc / (1.0 - pm)
