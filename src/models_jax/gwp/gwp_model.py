import jax.numpy as jnp

def battery_lifecycle_gwp(GWP_battery, E_battery_design):
    return GWP_battery * E_battery_design / 1000.0

def battery_annual_ops_gwp(n_battery_required_annual, battery_LC_GWP):
    return n_battery_required_annual * battery_LC_GWP

def battery_flight_cycle_gwp(battery_annual_OPS_GWP, FC_a):
    return battery_annual_OPS_GWP / FC_a

def gwp_operational_per_cycle(energy_trip, GWP_energy):
    return (energy_trip / 1000.0) * GWP_energy

def gwp_flight(GWP_operational_per_cycle, battery_flight_cycle_GWP):
    return GWP_operational_per_cycle + battery_flight_cycle_GWP

def gwp_annual(GWP_flight, FC_a):
    return GWP_flight * FC_a

def gwp_energy_fraction(energy_trip, GWP_energy, GWP_flight):
    return (energy_trip / 1000.0 * GWP_energy) / GWP_flight

def gwp_battery_fraction(gwp_energy_fraction):
    return 1.0 - gwp_energy_fraction
