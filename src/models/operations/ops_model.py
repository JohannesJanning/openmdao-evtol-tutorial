def turnaround_time(c_charge: float, DOD: float) -> float:
    return (1 / c_charge) * DOD * 3600

def time_efficiency_ratio(time_turnaround: float, time_trip: float) -> float:
    return time_turnaround / time_trip + 1

def daily_flight_cycles(T_D: float, time_trip: float, DH: float) -> float:
    return T_D / (time_trip * DH)

def annual_flight_cycles(N_wd: int, FC_d: float) -> float:
    return N_wd * FC_d

def daily_flight_hours(FC_d: float, time_trip: float) -> float:
    return FC_d * time_trip / 3600

def annual_flight_hours(FC_a: float, time_trip: float) -> float:
    return FC_a * time_trip / 3600
