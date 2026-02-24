import openmdao.api as om
import numpy as np
import jax
import jax.numpy as jnp

from src.models_jax.momentum.V_cruise import cruise_speed
from src.models_jax.momentum.V_climb import climb_speed
from src.models_jax.momentum.V_climb_horizontal import horizontal_climb_speed
from src.models_jax.momentum.T_climb_total import total_thrust_required_climb
from src.models_jax.momentum.T_cruise_total import total_thrust_required_cruise
from src.models_jax.momentum.T_hover_total import total_thrust_required_hover
from src.models_jax.momentum.T_prop import thrust_per_propeller
from src.models_jax.momentum.A_disk import propeller_disk_area
from src.models_jax.momentum.Disk_Loading import disk_loading_hover
from src.models_jax.momentum.P_hover_total import power_required_hover
from src.models_jax.momentum.P_total_hor import power_total_required
from src.models_jax.aerodynamics.drag import drag_calculation


class PerformanceComp(om.ExplicitComponent):
    """Compute speeds and required powers from geometry and MTOM.

    Inputs: b,c, CL_cruise, CD_cruise, CL_climb, CD_climb, MTOM
    Outputs: V_cruise, V_climb, V_climb_hor, P_req_total_hover, P_req_total_climb, P_req_total_cruise
    """

    def initialize(self):
        self.options.declare('parameters')

    def setup(self):
        self.add_input('b', val=10.0)
        self.add_input('c', val=1.5)
        self.add_input('CL_cruise', val=0.5)
        self.add_input('CD_cruise', val=0.02)
        self.add_input('CL_climb', val=0.6)
        self.add_input('CD_climb', val=0.03)
        self.add_input('MTOM', val=1500.0)
        self.add_input('r_cruise', val=1.5)
        self.add_input('r_hover', val=0.5)

        self.add_output('V_cruise', val=0.0)
        self.add_output('V_climb', val=0.0)
        self.add_output('V_climb_hor', val=0.0)
        self.add_output('P_req_total_hover', val=0.0)
        self.add_output('P_req_total_climb', val=0.0)
        self.add_output('P_req_total_cruise', val=0.0)

        # We'll provide analytic JAX partials in compute_partials
        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        p = self.options['parameters']
        b = inputs['b'][0]
        c = inputs['c'][0]
        CLc = inputs['CL_cruise'][0]
        CDc = inputs['CD_cruise'][0]
        CLcl = inputs['CL_climb'][0]
        CDcl = inputs['CD_climb'][0]
        MTOM = inputs['MTOM'][0]

        # speeds
        V_cruise = float(np.asarray(cruise_speed(MTOM, p.g, CLc, CDc, p.alpha_deg_cruise, c, b, p.rho)))
        V_climb = float(np.asarray(climb_speed(MTOM, p.g, p.theta_deg_climb, CLcl, c, b, p.rho)))
        V_climb_hor = float(np.asarray(horizontal_climb_speed(V_climb, p.theta_deg_climb)))

        # compute prop areas
        r_cr = inputs['r_cruise'][0]
        r_hv = inputs['r_hover'][0]

        A_prop_hor = float(np.asarray(propeller_disk_area(r_cr)))
        A_hover = float(np.asarray(propeller_disk_area(r_hv)))

        # climb
        # include aerodynamic drag in climb and cruise thrust calculations
        D_climb = float(np.asarray(drag_calculation(p.rho, V_climb, c, b, CDcl)))
        T_req_total_climb = float(np.asarray(total_thrust_required_climb(D_climb, MTOM, p.g, p.theta_deg_climb)))
        T_req_prop_climb = float(np.asarray(thrust_per_propeller(T_req_total_climb, p.n_prop_hor)))
        P_req_total_climb = float(np.asarray(power_total_required(V_climb, T_req_total_climb, T_req_prop_climb, p.rho, A_prop_hor, p.n_prop_hor, p.eta_c)))

        # cruise (D placeholder until drag component is added)
        D_cruise = float(np.asarray(drag_calculation(p.rho, V_cruise, c, b, CDc)))
        T_req_total_cruise = float(np.asarray(total_thrust_required_cruise(D_cruise, p.alpha_deg_cruise)))
        T_req_prop_cruise = float(np.asarray(thrust_per_propeller(T_req_total_cruise, p.n_prop_hor)))
        P_req_total_cruise = float(np.asarray(power_total_required(V_cruise, T_req_total_cruise, T_req_prop_cruise, p.rho, A_prop_hor, p.n_prop_hor, p.eta_c)))

        # hover
        T_req_total_hover = float(np.asarray(total_thrust_required_hover(MTOM, p.g)))
        T_req_prop_hover = float(np.asarray(thrust_per_propeller(T_req_total_hover, p.n_prop_vert)))
        sigma_hover = float(np.asarray(disk_loading_hover(T_req_prop_hover, A_hover)))
        P_req_total_hover = float(np.asarray(power_required_hover(sigma_hover, T_req_total_hover, p.rho, p.eta_h)))

        outputs['V_cruise'] = V_cruise
        outputs['V_climb'] = V_climb
        outputs['V_climb_hor'] = V_climb_hor
        outputs['P_req_total_hover'] = P_req_total_hover
        outputs['P_req_total_climb'] = P_req_total_climb
        outputs['P_req_total_cruise'] = P_req_total_cruise

    def compute_partials(self, inputs, partials):
        # Build input vector in a stable order
        in_names = ['b', 'c', 'CL_cruise', 'CD_cruise', 'CL_climb', 'CD_climb', 'MTOM', 'r_cruise', 'r_hover']
        out_names = ['V_cruise', 'V_climb', 'V_climb_hor', 'P_req_total_hover', 'P_req_total_climb', 'P_req_total_cruise']

        x = jnp.array([inputs[n][0] for n in in_names])

        def fun(x):
            b, c, CLc, CDc, CLcl, CDcl, MTOM, r_cr, r_hv = x

            V_cruise = cruise_speed(MTOM, self.options['parameters'].g, CLc, CDc, self.options['parameters'].alpha_deg_cruise, c, b, self.options['parameters'].rho)
            V_climb = climb_speed(MTOM, self.options['parameters'].g, self.options['parameters'].theta_deg_climb, CLcl, c, b, self.options['parameters'].rho)
            V_climb_hor = horizontal_climb_speed(V_climb, self.options['parameters'].theta_deg_climb)

            A_prop_hor = propeller_disk_area(r_cr)
            A_hover = propeller_disk_area(r_hv)

            D_climb = drag_calculation(self.options['parameters'].rho, V_climb, c, b, CDcl)
            T_req_total_climb = total_thrust_required_climb(D_climb, MTOM, self.options['parameters'].g, self.options['parameters'].theta_deg_climb)
            T_req_prop_climb = thrust_per_propeller(T_req_total_climb, self.options['parameters'].n_prop_hor)
            P_req_total_climb = power_total_required(V_climb, T_req_total_climb, T_req_prop_climb, self.options['parameters'].rho, A_prop_hor, self.options['parameters'].n_prop_hor, self.options['parameters'].eta_c)

            D_cruise = drag_calculation(self.options['parameters'].rho, V_cruise, c, b, CDc)
            T_req_total_cruise = total_thrust_required_cruise(D_cruise, self.options['parameters'].alpha_deg_cruise)
            T_req_prop_cruise = thrust_per_propeller(T_req_total_cruise, self.options['parameters'].n_prop_hor)
            P_req_total_cruise = power_total_required(V_cruise, T_req_total_cruise, T_req_prop_cruise, self.options['parameters'].rho, A_prop_hor, self.options['parameters'].n_prop_hor, self.options['parameters'].eta_c)

            T_req_total_hover = total_thrust_required_hover(MTOM, self.options['parameters'].g)
            T_req_prop_hover = thrust_per_propeller(T_req_total_hover, self.options['parameters'].n_prop_vert)
            sigma_hover = disk_loading_hover(T_req_prop_hover, A_hover)
            P_req_total_hover = power_required_hover(sigma_hover, T_req_total_hover, self.options['parameters'].rho, self.options['parameters'].eta_h)

            return jnp.array([V_cruise, V_climb, V_climb_hor, P_req_total_hover, P_req_total_climb, P_req_total_cruise])

        # compute jacobian (outputs x inputs)
        J = jax.jacfwd(fun)(x)
        J = np.asarray(J)
        # sanitize any NaN/Inf entries produced by AD
        J = np.nan_to_num(J, nan=0.0, posinf=0.0, neginf=0.0)

        # assign partials
        for i, out in enumerate(out_names):
            for j, inp in enumerate(in_names):
                partials[(out, inp)] = J[i, j]
