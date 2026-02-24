import openmdao.api as om


class GeometryComp(om.ExplicitComponent):
    """Compute simple geometry-derived spans used for constraints.

    Outputs:
        rotor_spacing: required wing spacing for hover rotors [m]
        vertiport_span: required vertiport span [m]
    """

    def initialize(self):
        self.options.declare('parameters')

    def setup(self):
        # inputs
        self.add_input('r_hover', val=1.0)

        # outputs
        self.add_output('rotor_spacing', val=1.0)
        self.add_output('vertiport_span', val=1.0)
        # provide analytic partials for total derivatives
        self.declare_partials('rotor_spacing', 'r_hover')
        self.declare_partials('vertiport_span', 'r_hover')

    def compute(self, inputs, outputs):
        params = self.options['parameters']
        r_hover = inputs['r_hover'][0]

        rotor_spacing = 2.0 * (3.0 * r_hover + 2.0 * params.d_rotors_space + params.r_fus_m)
        vertiport_span = 2.0 * (4.0 * r_hover + 2.0 * params.d_rotors_space + params.r_fus_m)

        outputs['rotor_spacing'] = float(rotor_spacing)
        outputs['vertiport_span'] = float(vertiport_span)

    def compute_partials(self, inputs, partials):
        # rotor_spacing = 2*(3*r_hover + 2*d_rotors_space + r_fus_m)
        # vertiport_span = 2*(4*r_hover + 2*d_rotors_space + r_fus_m)
        partials['rotor_spacing', 'r_hover'] = 6.0
        partials['vertiport_span', 'r_hover'] = 8.0
