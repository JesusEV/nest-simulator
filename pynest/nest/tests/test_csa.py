# -*- coding: utf-8 -*-
#
# test_csa.py
#
# This file is part of NEST.
#
# Copyright (C) 2004 The NEST Initiative
#
# NEST is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# NEST is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with NEST.  If not, see <http://www.gnu.org/licenses/>.

"""
CSA tests
"""

import unittest
import nest

try:
    import csa
    HAVE_CSA = True
except ImportError:
    HAVE_CSA = False

try:
    import numpy
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False

nest.ll_api.sli_run("statusdict/have_libneurosim ::")
HAVE_LIBNEUROSIM = nest.ll_api.sli_pop()


@nest.ll_api.check_stack
@unittest.skipIf(not HAVE_CSA, 'Python CSA package is not available')
@unittest.skipIf(
    not HAVE_LIBNEUROSIM,
    'NEST was built without support for libneurosim'
)
class CSATestCase(unittest.TestCase):
    """CSA tests"""

    def test_CSA_OneToOne_params(self):
        """One-to-one connectivity using conngen Connect with paramters"""

        nest.ResetKernel()

        n_neurons = 4
        weight = 10000.0
        delay = 2.0

        sources = nest.Create("iaf_psc_alpha", n_neurons)
        targets = nest.Create("iaf_psc_alpha", n_neurons)

        # Create a connection set with values for weight and delay
        cs = csa.cset(csa.oneToOne, weight, delay)

        # Connect sources and targets using the connection set cs and
        # a parameter map mapping weight to position 0 in the value
        # set and delay to position 1
        params_map = {"weight": 0, "delay": 1}
        connspec = {"rule": "conngen", "cg": cs, "params_map": params_map}
        nest.Connect(pre, post, connspec)

        for i in range(n_neurons):
            # We expect all connections from sources to have the
            # correct targets, weights and delays
            conns = nest.GetStatus(nest.GetConnections(sources[i]))
            self.assertEqual(len(conns), 1)
            self.assertEqual(conns[0]["target"], targets[i].get('global_id'))
            self.assertEqual(conns[0]["weight"], weight)
            self.assertEqual(conns[0]["delay"], delay)

            # We expect the targets to have no connections at all
            conns = nest.GetStatus(nest.GetConnections(targets[i]))
            self.assertEqual(len(conns), 0)

    def test_CSA_OneToOne_synmodel(self):
        """One-to-one connectivity using conngen Connect with synmodel"""

        nest.ResetKernel()

        n_neurons = 4
        synmodel = "stdp_synapse"

        sources = nest.Create("iaf_psc_alpha", n_neurons)
        targets = nest.Create("iaf_psc_alpha", n_neurons)

        # Create a plain connection set
        cs = csa.cset(csa.oneToOne)

        # Connect with a non-standard synapse model
        connspec = {"rule": "conngen", "cg": cs}
        synspec = {'synapse_model': synmodel}
        nest.Connect(pre, post, connspec, synspec)

        for i in range(n_neurons):
            # We expect all connections to have the correct targets
            # and the non-standard synapse model set
            conns = nest.GetStatus(nest.GetConnections(sources[i]))
            self.assertEqual(len(conns), 1)
            self.assertEqual(conns[0]["target"], targets[i].get('global_id'))
            self.assertEqual(conns[0]["synapse_model"], synmodel)

            # We expect the targets to have no connections at all
            conns = nest.GetStatus(nest.GetConnections(targets[i]))
            self.assertEqual(len(conns), 0)

    def test_CSA_error_unknown_synapse(self):
        """
        Error handling of conngen Connect in case of unknown synapse model
        """

        nest.ResetKernel()

        # Create a plain connection set
        cs = csa.cset(csa.oneToOne)
        connspec = {"rule": "conngen", "cg": cs}
        synspec = {'synapse_model': synmodel}

        n_neurons = 4

        pop = nest.Create("iaf_psc_alpha", n_neurons)

        # We expect conngen Connect to fail with an UnknownSynapseType
        # exception if an unknown synapse model is given
        self.assertRaisesRegex(nest.kernel.NESTError, "UnknownSynapseType",
                               nest.Connect, pop, pop, connspec, synspec)


def suite():

    suite = unittest.makeSuite(CSATestCase, 'test')
    return suite


def run():
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())


if __name__ == "__main__":
    run()
