/*******************************************************************************
 * This file is part of Tissue Forge.
 * Copyright (c) 2022-2024 T.J. Sego
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 * 
 ******************************************************************************/

%{

#include "tfPotential.h"
#include <langs/py/tfPotentialPy.h>

%}


%ignore TissueForge::Potential::custom;
%ignore TissueForge::potential_err;
%ignore TissueForge::potential_null;
%rename(_call) TissueForge::Potential::operator();
%rename(custom) TissueForge::py::PotentialPy::customPy(FloatP_t, FloatP_t, PyObject*, PyObject*, PyObject*, FloatP_t*, uint32_t*);

%rename(_Potential) TissueForge::Potential;
%rename(Potential) TissueForge::py::PotentialPy;

%include "tfPotential.h"
%include <langs/py/tfPotentialPy.h>

%extend TissueForge::Potential {
    %pythoncode %{
        from enum import Enum as EnumPy

        class Constants(EnumPy):
            degree = potential_degree
            chunk = potential_chunk
            ivalsa = potential_ivalsa
            ivalsb = potential_ivalsb
            N = potential_N
            align = potential_align
            ivalsmax = potential_ivalsmax

        class Flags(EnumPy):
            none = POTENTIAL_NONE
            lj126 = POTENTIAL_LJ126
            ewald = POTENTIAL_EWALD
            coulomb = POTENTIAL_COULOMB
            single = POTENTIAL_SINGLE
            r2 = POTENTIAL_R2
            r = POTENTIAL_R
            angle = POTENTIAL_ANGLE
            harmonic = POTENTIAL_HARMONIC
            dihedral = POTENTIAL_DIHEDRAL
            switch = POTENTIAL_SWITCH
            reactive = POTENTIAL_REACTIVE
            scaled = POTENTIAL_SCALED
            shifted = POTENTIAL_SHIFTED
            bound = POTENTIAL_BOUND
            psum = POTENTIAL_SUM
            periodic = POTENTIAL_PERIODIC
            coulombr = POTENTIAL_COULOMBR

        class Kind(EnumPy):
            potential = POTENTIAL_KIND_POTENTIAL
            dpd = POTENTIAL_KIND_DPD
            byparticles = POTENTIAL_KIND_BYPARTICLES
            combination = POTENTIAL_KIND_COMBINATION

        def __call__(self, *args):
            return self._call(*args)

        @property
        def min(self) -> float:
            """Minimum distance of evaluation"""
            return self.getMin()

        @property
        def max(self) -> float:
            """Maximum distance of evaluation"""
            return self.getMax()

        @property
        def cutoff(self) -> float:
            """Cutoff distance"""
            return self.getCutoff()

        @property
        def domain(self) -> (float, float):
            """Evaluation domain"""
            return self.getDomain()

        @property
        def intervals(self) -> int:
            """Evaluation intervals"""
            return self.getIntervals()

        @property
        def bound(self) -> bool:
            """Bound flag"""
            return self.getBound()

        @bound.setter
        def bound(self, bound: bool):
            self.setBound(bound)

        @property
        def r0(self) -> float:
            """Potential r0 value"""
            return self.getR0()

        @r0.setter
        def r0(self, r0: float):
            self.setR0(r0)

        @property
        def shifted(self) -> bool:
            """Shifted flag"""
            return self.getShifted()

        @property
        def periodic(self) -> bool:
            """Periodic flag"""
            return self.getPeriodic()

        @property
        def r_square(self) -> bool:
            """Potential r2 value"""
            return self.getRSquare()

        def plot(self, s=1.0, force=True, potential=False, show=True, ymin=None, ymax=None, *args, **kwargs):
            """
            Potential plot function

            :param s: sum of theoretical radii of two interacting particles
            :param force: flag to plot evaluations of the force magnitude
            :param potential: flag to plot evaluations of the potential
            :param show: flag to show the plot
            :param ymin: minimum vertical plot value
            :param ymax: maximum vertical plot value
            :return: plot lines
            """
            import matplotlib.pyplot as plt
            import numpy as n
            import warnings

            min = kwargs["min"] if "min" in kwargs else 0.00001
            max = kwargs["max"] if "max" in kwargs else min + 1
            step = kwargs["step"] if "step" in kwargs else 0.001
            range = kwargs["range"] if "range" in kwargs else (min, max, step)

            if isinstance(min, float) or isinstance(min, int):
                xx = n.arange(*range)
            else:
                t = 0
                xx = list()
                while t <= 1:
                    xx.append((min + (max - min) * t).asVector())
                    t += step

            yforce = None
            ypot = None

            if force:
                sh = s / 2
                yforce = [self.force(x, sh, sh) for x in xx]

            if potential:
                ypot = [self(x, s) for x in xx]

            if not isinstance(xx[0], float):
                xx = [FVector3(xxx).length() for xxx in xx]

            if not ymin:
                y = n.array([])
                if yforce:
                    y = n.concatenate((y, n.asarray(yforce).flat))
                if ypot:
                    y = n.concatenate((y, ypot))
                ymin = n.amin(y)

            if not ymax:
                y = n.array([])
                if yforce:
                    y = n.concatenate((y, n.asarray(yforce).flat))
                if ypot:
                    y = n.concatenate((y, ypot))
                ymax = n.amax(y)

            yrange = n.abs(ymax - ymin)

            lines = None

            print("ymax: ", ymax, "ymin:", ymin, "yrange:", yrange)

            print("Ylim: ", ymin - 0.1 * yrange, ymax + 0.1 * yrange )

            plt.ylim(ymin - 0.1 * yrange, ymax + 0.1 * yrange )

            if yforce and not ypot:
                lines = plt.plot(xx, yforce, label='force')
            elif ypot and not yforce:
                lines = plt.plot(xx, ypot, label='potential')
            elif yforce and ypot:
                lines = [plt.plot(xx, yforce, label='force'), plt.plot(xx, ypot, label='potential')]

            plt.legend()

            plt.title(self.name)

            if show:
                plt.show()

            return lines

        def __reduce__(self):
            return Potential.fromString, (self.toString(),)
    %}
}
