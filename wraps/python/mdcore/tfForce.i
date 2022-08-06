/*******************************************************************************
 * This file is part of Tissue Forge.
 * Copyright (c) 2022 T.J. Sego
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

#include "tfForce.h"
#include <langs/py/tfForcePy.h>

%}


%rename(_CustomForce) TissueForge::CustomForce;
%rename(CustomForce) TissueForge::py::CustomForcePy;

%include "tfForce.h"
%include <langs/py/tfForcePy.h>

%extend TissueForge::Force {
    %pythoncode %{
        def __reduce__(self):
            return Force.fromString, (self.toString(),)
    %}
}

%extend TissueForge::ForceSum {
    %pythoncode %{
        def __reduce__(self):
            return ForceSum_fromStr, (self.toString(),)
    %}
}

%extend TissueForge::Berendsen {
    %pythoncode %{
        def __reduce__(self):
            return Berendsen_fromStr, (self.toString(),)
    %}
}

%extend TissueForge::Gaussian {
    %pythoncode %{
        def __reduce__(self):
            return Gaussian_fromStr, (self.toString(),)
    %}
}

%extend TissueForge::Friction {
    %pythoncode %{
        def __reduce__(self):
            return Friction_fromStr, (self.toString(),)
    %}
}

%extend TissueForge::py::CustomForcePy {
    %pythoncode %{
        @property
        def value(self):
            """
            Current value of the force. 
            
            This can be set to a function that takes no arguments and returns a 3-component list of floats. 
            """
            return self.getValue()

        @value.setter
        def value(self, value):
            self.setValue(value)

        @property
        def period(self):
            """Period of the force"""
            return self.getPeriod()

        @period.setter
        def period(self, period):
            self.setPeriod(period)
    %}
}
