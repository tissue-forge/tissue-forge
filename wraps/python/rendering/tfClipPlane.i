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

#include <rendering/tfClipPlane.h>

%}


%rename(_rendering_ClipPlane) TissueForge::rendering::ClipPlane;
%rename(_rendering_ClipPlanes) TissueForge::rendering::ClipPlanes;

%include <rendering/tfClipPlane.h>

%extend TissueForge::rendering::ClipPlane {
    %pythoncode %{
        @property
        def point(self) -> fVector3:
            return self.getPoint()

        @property
        def normal(self) -> fVector3:
            return self.getNormal()

        @property
        def equation(self) -> fVector4:
            return self.getEquation()

        @equation.setter
        def equation(self, _equation):
            if isinstance(_equation, list):
                p = fVector4(_equation)
            elif isinstance(_equation, fVector4):
                p = _equation
            else:
                p = fVector4(list(_equation))
            return self.setEquation(p)
    %}
}

%extend TissueForge::rendering::ClipPlanes {
    %pythoncode %{
        def __len__(self) -> int:
            return self.len()

        def __getitem__(self, item: int):
            return self.getClipPlaneEquation(item)

        def __setitem__(self, item: int, val):
            self.setClipPlaneEquation(item, val)
    %}
}
