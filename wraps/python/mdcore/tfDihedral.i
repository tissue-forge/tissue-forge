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

#include "tfDihedral.h"

%}


%rename(_getitem) TissueForge::DihedralHandle::operator[];

%ignore Dihedral_StylePtr;
%ignore dihedral_err;
%ignore dihedral_eval;
%ignore dihedral_evalf;
%ignore Dihedral_IdsForParticle;

%include "tfDihedral.h"

%template(vectorDihedralHandle) std::vector<TissueForge::DihedralHandle>;
%template(vectorDihedralHandle_p) std::vector<TissueForge::DihedralHandle*>;

%extend TissueForge::Dihedral {
    %pythoncode %{
        @property
        def style_def(self):
            """default style"""
            return self.styleDef();

        def __reduce__(self):
            return Dihedral.fromString, (self.toString(),)
    %}
}

%extend TissueForge::DihedralHandle {
    %pythoncode %{
        def __getitem__(self, index: int):
            return self.parts.__getitem__(index)

        def __contains__(self, item):
            return self.has(item)

        def __str__(self) -> str:
            return self.str()

        def __lt__(self, rhs) -> bool:
            return self.id < rhs.id

        def __gt__(self, rhs) -> bool:
            return rhs < self

        def __le__(self, rhs) -> bool:
            return not (self > rhs)

        def __ge__(self, rhs) -> bool:
            return not (self < rhs)

        def __eq__(self, rhs) -> bool:
            return self.id == rhs.id

        def __ne__(self, rhs) -> bool:
            return not (self == rhs)

        @property
        def angle(self):
            """dihedral angle"""
            return self.getAngle()

        @property
        def energy(self):
            """dihedral energy"""
            return self.getEnergy()

        @property
        def parts(self):
            """bonded particles"""
            return ParticleList(self.getParts())

        @property
        def potential(self):
            """dihedral potential"""
            return self.getPotential()

        @property
        def id(self):
            """dihedral id"""
            return self.getId()

        @property
        def dissociation_energy(self):
            """bond dissociation energy"""
            return self.getDissociationEnergy()

        @dissociation_energy.setter
        def dissociation_energy(self, dissociation_energy):
            self.setDissociationEnergy(dissociation_energy)

        @property
        def half_life(self):
            """dihedral half life"""
            return self.getHalfLife()

        @half_life.setter
        def half_life(self, half_life):
            self.setHalfLife(half_life)

        @property
        def active(self):
            """active flag"""
            return self.getActive()

        @property
        def style(self):
            """dihedral style"""
            return self.getStyle()

        @style.setter
        def style(self, style):
            self.setStyle(style)

        @property
        def age(self):
            """dihedral age"""
            return self.getAge()
    %}
}
