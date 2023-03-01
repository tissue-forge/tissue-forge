/*******************************************************************************
 * This file is part of Tissue Forge.
 * Copyright (c) 2022, 2023 T.J. Sego
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

#include <state/tfSpecies.h>

%}


%rename(_state_Species) Species;

%include <state/tfSpecies.h>

%extend TissueForge::state::Species {
    %pythoncode %{
        def __str__(self) -> str:
            return self.str()

        def __reduce__(self):
            return self.fromString, (self.toString(),)

        @property
        def id(self) -> str:
            return self.getId()

        @id.setter
        def id(self, sid: str):
            self.setId(sid)

        @property
        def name(self) -> str:
            return self.getName()

        @name.setter
        def name(self, name: str):
            self.setName(name)

        @property
        def species_type(self) -> str:
            return self.getSpeciesType()

        @species_type.setter
        def species_type(self, sid: str):
            self.setSpeciesType(sid)

        @property
        def compartment(self) -> str:
            return self.getCompartment()

        @compartment.setter
        def compartment(self, sid: str):
            self.setCompartment(sid)

        @property
        def initial_amount(self) -> float:
            return self.getInitialAmount()

        @initial_amount.setter
        def initial_amount(self, value: float):
            self.setInitialAmount(value)

        @property
        def initial_concentration(self) -> float:
            return self.getInitialConcentration()

        @initial_concentration.setter
        def initial_concentration(self, value: float):
            self.setInitialConcentration(value)

        @property
        def substance_units(self) -> str:
            return self.getSubstanceUnits()

        @substance_units.setter
        def substance_units(self, sid: str):
            self.setSubstanceUnits(sid)

        @property
        def spatial_size_units(self) -> str:
            return self.getSpatialSizeUnits()

        @spatial_size_units.setter
        def spatial_size_units(self, sid: str):
            self.setSpatialSizeUnits(sid)

        @property
        def units(self) -> str:
            return self.getUnits()

        @units.setter
        def units(self, sname: str):
            self.setUnits(sname)

        @property
        def has_only_substance_units(self) -> bool:
            return self.getHasOnlySubstanceUnits()

        @has_only_substance_units.setter
        def has_only_substance_units(self, value: bool):
            self.setHasOnlySubstanceUnits(value);

        @property
        def boundary_condition(self) -> bool:
            return self.getBoundaryCondition()

        @boundary_condition.setter
        def boundary_condition(self, value: bool):
            self.setBoundaryCondition(value)

        @property
        def charge(self) -> int:
            return self.getCharge()

        @charge.setter
        def charge(self, value: int):
            self.setCharge(value)

        @property
        def constant(self) -> bool:
            return self.getConstant()

        @constant.setter
        def constant(self, value: bool):
            self.setConstant(value)

        @property
        def conversion_factor(self) -> str:
            return self.getConversionFactor()

        @conversion_factor.setter
        def conversion_factor(self, sid: str):
            self.setConversionFactor(sid)
    %}
}
