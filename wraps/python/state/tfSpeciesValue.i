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

#include <state/tfSpeciesValue.h>

%}


%rename(_secrete1) TissueForge::state::SpeciesValue::secrete(const FloatP_t &, const struct TissueForge::ParticleList &);
%rename(_secrete2) TissueForge::state::SpeciesValue::secrete(const FloatP_t &, const FloatP_t &);

%rename(_state_SpeciesValue) SpeciesValue;

%include <state/tfSpeciesValue.h>

%extend TissueForge::state::SpeciesValue {
    %pythoncode %{
        def secrete(self, amount, to=None, distance=None):
            """
            Secrete this species into a neighborhood.

            Requires either a list of neighboring particles or neighborhood distance.

            :param amount: Amount to secrete.
            :type amount: float
            :param to: Optional list of particles to secrete to.
            :type to: ParticleList
            :param distance: Neighborhood distance.
            :type distance: float
            :return: Amount actually secreted, accounting for availability and other subtleties. 
            :rtype: float
            """

            if to is not None:
                return self._secrete1(amount, to)
            elif distance is not None:
                return self._secrete2(amount, distance)
            raise ValueError('A neighbor list or neighbor distance must be specified')

        @property
        def boundary_condition(self) -> bool:
            return self.getBoundaryCondition()

        @boundary_condition.setter
        def boundary_condition(self, value: int):
            self.setBoundaryCondition(value)

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
        def constant(self) -> bool:
            return self.getConstant()

        @constant.setter
        def constant(self, value: int):
            self.setConstant(value)

        @property
        def value(self) -> float:
            return self.getValue()

        @value.setter
        def value(self, _value: float):
            self.setValue(_value)
    %}
}
