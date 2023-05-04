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

#include <state/tfStateVector.h>

%}


#include <state/SpeciesList.h>

%rename(_state_StateVector) StateVector;

%include <state/tfStateVector.h>

%extend TissueForge::state::StateVector {
    %pythoncode %{
        def __str__(self) -> str:
            return self.str()

        def __len__(self) -> int:
            return self.species.size()

        def __getattr__(self, item: str):
            if item == 'this':
                return object.__getattr__(self, item)

            from . import _tissue_forge

            sl = _tissue_forge._state_StateVector_species_get(self)
            idx = sl.index_of(item)
            if idx >= 0:
                return _state_SpeciesValue(self, idx)
            raise AttributeError

        def __setattr__(self, item: str, value: float) -> None:
            if item == 'this':
                return object.__setattr__(self, item, value)

            idx = self.species.index_of(item)
            if idx >= 0:
                self.setItem(idx, value)
                return
            return object.__setattr__(self, item, value)

        def __getitem__(self, item: int):
            if item < len(self):
                return self.item(item)
            return None

        def __setitem__(self, item: int, value: float) -> None:
            if item < len(self):
                self.setItem(item, value)

        def __reduce__(self):
            return self.fromString, (self.toString(),)
            
    %}
}
