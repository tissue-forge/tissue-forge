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

#include <state/tfSpeciesList.h>

%}


%rename(_state_SpeciesList) SpeciesList;

%template(vectorSpecies) std::vector<TissueForge::state::Species*>;

%include <state/tfSpeciesList.h>

%extend TissueForge::state::SpeciesList {
    %pythoncode %{
        def __str__(self) -> str:
            return self.str()

        def __len__(self) -> int:
            return self.size()

        def __getattr__(self, item: str):
            if item == 'this':
                return object.__getattr__(self, item)

            result = self[item]
            if result is not None:
                return result
            raise AttributeError

        def __setattr__(self, item: str, value) -> None:
            if item == 'this':
                return object.__setattr__(self, item, value)

            if self.item(item) is not None:
                if not isinstance(value, _state_Species):
                    raise TypeError("Not a species")
                self.insert(value)
                return
            return object.__setattr__(self, item, value)

        def __getitem__(self, item) -> _state_Species:
            if isinstance(item, str):
                item = self.index_of(item)

            if item < len(self):
                return self.item(item)
            return None

        def __setitem__(self, item, value) -> None:
            if self.item(item) is not None:
                if not isinstance(value, _state_Species):
                    raise TypeError("Not a species")
                self.insert(value)

        def __reduce__(self):
            return self.fromString, (self.toString(),)
    %}
}
