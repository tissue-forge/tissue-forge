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

#include <state/tfSpeciesReactions.h>

%}

%ignore TissueForge::state::SpeciesReactions::models;

%rename(copy_from) TissueForge::state::SpeciesReactions::copyFrom;
%rename(map_values_to) TissueForge::state::SpeciesReactions::mapValuesTo;
%rename(map_values_from) TissueForge::state::SpeciesReactions::mapValuesFrom;

%rename(_state_SpeciesReactions) SpeciesReactions;

%include <state/tfSpeciesReactions.h>

%extend TissueForge::state::SpeciesReactions {
    %pythoncode %{
        def __getitem__(self, key: str):
            result = self._getModel(key)
            if result is None:
                raise KeyError
            return result

        def __reduce__(self):
            return self.fromString, (self.toString(),)

        @property
        def model_names(self):
            """List of model names"""
            return self._modelNames()
    %}
}
