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

#include <state/tfSpeciesReactionDef.h>

%}

%rename(uri_or_sbml) TissueForge::state::SpeciesReactionDef::uriOrSBML;
%rename(species_names) TissueForge::state::SpeciesReactionDef::speciesNames;
%rename(model_names) TissueForge::state::SpeciesReactionDef::modelNames;
%rename(step_size) TissueForge::state::SpeciesReactionDef::stepSize;
%rename(num_steps) TissueForge::state::SpeciesReactionDef::numSteps;
%rename(from_antimony_string) TissueForge::state::SpeciesReactionDef::fromAntimonyString;
%rename(from_antimony_file) TissueForge::state::SpeciesReactionDef::fromAntimonyFile;
%rename(integrator_name) TissueForge::state::SpeciesReactionDef::integratorName;
%rename(integrator_enum) TissueForge::state::SpeciesReactionDef::integratorEnum;

%rename(_state_SpeciesReactionDef) SpeciesReactionDef;

%include <state/tfSpeciesReactionDef.h>

%extend TissueForge::state::SpeciesReactionDef {
    %pythoncode %{
        def __reduce__(self):
            return self.fromString, (self.toString(),)
    %}
}
