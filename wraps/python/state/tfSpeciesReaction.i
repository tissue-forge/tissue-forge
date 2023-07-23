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

#include <state/tfSpeciesReaction.h>

%}

%rename(_mapName1) TissueForge::state::SpeciesReaction::mapName(const std::string&, const std::string&);
%rename(_mapName2) TissueForge::state::SpeciesReaction::mapName(const std::string&);
%rename(_getSpeciesNames) TissueForge::state::SpeciesReaction::getSpeciesNames;
%rename(_getModelNames) TissueForge::state::SpeciesReaction::getModelNames;
%rename(reset_map) TissueForge::state::SpeciesReaction::resetMap;
%rename(map_values_to) TissueForge::state::SpeciesReaction::mapValuesTo;
%rename(map_values_from) TissueForge::state::SpeciesReaction::mapValuesFrom;
%rename(_setStepSize) TissueForge::state::SpeciesReaction::setStepSize;
%rename(_getStepSize) TissueForge::state::SpeciesReaction::getStepSize;
%rename(_setNumSteps) TissueForge::state::SpeciesReaction::setNumSteps;
%rename(_getNumSteps) TissueForge::state::SpeciesReaction::getNumSteps;
%rename(_setCurrentTime) TissueForge::state::SpeciesReaction::setCurrentTime;
%rename(_getCurrentTime) TissueForge::state::SpeciesReaction::getCurrentTime;
%rename(_setModelValue) TissueForge::state::SpeciesReaction::setModelValue;
%rename(_getModelValue) TissueForge::state::SpeciesReaction::getModelValue;
%rename(_hasModelValue) TissueForge::state::SpeciesReaction::hasModelValue;
%rename(_getIntegratorName) TissueForge::state::SpeciesReaction::getIntegratorName;
%rename(_setIntegratorName) TissueForge::state::SpeciesReaction::setIntegratorName;
%rename(has_integrator) TissueForge::state::SpeciesReaction::hasIntegrator;
%rename(_getIntegrator) TissueForge::state::SpeciesReaction::getIntegrator;
%rename(_setIntegrator) TissueForge::state::SpeciesReaction::setIntegrator;

%rename(_state_SpeciesReaction) SpeciesReaction;

%include <state/tfSpeciesReaction.h>

%extend TissueForge::state::SpeciesReaction {
    %pythoncode %{
        from enum import Enum as EnumPy

        class Integrators(EnumPy):
            none = SPECIESREACTIONINT_NONE
            cvode = SPECIESREACTIONINT_CVODE
            euler = SPECIESREACTIONINT_EULER
            gillespie = SPECIESREACTIONINT_GILLESPIE
            nleq1 = SPECIESREACTIONINT_NLEQ1
            nleq2 = SPECIESREACTIONINT_NLEQ2
            rk4 = SPECIESREACTIONINT_RK4
            rk45 = SPECIESREACTIONINT_RK4

        def map_name(self, species_name: str, model_name: str = None):
            """
            Map a species name to a reaction model variable name

            :param species_name: name of species
            :param model_name: name of model variable, optional (default species_name)
            :return: None
            """
            if model_name is None:
                return self._mapName2(species_name)
            return self._mapName1(species_name, model_name)

        @property
        def species_names(self):
            """List of mapped species names"""
            return self._getSpeciesNames()

        @property
        def model_names(self):
            """List of mapped reaction model variable names"""
            return self._getModelNames()

        def __getitem__(self, key: str) -> float:
            return self._getModelValue(key)

        def __setitem__(self, key: str, val: float):
            self._setModelValue(key, val)

        step_size = property(_getStepSize, _setStepSize, doc="Period of model time per flux step.")
        num_steps = property(_getNumSteps, _setNumSteps, doc="Number of reaction simulator steps per integration step")
        current_time = property(_getCurrentTime, _setCurrentTime, "Current reaction model time")
        integrator_name = property(_getIntegratorName, _setIntegratorName, "Name of integrator")
        integrator = property(_getIntegrator, _setIntegrator, "Integrator enum")

        def __reduce__(self):
            return self.fromString, (self.toString(),)
    %}
}
