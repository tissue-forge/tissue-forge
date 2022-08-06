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

#include <rendering/tfArrowRenderer.h>

%}


%ignore TissueForge::rendering::ArrowInstanceData;
%ignore TissueForge::rendering::ArrowRenderer;
%ignore TissueForge::rendering::vectorFrameRotation;

%rename(_rendering_ArrowData) TissueForge::rendering::ArrowData;

%include <rendering/tfArrowRenderer.h>

%template(pairInt_ArrowData_p) std::pair<int, TissueForge::rendering::ArrowData*>;
