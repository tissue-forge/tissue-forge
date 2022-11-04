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

#include "tfError.h"

%}

%ignore TissueForge::errSet;
%ignore TissueForge::expSet;

%rename(err_occurred) TissueForge::errOccurred;
%rename(err_clear) TissueForge::errClear;
%rename(_msg) TissueForge::Error::msg;

%include "tfError.h"

%extend TissueForge::Error {
    %pythoncode %{
        def __str__(self) -> str:
            return errStr(self)
    %}
}
