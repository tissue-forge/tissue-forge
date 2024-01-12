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

#include "tfLogger.h"

%}


%include "tfLogger.h"

%extend TissueForge::Logger {
    %pythoncode %{
        CURRENT = LOG_CURRENT  #: :meta hide-value:
        FATAL = LOG_FATAL  #: :meta hide-value:
        CRITICAL = LOG_CRITICAL  #: :meta hide-value:
        ERROR = LOG_ERROR  #: :meta hide-value:
        WARNING = LOG_WARNING  #: :meta hide-value:
        NOTICE = LOG_NOTICE  #: :meta hide-value:
        INFORMATION = LOG_INFORMATION  #: :meta hide-value:
        DEBUG = LOG_DEBUG  #: :meta hide-value:
        TRACE = LOG_TRACE  #: :meta hide-value:
    %}
}
