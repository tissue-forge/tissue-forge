/*******************************************************************************
 * This file is part of mdcore.
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

#ifndef _MDCORE_SOURCE_TF_ENGINE_ADVANCE_H_
#define _MDCORE_SOURCE_TF_ENGINE_ADVANCE_H_


namespace TissueForge{ 


    /**
     * @brief Update the particle velocities and positions, re-shuffle if
     *      appropriate.
     * @param e The #engine on which to run.
     *
     * @return #engine_err_ok or < 0 on error (see #engine_err).
     */
    CAPI_FUNC(int) engine_advance(struct engine *e);

};

#endif // _MDCORE_SOURCE_TF_ENGINE_ADVANCE_H_