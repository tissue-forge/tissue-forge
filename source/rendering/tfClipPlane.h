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

#ifndef _SOURCE_RENDERING_TFCUTPLANE_H_
#define _SOURCE_RENDERING_TFCUTPLANE_H_

#include <tfSpace_cell.h>
#include <types/tf_types.h>

#include <vector>


namespace TissueForge {


    CPPAPI_FUNC(std::vector<fVector4>) parsePlaneEquation(const std::vector<std::tuple<fVector3, fVector3> > &clipPlanes);


    namespace rendering {


        struct CAPI_EXPORT ClipPlane
        {
            /** Index of the clip plane. Less than zero if clip plane has been destroyed. */
            int index;
            ClipPlane(int i);

            /**
             * @brief Get the point of the clip plane
             * 
             * @return fVector3 
             */
            fVector3 getPoint();

            /**
             * @brief Get the normal vector of the clip plane
             * 
             * @return fVector3 
             */
            fVector3 getNormal();

            /**
             * @brief Get the coefficients of the plane equation of the clip plane
             * 
             * @return fVector4 
             */
            fVector4 getEquation();

            /**
             * @brief Set the coefficients of the plane equation of the clip plane
             * 
             * @param pe 
             * @return HRESULT 
             */
            HRESULT setEquation(const fVector4 &pe);

            /**
             * @brief Set the coefficients of the plane equation of the clip plane
             * using a point on the plane and its normal
             * 
             * @param point plane point
             * @param normal plane normal vector
             * @return HRESULT 
             */
            HRESULT setEquation(const fVector3 &point, const fVector3 &normal);

            /**
             * @brief Destroy the clip plane
             * 
             * @return HRESULT 
             */
            HRESULT destroy();
        };

        struct CAPI_EXPORT ClipPlanes {

            /**
             * @brief Get the number of clip planes
             * 
             * @return int 
             */
            static int len();

            /**
             * @brief Get the coefficients of the equation of a clip plane
             * 
             * @param index index of the clip plane
             * @return const fVector4& 
             */
            static const fVector4 &getClipPlaneEquation(const unsigned int &index);

            /**
             * @brief Set the coefficients of the equation of a clip plane. 
             * 
             * The clip plane must already exist
             * 
             * @param index index of the clip plane
             * @param pe coefficients of the plane equation of the clip plane
             * @return HRESULT 
             */
            static HRESULT setClipPlaneEquation(const unsigned int &index, const fVector4 &pe);

            /**
             * @brief Get a clip plane by index
             * 
             * @param index index of the clip plane
             * @return ClipPlane 
             */
            static ClipPlane item(const unsigned int &index);

            /**
             * @brief Create a clip plane
             * 
             * @param pe coefficients of the equation of the plane
             * @return ClipPlane 
             */
            static ClipPlane create(const fVector4 &pe);

            /**
             * @brief Create a clip plane
             * 
             * @param point point on the clip plane
             * @param normal normal of the clip plane
             * @return ClipPlane 
             */
            static ClipPlane create(const fVector3 &point, const fVector3 &normal);
        };

        /**
         * get a reference to the cut planes collection.
         */
        CPPAPI_FUNC(ClipPlanes*) getClipPlanes();

    }

}

#endif // _SOURCE_RENDERING_TFCUTPLANE_H_