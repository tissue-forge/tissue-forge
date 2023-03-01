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

#ifndef _SOURCE_RENDERING_TFORIENTATIONRENDERER_H_
#define _SOURCE_RENDERING_TFORIENTATIONRENDERER_H_

#include "tfSubRenderer.h"
#include "tfArrowRenderer.h"

#include <shaders/tfPhong.h>
#include "tfStyle.h"

#include <Magnum/GL/Mesh.h>
#include <Magnum/Math/Color.h>
#include <Magnum/Mesh.h>

#include <utility>
#include <vector>


namespace TissueForge::rendering {


    /**
     * @brief Orientaton renderer. 
     * 
     * Visualizes the orientation of the scene. 
     * 
     */
    struct OrientationRenderer : SubRenderer {

        // Arrow inventory
        std::vector<ArrowData *> arrows;

        OrientationRenderer();
        ~OrientationRenderer();

        HRESULT start(const std::vector<fVector4> &clipPlanes) override;
        HRESULT draw(ArcBallCamera *camera, const iVector2 &viewportSize, const fMatrix4 &modelViewMat) override;
        void setAmbientColor(const Magnum::Color3& color) override;
        void setDiffuseColor(const Magnum::Color3& color) override;
        void setSpecularColor(const Magnum::Color3& color) override;
        void setShininess(float shininess) override;
        void setLightDirection(const fVector3& lightDir) override;
        void setLightColor(const Magnum::Color3 &color) override;

        /**
         * @brief Gets the global instance of the renderer. 
         * 
         * Cannot be used until the universe renderer has been initialized. 
         * 
         * @return OrientationRenderer* 
         */
        static OrientationRenderer *get();

        void showAxes(const bool &show) {
            _showAxes = show;
        }

    private:

        int _arrowDetail = 10;
        int _showAxes;

        Magnum::GL::Buffer _bufferHead{Corrade::Containers::NoCreate};
        Magnum::GL::Buffer _bufferCylinder{Corrade::Containers::NoCreate};
        Magnum::GL::Buffer _bufferOrigin{Corrade::Containers::NoCreate};
        Magnum::GL::Mesh _meshHead{Corrade::Containers::NoCreate};
        Magnum::GL::Mesh _meshCylinder{Corrade::Containers::NoCreate};
        Magnum::GL::Mesh _meshOrigin{Corrade::Containers::NoCreate};
        shaders::Phong _shader{Corrade::Containers::NoCreate};

        fMatrix4 modelViewMat;
        fMatrix4 staticTransformationMat;

        ArrowData *arrowx=NULL, *arrowy=NULL, *arrowz=NULL;

    };

};

#endif // _SOURCE_RENDERING_TFORIENTATIONRENDERER_H_
