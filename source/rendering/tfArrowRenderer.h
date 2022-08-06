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

#ifndef _SOURCE_RENDERING_TFARROWRENDERER_H_
#define _SOURCE_RENDERING_TFARROWRENDERER_H_

#include "tfSubRenderer.h"

#include <shaders/tfPhong.h>
#include "tfStyle.h"

#include <Magnum/GL/Mesh.h>
#include <Magnum/Math/Color.h>
#include <Magnum/Mesh.h>

#include <utility>
#include <vector>


namespace TissueForge::rendering {


    /**
     * @brief Vector visualization specification. 
     * 
     * ArrowRenderer uses instances of ArrowData to visualize 
     * vectors as arrows in the simulation domain. 
     * 
     */
    struct CAPI_EXPORT ArrowData {
        // Position of the arrow
        fVector3 position;

        // Vector components of the arrow
        fVector3 components;

        // Tissue Forge style
        Style style;

        // Scaling applied to arrow
        float scale = 1.0;

    private:

        int id;
        friend struct ArrowRenderer;
    };

    struct ArrowInstanceData {
        Magnum::Matrix4 transformationMatrix;
        Magnum::Matrix3 normalMatrix;
        Magnum::Color4 color;
    };

    /**
     * @brief Vector renderer. 
     * 
     * Vector visualization specification can be passed 
     * dynamically. Visualization specs are not managed by 
     * the renderer. It is the responsibility of the client 
     * to manage specs appropriately. 
     * 
     * By default, a vector is visualized with the same 
     * orientation as its underlying data, where one unit 
     * of magnitude of the vector corresponds to a visualized 
     * arrow with a length of one in the scene. 
     * 
     */
    struct ArrowRenderer : SubRenderer {

        // Current number of arrows in inventory
        int nr_arrows;

        // Arrow inventory
        std::vector<ArrowData *> arrows;

        ArrowRenderer();
        ArrowRenderer(const ArrowRenderer &other);
        ~ArrowRenderer();

        HRESULT start(const std::vector<fVector4> &clipPlanes) override;
        HRESULT draw(ArcBallCamera *camera, const iVector2 &viewportSize, const fMatrix4 &modelViewMat) override;
        const unsigned addClipPlaneEquation(const Magnum::Vector4& pe) override;
        const unsigned removeClipPlaneEquation(const unsigned int &id) override;
        void setClipPlaneEquation(unsigned id, const Magnum::Vector4& pe) override;
        void setAmbientColor(const Magnum::Color3& color) override;
        void setDiffuseColor(const Magnum::Color3& color) override;
        void setSpecularColor(const Magnum::Color3& color) override;
        void setShininess(float shininess) override;
        void setLightDirection(const fVector3& lightDir) override;
        void setLightColor(const Magnum::Color3 &color) override;

        /**
        * @brief Adds a vector visualization specification. 
        * 
        * The passed pointer is borrowed. The client is 
        * responsible for maintaining the underlying data. 
        * The returned integer can be used to reference the 
        * arrow when doing subsequent operations with the 
        * renderer (e.g., removing an arrow from the scene). 
        * 
        * @param arrow pointer to visualization specs
        * @return id of arrow according to the renderer
        */
        int addArrow(ArrowData *arrow);

        /**
        * @brief Adds a vector visualization specification. 
        * 
        * The passed pointer is borrowed. The client is 
        * responsible for maintaining the underlying data. 
        * The returned integer can be used to reference the 
        * arrow when doing subsequent operations with the 
        * renderer (e.g., removing an arrow from the scene). 
        * 
        * @param position position of vector
        * @param components components of vector
        * @param style style of vector
        * @param scale scale of vector; defaults to 1.0
        * @return id of arrow according to the renderer and arrow
        */
        std::pair<int, ArrowData*> addArrow(
            const fVector3 &position, 
            const fVector3 &components, 
            const Style &style, 
            const float &scale=1.0
        );

        /**
        * @brief Removes a vector visualization specification. 
        * 
        * The removed pointer is only forgotten. The client is 
        * responsible for clearing the underlying data. 
        * 
        * @param arrowId id of arrow according to the renderer
        */
        HRESULT removeArrow(const int &arrowId);

        /**
        * @brief Gets a vector visualization specification. 
        * 
        * @param arrowId id of arrow according to the renderer
        */
        ArrowData *getArrow(const int &arrowId);

        /**
        * @brief Gets the global instance of the renderer. 
        * 
        * Cannot be used until the universe renderer has been initialized. 
        */
        static ArrowRenderer *get();

    private:

        int _arrowDetail = 10;
        
        std::vector<Magnum::Vector4> _clipPlanes;

        Magnum::GL::Buffer _bufferHead{Corrade::Containers::NoCreate};
        Magnum::GL::Buffer _bufferCylinder{Corrade::Containers::NoCreate};
        Magnum::GL::Mesh _meshHead{Corrade::Containers::NoCreate};
        Magnum::GL::Mesh _meshCylinder{Corrade::Containers::NoCreate};
        shaders::Phong _shader{Corrade::Containers::NoCreate};

        /**
        * @brief Get the next data id
        * 
        * @return int 
        */
        int nextDataId();

    };

    /**
    * @brief Generates a 3x3 rotation matrix into the frame of a vector. 
    * 
    * The orientation of the second and third axes of the resulting transformation are arbitrary. 
    * 
    * @param vec Vector along which the first axis of the transformed frame is aligned. 
    * @return fMatrix3 
    */
    fMatrix3 vectorFrameRotation(const fVector3 &vec);

};

#endif // _SOURCE_RENDERING_TFARROWRENDERER_H_