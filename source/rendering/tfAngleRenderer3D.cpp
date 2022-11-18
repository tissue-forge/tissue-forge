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

#include "tfAngleRenderer3D.h"

#include <Magnum/Trade/MeshData.h>
#include <Magnum/MeshTools/Compile.h>
#include <Magnum/Primitives/Cylinder.h>

#include "tfArrowRenderer.h"
#include "tfBondRenderer3D.h"
#include <tfAngle.h>
#include <tfEngine.h>
#include "tfUniverseRenderer.h"
#include <tfLogger.h>
#include "tfStyle.h"
#include <tf_metrics.h>


using namespace TissueForge;


HRESULT rendering::AngleRenderer3D::start(const std::vector<fVector4> &clipPlanes) {

    // create the shader
    _shaderArcs = shaders::Phong {
        shaders::Phong::Flag::VertexColor, 
        1, 
        (unsigned int)rendering::UniverseRenderer::maxClipPlaneCount()
    };
    _shaderBnds = shaders::Phong {
        shaders::Phong::Flag::VertexColor | 
        shaders::Phong::Flag::InstancedTransformation, 
        1, 
        (unsigned int)rendering::UniverseRenderer::maxClipPlaneCount()
    };
    _shaderArcs.setShininess(2000.0f)
        .setLightPositions({{-20, 40, 20, 0.f}})
        .setLightColors({Magnum::Color3{0.9, 0.9, 0.9}})
        .setShininess(100)
        .setAmbientColor({0.4, 0.4, 0.4, 1})
        .setDiffuseColor({1, 1, 1, 0})
        .setSpecularColor({0.2, 0.2, 0.2, 0});
    _shaderBnds.setShininess(2000.0f)
        .setLightPositions({{-20, 40, 20, 0.f}})
        .setLightColors({Magnum::Color3{0.9, 0.9, 0.9}})
        .setShininess(100)
        .setAmbientColor({0.4, 0.4, 0.4, 1})
        .setDiffuseColor({1, 1, 1, 0})
        .setSpecularColor({0.2, 0.2, 0.2, 0});
    
    // create the buffers
    _bufferArcs = Magnum::GL::Buffer();
    _bufferBnds = Magnum::GL::Buffer();

    // create the meshes
    _meshArcs = Magnum::GL::Mesh{};
    _meshArcs.setPrimitive(Magnum::MeshPrimitive::Triangles);
    
    unsigned int numRings = _detailBnds;
    unsigned int numSegments = _detailBnds;
    _meshBnds = Magnum::MeshTools::compile(Magnum::Primitives::cylinderSolid(
        numRings, 
        numSegments, 
        0.5
    ));

    _meshArcs.addVertexBuffer(_bufferArcs, 0, 
        Magnum::Shaders::Phong::Position{}, 
        Magnum::Shaders::Phong::Normal{}, 
        Magnum::Shaders::Phong::Color4{}
    );
    _meshBnds.addVertexBufferInstanced(_bufferBnds, 1, 0, 
        Magnum::Shaders::Phong::TransformationMatrix{}, 
        Magnum::Shaders::Phong::NormalMatrix{}, 
        Magnum::Shaders::Phong::Color4{}
    );

    _meshArcs.setCount(0);
    _meshBnds.setInstanceCount(0);

    return S_OK;
}

void rendering::render_arc3d(
    rendering::Angle3DInstanceData *arcData, 
    const unsigned int &idx, 
    const fVector3 &posi, 
    const fVector3 &posj, 
    const fVector3 &posk, 
    const rendering::Style &s, 
    const unsigned int &numSegments, 
    const unsigned int &faceDetail, 
    const float &radius)
{
    fVector3 relpos0 = metrics::relativePosition(posi, posj);
    const fVector3 relpos1 = metrics::relativePosition(posk, posj);
    const float lenpos0 = relpos0.length();

    const float len = std::min(lenpos0, relpos1.length()) * 0.5; // length of relative displacement to current cross-section
    if(len == 0) 
        return;

    fVector3 _rotAxis = relpos0.cross(relpos1);
    if(_rotAxis.dot() == 0) 
        _rotAxis = fMatrix4::from(rendering::vectorFrameRotation(relpos0), {0.f, 0.f, 0.f}).transformVector({0, 1, 0});
    const fVector3 rotAxis = _rotAxis.normalized(); // axis about which to rotate vector to current cross-section

    fVector3 dir = relpos0 / lenpos0; // relative displacement to current cross-section w.r.t. the origin
    fVector3 posa, posb = posj + dir * len; // position of two current cross-sections

    std::vector<fVector3> templatePoints, templateNormals; // template cross-section points
    std::vector<fVector3> pointsa, pointsb; // current cross-section points
    std::vector<fVector3> normalsa, normalsb; // current cross-section normals
    templatePoints.reserve(faceDetail); templateNormals.reserve(faceDetail);
    pointsa.reserve(faceDetail); pointsb.reserve(faceDetail);
    normalsa.reserve(faceDetail); normalsb.reserve(faceDetail);
    Magnum::Math::Rad templateAngleIncr(2.f * (float)M_PI / (faceDetail));
    fMatrix4 ptTransform_b = fMatrix4::from(fMatrix3(dir, rotAxis, dir.cross(rotAxis)), posb);
    for(size_t i = 0; i < faceDetail; i++) {
        std::pair<float, float> angcoords = Magnum::Math::sincos(templateAngleIncr * i);
        const fVector3 vn(angcoords.first, angcoords.second, 0);
        templateNormals.push_back(vn);
        normalsb.push_back(ptTransform_b.transformVector(vn));
        const fVector3 pt = vn * radius;
        templatePoints.push_back(pt);
        pointsb.push_back(ptTransform_b.transformPoint(pt));
    }

    // transforms relative displacement of cross-section at every segment
    const fMatrix4 dirTransform = fMatrix4::rotation(relpos0.angle(relpos1) / (float)(numSegments), rotAxis);
    const Magnum::Color4 color = {s.color[0], s.color[1], s.color[2], (float)s.getVisible()};
    const size_t vertexIdxIncr = 6 * faceDetail;
    for(size_t i = 0; i < numSegments; i++) {
        
        // get the current cross-section locations

        posa = posb;
        dir = dirTransform.transformVector(dir);
        posb = posj + dir * len;
        ptTransform_b = fMatrix4::from(fMatrix3(dir, rotAxis, dir.cross(rotAxis)), posb);

        // get the current cross-section points

        pointsa = pointsb;
        normalsa = normalsb;
        for(size_t j = 0; j < faceDetail; j++) {
            pointsb[j] = ptTransform_b.transformPoint(templatePoints[j]);
            normalsb[j] = ptTransform_b.transformVector(templateNormals[j]);
        }

        // construct the surfaces

        rendering::Angle3DInstanceData *_arcData = &arcData[idx + i * vertexIdxIncr];
        for(size_t j = 0; j < faceDetail; j++) {
            const fVector3 via = pointsa[j];
            const fVector3 vja = pointsa[j == faceDetail - 1 ? 0 : j + 1];
            const fVector3 vib = pointsb[j];
            const fVector3 vjb = pointsb[j == faceDetail - 1 ? 0 : j + 1];

            const fVector3 nia = normalsa[j];
            const fVector3 nja = normalsa[j == faceDetail - 1 ? 0 : j + 1];
            const fVector3 nib = normalsb[j];
            const fVector3 njb = normalsb[j == faceDetail - 1 ? 0 : j + 1];

            rendering::Angle3DInstanceData *_arcData_j = &_arcData[6 * j];

            _arcData_j[0].position = vja;
            _arcData_j[0].normal = nja;
            _arcData_j[0].color = color;
            _arcData_j[1].position = via;
            _arcData_j[1].normal = nia;
            _arcData_j[1].color = color;
            _arcData_j[2].position = vib;
            _arcData_j[2].normal = nib;
            _arcData_j[2].color = color;

            _arcData_j[3].position = vib;
            _arcData_j[3].normal = nib;
            _arcData_j[3].color = color;
            _arcData_j[4].position = vjb;
            _arcData_j[4].normal = njb;
            _arcData_j[4].color = color;
            _arcData_j[5].position = vja;
            _arcData_j[5].normal = nja;
            _arcData_j[5].color = color;
        }

    }

}

HRESULT rendering::AngleRenderer3D::draw(rendering::ArcBallCamera *camera, const iVector2 &viewportSize, const fMatrix4 &modelViewMat) {
    TF_Log(LOG_DEBUG) << "";

    if(_Engine.nr_active_angles > 0) {
        int vertsPerArc = _segmentsArcs * _detailBnds * 6;
        int countArcs = _Engine.nr_active_angles * vertsPerArc;
        int countBnds = _Engine.nr_active_angles * 2;
        _meshArcs.setCount(countArcs);
        _meshBnds.setInstanceCount(countBnds);

        _bufferArcs.setData(
            {NULL, countArcs * sizeof(Angle3DInstanceData)},
            Magnum::GL::BufferUsage::DynamicDraw
        );
        _bufferBnds.setData(
            {NULL, countBnds * sizeof(Bond3DInstanceData)},
            Magnum::GL::BufferUsage::DynamicDraw
        );
        
        Angle3DInstanceData* dataArcs = (Angle3DInstanceData*)(void*)_bufferArcs.map(
           0,
           countArcs * sizeof(Angle3DInstanceData),
           Magnum::GL::Buffer::MapFlag::Write|Magnum::GL::Buffer::MapFlag::InvalidateBuffer
        );
        Bond3DInstanceData* dataBnds = (Bond3DInstanceData*)(void*)_bufferBnds.map(
           0,
           countBnds * sizeof(Bond3DInstanceData),
           Magnum::GL::Buffer::MapFlag::Write|Magnum::GL::Buffer::MapFlag::InvalidateBuffer
        );
        
        int i = 0;
        float radius;
        fVector3 papos, pbpos;
        for(int j = 0; j < _Engine.nr_angles; ++j) {
            Angle *bond = &_Engine.angles[j];
            if(bond->flags & ANGLE_ACTIVE) {
                Particle *pi = _Engine.s.partlist[bond->i];
                Particle *pj = _Engine.s.partlist[bond->j];
                Particle *pk = _Engine.s.partlist[bond->k];

                radius = _radiusBnds > 0 ? _radiusBnds : std::min(std::min(pi->radius, pj->radius), pk->radius) * 0.4;

                const fVector3 pjpos = pj->global_position();
                const fVector3 pipos = pjpos + metrics::relativePosition(pi->global_position(), pjpos);
                const fVector3 pkpos = pjpos + metrics::relativePosition(pk->global_position(), pjpos);

                render_arc3d(dataArcs, i * vertsPerArc, pipos, pjpos, pkpos, *bond->style, _segmentsArcs, _detailBnds, radius * 0.5);
                render_bond3d(dataBnds, i * 2    , pipos, pjpos, *bond->style, radius);
                render_bond3d(dataBnds, i * 2 + 1, pkpos, pjpos, *bond->style, radius);
                i++;
            }
        }
        assert(i == _Engine.nr_active_angles);
        _bufferArcs.unmap();
        _bufferBnds.unmap();
        
        _shaderArcs
            .setProjectionMatrix(camera->projectionMatrix())
            .setTransformationMatrix(camera->cameraMatrix() * modelViewMat)
            .setNormalMatrix(camera->viewMatrix().normalMatrix())
            .draw(_meshArcs);
        _shaderBnds
            .setProjectionMatrix(camera->projectionMatrix())
            .setTransformationMatrix(camera->cameraMatrix() * modelViewMat)
            .setNormalMatrix(camera->viewMatrix().normalMatrix())
            .draw(_meshBnds);
    }
    return S_OK;
}

const unsigned rendering::AngleRenderer3D::addClipPlaneEquation(const Magnum::Vector4& pe) {
    unsigned int id = _clipPlanes.size();
    _clipPlanes.push_back(pe);
    _shaderBnds.setclipPlaneEquation(id, pe);
    return id;
}

const unsigned rendering::AngleRenderer3D::removeClipPlaneEquation(const unsigned int &id) {
    _clipPlanes.erase(_clipPlanes.begin() + id);

    for(unsigned int i = id; i < _clipPlanes.size(); i++) {
        _shaderBnds.setclipPlaneEquation(i, _clipPlanes[i]);
    }

    return _clipPlanes.size();
}

void rendering::AngleRenderer3D::setClipPlaneEquation(unsigned id, const Magnum::Vector4& pe) {
    if(id > _shaderBnds.clipPlaneCount()) {
        tf_exp(std::invalid_argument("invalid id for clip plane"));
    }

    _shaderBnds.setclipPlaneEquation(id, pe);
    _clipPlanes[id] = pe;
}
