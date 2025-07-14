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

#include "tfApplication.h"
#include "tfWindowlessApplication.h"

#include <Magnum/GL/Buffer.h>
#include <Magnum/GL/Framebuffer.h>
#include <Magnum/GL/Mesh.h>
#include <Magnum/GL/Renderbuffer.h>
#include <Magnum/GL/RenderbufferFormat.h>
#include <Magnum/GL/DefaultFramebuffer.h>
#include <Magnum/GL/Renderer.h>
#include <Magnum/GL/PixelFormat.h>

#include <Magnum/Shaders/VertexColor.h>
#include <Magnum/PixelFormat.h>
#include <Magnum/Image.h>
#include <Magnum/Animation/Easing.h>


#include <Corrade/Utility/Directory.h>
#include <Corrade/Utility/String.h>
#include <Magnum/Math/Color.h>

#include <tf_util.h>
#include <tfLogger.h>


using namespace Magnum;
using namespace TissueForge;


#include <iostream>


static Magnum::GL::AbstractFramebuffer *getFrameBuffer() {
    util::PerformanceTimer t1(engine_timer_image_data);
    util::PerformanceTimer t2(engine_timer_render_total);
    
    TF_Log(LOG_TRACE);
    
    if(!Magnum::GL::Context::hasCurrent()) {
        tf_error(E_FAIL, "No current OpenGL context");
        return NULL;
    }
    
    Simulator *sim = Simulator::get();
    
    sim->app->redraw();
    
    return &sim->app->framebuffer();
}

typedef Corrade::Containers::Array<char> (*imgCnv_t)(ImageView2D);
typedef Corrade::Containers::Array<char> (*imgGen_t)();

Corrade::Containers::Array<char> getImageData(imgCnv_t imgCnv, const PixelFormat &format) {
    util::PerformanceTimer t1(engine_timer_image_data);
    util::PerformanceTimer t2(engine_timer_render_total);
    
    TF_Log(LOG_TRACE);
    
    Magnum::GL::AbstractFramebuffer *framebuffer = getFrameBuffer();

    if(!framebuffer) 
        return Corrade::Containers::Array<char>();

    return imgCnv(framebuffer->read(framebuffer->viewport(), format));
}

Corrade::Containers::Array<char> rendering::JpegImageData() {
    return getImageData((imgCnv_t)[](ImageView2D image) { return convertImageDataToJpeg(image, 100); }, PixelFormat::RGB8Unorm);
}

Corrade::Containers::Array<char> rendering::BMPImageData() {
    return getImageData((imgCnv_t)convertImageDataToBMP, PixelFormat::RGB8Unorm);
}

Corrade::Containers::Array<char> rendering::HDRImageData() {
    return getImageData((imgCnv_t)convertImageDataToHDR, PixelFormat::RGB32F);
}

Corrade::Containers::Array<char> rendering::PNGImageData() {
    return getImageData((imgCnv_t)convertImageDataToPNG, PixelFormat::RGBA8Unorm);
}

Corrade::Containers::Array<char> rendering::TGAImageData() {
    return getImageData((imgCnv_t)convertImageDataToTGA, PixelFormat::RGBA8Unorm);
}

std::tuple<char*, size_t> rendering::framebufferImageData() {
    util::PerformanceTimer t1(engine_timer_image_data);
    util::PerformanceTimer t2(engine_timer_render_total);
    
    TF_Log(LOG_TRACE);
    
    auto jpegData = rendering::JpegImageData();

    return std::make_tuple(jpegData.data(), jpegData.size());
}

HRESULT rendering::screenshot(const std::string &filePath) {
    TF_Log(LOG_TRACE);

    std::string filePath_l = Utility::String::lowercase(Containers::StringView(filePath));

    imgGen_t imgGen;

    if(Utility::String::endsWith(filePath_l, ".bmp"))
        imgGen = rendering::BMPImageData;
    else if(Utility::String::endsWith(filePath_l, ".hdr"))
        imgGen = rendering::HDRImageData;
    else if(Utility::String::endsWith(filePath_l, ".jpe") || Utility::String::endsWith(filePath_l, ".jpg") || Utility::String::endsWith(filePath_l, ".jpeg"))
        imgGen = rendering::JpegImageData;
    else if(Utility::String::endsWith(filePath_l, ".png"))
        imgGen = rendering::PNGImageData;
    else if(Utility::String::endsWith(filePath_l, ".tga"))
        imgGen = rendering::TGAImageData;
    else {
        TF_Log(LOG_ERROR) << "Cannot determined file format from file path: " << filePath;
        return E_FAIL;
    }
    
    if(!Utility::Directory::write(filePath, imgGen())) {
        std::string msg = "Cannot write to file: " + filePath;
        tf_error(E_FAIL, msg.c_str());
        return E_FAIL;
    }

    return S_OK;
}

HRESULT rendering::Application::simulationStep() {
    
    /* Pause simulation if the mouse was pressed (camera is moving around).
     This avoid freezing GUI while running the simulation */
     
     // TODO: move substeps to universe step.
    
    static Float offset = 0.0f;
    if(_dynamicBoundary) {
        /* Change fluid boundary */
        static Float step = 2.0e-3f;
        if(_boundaryOffset > 1.0f || _boundaryOffset < 0.0f) {
            step *= -1.0f;
        }
        _boundaryOffset += step;
        offset = Math::lerp(0.0f, 0.5f, Animation::Easing::quadraticInOut(_boundaryOffset));
    }
    
    currentStep += 1;
    
    // TODO: get rid of this
    return Universe::step(0,0);
}

HRESULT rendering::Application::run(double et)
{
    TF_Log(LOG_TRACE);
    Universe_SetFlag(Universe::Flags::RUNNING, true);
    HRESULT result = messageLoop(et);
    Universe_SetFlag(Universe::Flags::RUNNING, false);
    return result;
}
