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

#include <tfDPDPotential.h>

#include <io/tfFIO.h>
#include <tf_mdcore_io.h>

#include <cmath>
#include <limits>


using namespace TissueForge;


DPDPotential::DPDPotential(FPTYPE alpha, FPTYPE gamma, FPTYPE sigma, FPTYPE cutoff, bool shifted) : Potential() {
    this->kind = POTENTIAL_KIND_DPD;
    this->alpha = alpha;
    this->gamma = gamma;
    this->sigma = sigma;
    this->a = std::sqrt(std::numeric_limits<FPTYPE>::epsilon());
    this->b = cutoff;
    this->name = "Dissapative Particle Dynamics";
    if(shifted) {
        this->flags |= POTENTIAL_SHIFTED;
    }
}

DPDPotential *DPDPotential::fromPot(Potential *pot) {
    if(pot->kind != POTENTIAL_KIND_DPD) 
        return NULL;
    return (DPDPotential*)pot;
}

std::string DPDPotential::toString() {
    io::IOElement fe = io::IOElement::create();
    io::MetaData metaData;
    if(io::toFile(this, metaData, fe) != S_OK) 
        return "";
    return io::toStr(fe, metaData);
}

DPDPotential *DPDPotential::fromString(const std::string &str) {
    return io::fromString<DPDPotential*>(str);
}


namespace TissueForge::io {


    HRESULT toFile(DPDPotential *dataElement, const MetaData &metaData, IOElement &fileElement) {

        TF_IOTOEASY(fileElement, metaData, "kind", dataElement->kind);
        TF_IOTOEASY(fileElement, metaData, "alpha", dataElement->alpha);
        TF_IOTOEASY(fileElement, metaData, "gamma", dataElement->gamma);
        TF_IOTOEASY(fileElement, metaData, "sigma", dataElement->sigma);
        TF_IOTOEASY(fileElement, metaData, "a", dataElement->a);
        TF_IOTOEASY(fileElement, metaData, "b", dataElement->b);
        TF_IOTOEASY(fileElement, metaData, "name", std::string(dataElement->name));
        TF_IOTOEASY(fileElement, metaData, "flags", dataElement->flags);

        fileElement.get()->type = "DPDPotential";

        return S_OK;
    }

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, DPDPotential **dataElement) {

        uint32_t kind, flags;
        TF_IOFROMEASY(fileElement, metaData, "kind", &kind);
        TF_IOFROMEASY(fileElement, metaData, "flags", &flags);

        FPTYPE alpha, gamma, sigma, b;
        TF_IOFROMEASY(fileElement, metaData, "alpha", &alpha);
        TF_IOFROMEASY(fileElement, metaData, "gamma", &gamma);
        TF_IOFROMEASY(fileElement, metaData, "sigma", &sigma);
        TF_IOFROMEASY(fileElement, metaData, "b", &b);

        *dataElement = new DPDPotential(alpha, gamma, sigma, b, flags & POTENTIAL_SHIFTED);

        return S_OK;
    }

};

DPDPotential *TissueForge::DPDPotential_fromStr(const std::string &str) {
    return DPDPotential::fromString(str);
}