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
    io::IOElement *fe = new io::IOElement();
    io::MetaData metaData;
    if(io::toFile(this, metaData, fe) != S_OK) 
        return "";
    return io::toStr(fe, metaData);
}

DPDPotential *DPDPotential::fromString(const std::string &str) {
    return io::fromString<DPDPotential*>(str);
}


namespace TissueForge::io {


    #define TF_DPDPOTENTIALIOTOEASY(fe, key, member) \
        fe = new IOElement(); \
        if(toFile(member, metaData, fe) != S_OK)  \
            return E_FAIL; \
        fe->parent = fileElement; \
        fileElement->children[key] = fe;

    #define TF_DPDPOTENTIALIOFROMEASY(feItr, children, metaData, key, member_p) \
        feItr = children.find(key); \
        if(feItr == children.end() || fromFile(*feItr->second, metaData, member_p) != S_OK) \
            return E_FAIL;

    HRESULT toFile(DPDPotential *dataElement, const MetaData &metaData, IOElement *fileElement) {

        IOElement *fe;

        TF_DPDPOTENTIALIOTOEASY(fe, "kind", dataElement->kind);
        TF_DPDPOTENTIALIOTOEASY(fe, "alpha", dataElement->alpha);
        TF_DPDPOTENTIALIOTOEASY(fe, "gamma", dataElement->gamma);
        TF_DPDPOTENTIALIOTOEASY(fe, "sigma", dataElement->sigma);
        TF_DPDPOTENTIALIOTOEASY(fe, "a", dataElement->a);
        TF_DPDPOTENTIALIOTOEASY(fe, "b", dataElement->b);
        TF_DPDPOTENTIALIOTOEASY(fe, "name", std::string(dataElement->name));
        TF_DPDPOTENTIALIOTOEASY(fe, "flags", dataElement->flags);

        fileElement->type = "DPDPotential";

        return S_OK;
    }

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, DPDPotential **dataElement) {

        IOChildMap::const_iterator feItr;

        uint32_t kind, flags;
        TF_DPDPOTENTIALIOFROMEASY(feItr, fileElement.children, metaData, "kind", &kind);
        TF_DPDPOTENTIALIOFROMEASY(feItr, fileElement.children, metaData, "flags", &flags);

        FPTYPE alpha, gamma, sigma, b;
        TF_DPDPOTENTIALIOFROMEASY(feItr, fileElement.children, metaData, "alpha", &alpha);
        TF_DPDPOTENTIALIOFROMEASY(feItr, fileElement.children, metaData, "gamma", &gamma);
        TF_DPDPOTENTIALIOFROMEASY(feItr, fileElement.children, metaData, "sigma", &sigma);
        TF_DPDPOTENTIALIOFROMEASY(feItr, fileElement.children, metaData, "b", &b);

        *dataElement = new DPDPotential(alpha, gamma, sigma, b, flags & POTENTIAL_SHIFTED);

        return S_OK;
    }

};

DPDPotential *TissueForge::DPDPotential_fromStr(const std::string &str) {
    return DPDPotential::fromString(str);
}