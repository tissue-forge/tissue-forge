/*******************************************************************************
 * This file is part of Tissue Forge.
 * Copyright (c) 2022 T.J. Sego and Tien Comlekoglu
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

#include "tf_mesh_create.h"

#include <tfError.h>

#include <map>

using namespace TissueForge;
using namespace TissueForge::models::vertex;

static int surfRelCoords[6][2] = {
    {1, 0}, 
    {0, 1}, 
    {-1, 0}, 
    {-1, -1}, 
    {0, -1}, 
    {1, -1}
};

static unsigned int surfSharedVerts[6][2][2] = {
    {{0, 4}, {1, 3}}, 
    {{1, 5}, {2, 4}}, 
    {{2, 0}, {3, 5}}, 
    {{3, 1}, {4, 0}}, 
    {{4, 2}, {5, 1}}, 
    {{5, 3}, {0, 2}}
};

static std::map<std::string, unsigned int> axesMap = {
    {"x", 0}, {"y", 1}, {"z", 2}
};


static HRESULT findThirdAxis(const std::string &ax_1, const std::string &ax_2, std::string &ax_3) {
    std::map<std::string, unsigned int> axesMapCopy(axesMap);

    if(ax_1 == ax_2) {
        tf_error(E_FAIL, "Invalid axes");
        return E_FAIL;
    }

    auto axesMapCopy_itr = axesMapCopy.find(ax_1);
    if(axesMapCopy_itr == axesMapCopy.end()) {
        tf_error(E_FAIL, "Invalid axis (1)");
        return E_FAIL;
    }
    axesMapCopy.erase(axesMapCopy_itr);

    axesMapCopy_itr = axesMapCopy.find(ax_2);
    if(axesMapCopy_itr == axesMapCopy.end()) {
        tf_error(E_FAIL, "Invalid axis (2)");
        return E_FAIL;
    }
    axesMapCopy.erase(axesMapCopy_itr);

    ax_3 = (*axesMapCopy.begin()).first;
    return S_OK;
}

std::vector<std::vector<Surface*> > TissueForge::models::vertex::createQuadMesh(
    Mesh *mesh, 
    SurfaceType *stype,
    const FVector3 &startPos, 
    const unsigned int &num_1, 
    const unsigned int &num_2, 
    const FloatP_t &len_1,
    const FloatP_t &len_2,
    const char *ax_1, 
    const char *ax_2) 
{
    std::string str_ax_1(ax_1);
    std::string str_ax_2(ax_2);
    std::string str_ax_3;
    if(findThirdAxis(str_ax_1, str_ax_2, str_ax_3) != S_OK) 
        return {};

    unsigned int ax_1_comp = axesMap[str_ax_1];
    unsigned int ax_2_comp = axesMap[str_ax_2];
    unsigned int ax_3_comp = axesMap[str_ax_3];
    FloatP_t z_cm = startPos[ax_3_comp];

    std::vector<std::vector<Surface*> > result = std::vector<std::vector<Surface*> >(num_1, std::vector<Surface*>(num_2, NULL));

    FVector3 vertRelCoords[4];
    vertRelCoords[1][ax_1_comp] = len_1;
    vertRelCoords[2][ax_1_comp] = len_1;
    vertRelCoords[2][ax_2_comp] = len_2;
    vertRelCoords[3][ax_2_comp] = len_2;
    std::vector<Surface*> surfs_new;

    for(size_t i = 0; i < num_1; i++) {

        FloatP_t x_start = startPos[ax_1_comp] + i * len_1;

        for(size_t j = 0; j < num_2; j++) {

            FloatP_t y_start = startPos[ax_2_comp] + j * len_2;

            Surface *sq_left = i == 0 ? NULL : result[i-1][j];
            Surface *sq_down = j == 0 ? NULL : result[i][j-1];

            Vertex *v0 = NULL, *v1 = NULL, *v2 = NULL, *v3 = NULL;
            if(sq_left) {
                v0 = sq_left->getVertices()[1];
                v3 = sq_left->getVertices()[2];
            } 
            if(sq_down) {
                if(!v0) 
                    v0 = sq_down->getVertices()[3];
                v1 = sq_down->getVertices()[2];
            }

            if(!v0) {
                FVector3 vpos;
                vpos[ax_1_comp] = x_start;
                vpos[ax_2_comp] = y_start;
                vpos[ax_3_comp] = z_cm;
                v0 = new Vertex(vpos);
            }
            if(!v1) {
                FVector3 vpos;
                vpos[ax_1_comp] = x_start + len_1;
                vpos[ax_2_comp] = y_start;
                vpos[ax_3_comp] = z_cm;
                v1 = new Vertex(vpos);
            }
            if(!v2) {
                FVector3 vpos;
                vpos[ax_1_comp] = x_start + len_1;
                vpos[ax_2_comp] = y_start + len_2;
                vpos[ax_3_comp] = z_cm;
                v2 = new Vertex(vpos);
            }
            if(!v3) {
                FVector3 vpos;
                vpos[ax_1_comp] = x_start;
                vpos[ax_2_comp] = y_start + len_2;
                vpos[ax_3_comp] = z_cm;
                v3 = new Vertex(vpos);
            }

            Surface *s_new = (*stype)({v0, v1, v2, v3});
            if(!s_new) {
                tf_error(E_FAIL, "Surface could not be created");
                return {};
            }
            else if(mesh->add(s_new) != S_OK) {
                tf_error(E_FAIL, "Surface could not be added to mesh");
                return {};
            }
            result[i][j] = s_new;
            surfs_new.push_back(s_new);
        }
    }

    if(Surface::sew(surfs_new) != S_OK) {
        tf_error(E_FAIL, "Sewing failed");
        return {};
    }
    return result;
}

std::vector<std::vector<std::vector<Body*> > > TissueForge::models::vertex::createPLPDMesh(
    Mesh *mesh, 
    BodyType *btype, 
    SurfaceType *stype,
    const FVector3 &startPos, 
    const unsigned int &num_1, 
    const unsigned int &num_2, 
    const unsigned int &num_3, 
    const FloatP_t &len_1,
    const FloatP_t &len_2,
    const FloatP_t &len_3,
    const char *ax_1, 
    const char *ax_2) 
{
    std::string str_ax_1(ax_1);
    std::string str_ax_2(ax_2);
    std::string str_ax_3;
    if(findThirdAxis(str_ax_1, str_ax_2, str_ax_3) != S_OK) 
        return {};

    unsigned int ax_1_comp = axesMap[str_ax_1];
    unsigned int ax_2_comp = axesMap[str_ax_2];
    unsigned int ax_3_comp = axesMap[str_ax_3];
    FloatP_t z_cm = startPos[ax_3_comp];

    FVector3 vertRelCoords[4];
    vertRelCoords[1][ax_1_comp] = len_1;
    vertRelCoords[2][ax_1_comp] = len_1;
    vertRelCoords[2][ax_2_comp] = len_2;
    vertRelCoords[3][ax_2_comp] = len_2;

    std::vector<std::vector<std::vector<Surface*> > > surfs_12, surfs_13, surfs_23;
    surfs_12.reserve(num_3 + 1);
    surfs_13.reserve(num_2 + 1);
    surfs_23.reserve(num_1 + 1);
    std::vector<Surface*> surfs_new;
    surfs_new.reserve((num_1 + 1) * (num_2 + 1) * (num_3 + 1));
    for(size_t i = 0; i <= num_3; i++) {
        FVector3 startPos_i(startPos);
        startPos_i[ax_3_comp] += i * len_3;
        surfs_12.push_back(
            createQuadMesh(mesh, stype, startPos_i, num_1, num_2, len_1, len_2, ax_1, ax_2)
        );
        for(auto &sv : surfs_12.back()) 
            for(auto s : sv) 
                surfs_new.push_back(s);
    }
    for(size_t i = 0; i <= num_2; i++) {
        FVector3 startPos_i(startPos);
        startPos_i[ax_2_comp] += i * len_2;
        surfs_13.push_back(
            createQuadMesh(mesh, stype, startPos_i, num_1, num_3, len_1, len_3, ax_1, str_ax_3.c_str())
        );
        for(auto &sv : surfs_13.back()) 
            for(auto s : sv) 
                surfs_new.push_back(s);
    }
    for(size_t i = 0; i <= num_1; i++) {
        FVector3 startPos_i(startPos);
        startPos_i[ax_1_comp] += i * len_1;
        surfs_23.push_back(
            createQuadMesh(mesh, stype, startPos_i, num_2, num_3, len_2, len_3, ax_2, str_ax_3.c_str())
        );
        for(auto &sv : surfs_23.back()) 
            for(auto s : sv) 
                surfs_new.push_back(s);
    }

    if(Surface::sew(surfs_new) != S_OK) {
        tf_error(E_FAIL, "Sewing failed");
        return {};
    }

    std::vector<std::vector<std::vector<Body*> > > result = 
        std::vector<std::vector<std::vector<Body*>>>(
            num_1, std::vector<std::vector<Body*>>(
                num_2, std::vector<Body*>(
                    num_3, NULL
    )));

    for(size_t i = 0; i < num_1; i++) {
        std::vector<std::vector<Surface*> > surfs_23_in = surfs_23[i];
        std::vector<std::vector<Surface*> > surfs_23_ip = surfs_23[i+1];
        for(size_t j = 0; j < num_2; j++) {
            std::vector<std::vector<Surface*> > surfs_13_jn = surfs_13[j];
            std::vector<std::vector<Surface*> > surfs_13_jp = surfs_13[j+1];
            for(size_t k = 0; k < num_3; k++) {
                std::vector<std::vector<Surface*> > surfs_12_kn = surfs_12[k];
                std::vector<std::vector<Surface*> > surfs_12_kp = surfs_12[k+1];

                Body *b_new = (*btype)({
                    surfs_23_in[j][k],
                    surfs_23_ip[j][k],
                    surfs_13_jn[i][k],
                    surfs_13_jp[i][k],
                    surfs_12_kn[i][j],
                    surfs_12_kp[i][j]
                });
                if(!b_new) {
                    tf_error(E_FAIL, "Body could not be created");
                    return {};
                } 
                else if(mesh->add(b_new) != S_OK) {
                    tf_error(E_FAIL, "Body could not be added to mesh");
                    return {};
                }
                result[i][j][k] = b_new;
            }
        }

    }

    return result;
}

std::vector<std::vector<Surface*> > TissueForge::models::vertex::createHex2DMesh(
    Mesh *mesh, 
    SurfaceType *stype, 
    const FVector3 &startPos, 
    const unsigned int &num_1, 
    const unsigned int &num_2, 
    const FloatP_t &hexRad,
    const char *ax_1, 
    const char *ax_2) 
{
    std::string str_ax_1(ax_1);
    std::string str_ax_2(ax_2);
    std::string str_ax_3;
    if(findThirdAxis(str_ax_1, str_ax_2, str_ax_3) != S_OK) 
        return {};

    unsigned int ax_1_comp = axesMap[str_ax_1];
    unsigned int ax_2_comp = axesMap[str_ax_2];
    unsigned int ax_3_comp = axesMap[str_ax_3];
    FloatP_t z_cm = startPos[ax_3_comp];

    FloatP_t hexRadMinor = std::sqrt(3) * 0.5 * hexRad;

    FVector3 poly_ax1, poly_ax2;
    poly_ax1[ax_1_comp] = 1.f;
    poly_ax2[ax_2_comp] = 1.f;

    std::vector<std::vector<Surface*> > result(num_1, std::vector<Surface*>(num_2, NULL));
    std::vector<Surface*> surfs_new;
    size_t offset = 0;

    for(size_t i = 0; i < num_1; i++) {
        for(size_t j = 0; j < num_2; j++) {
            FVector3 off;
            off[ax_1_comp] = (2 * i - (FloatP_t)offset / 2) * hexRad;
            off[ax_2_comp] = (2 * j + (FloatP_t)(offset % 2 != 0)) * hexRadMinor;
            FVector3 center = startPos + off;
            Surface *s_new = stype->nPolygon(6, center, hexRad, poly_ax1, poly_ax2);
            if(!s_new) {
                tf_error(E_FAIL, "Surface could not be created");
                return {};
            }
            else if(mesh->add(s_new) != S_OK) {
                tf_error(E_FAIL, "Surface could not be added to mesh");
                return {};
            }
            result[i][j] = s_new;
            surfs_new.push_back(s_new);
        }
        offset++;
    }

    if(Surface::sew(surfs_new) != S_OK) {
        tf_error(E_FAIL, "Sewing failed");
        return {};
    }
    return result;
}

std::vector<std::vector<std::vector<Body*>> > TissueForge::models::vertex::createHex3DMesh(
    Mesh *mesh, 
    BodyType *btype, 
    SurfaceType *stype,
    const FVector3 &startPos, 
    const unsigned int &num_1, 
    const unsigned int &num_2, 
    const unsigned int &num_3, 
    const FloatP_t &hexRad,
    const FloatP_t &hex_height,
    const char *ax_1, 
    const char *ax_2) 
{
    std::string str_ax_1(ax_1);
    std::string str_ax_2(ax_2);
    std::string str_ax_3;
    if(findThirdAxis(str_ax_1, str_ax_2, str_ax_3) != S_OK) 
        return {};

    unsigned int ax_1_comp = axesMap[str_ax_1];
    unsigned int ax_2_comp = axesMap[str_ax_2];
    unsigned int ax_3_comp = axesMap[str_ax_3];
    FloatP_t z_cm = startPos[ax_3_comp];

    std::vector<std::vector<std::vector<Surface*> > > surfs_3;
    surfs_3.reserve(num_3 + 1);
    for(size_t i = 0; i <= num_3; i++) {
        FVector3 startPos_i(startPos);
        startPos_i[ax_3_comp] += i * hex_height;
        surfs_3.push_back(createHex2DMesh(mesh, stype, startPos_i, num_1, num_2, hexRad, ax_1, ax_2));
    }

    std::vector<std::vector<std::vector<Body*> > > result = 
        std::vector<std::vector<std::vector<Body*>>>(
            num_1, std::vector<std::vector<Body*>>(
                num_2, std::vector<Body*>(
                    num_3, NULL
    )));

    std::vector<std::vector<std::vector<std::vector<Surface*> > > > surfSides = 
        std::vector<std::vector<std::vector<std::vector<Surface*> > > >(num_1, 
            std::vector<std::vector<std::vector<Surface*> > >(num_2, 
                std::vector<std::vector<Surface*> >(num_3, 
                    std::vector<Surface*>(6, NULL
    ))));

    for(size_t i = 0; i < num_1; i++) {
        for(size_t j = 0; j < num_2; j++) {

            int surfRelCoords_j[6][2];
            for(size_t ki = 0; ki < 6; ki++) 
                for(size_t kj = 0; kj < 2; kj++) 
                    surfRelCoords_j[ki][kj] = surfRelCoords[ki][kj];
            if(i % 2 != 0) {
                unsigned int idx_adjust[] = {0, 2, 3, 5};
                for(size_t k = 0; k < 4; k++) 
                    surfRelCoords_j[k][1]++;
            }
                
            std::map<int, std::map<int, unsigned int> > surfNbsLabs_j;
            for(size_t k = 0; k < 6; k++) {
                int lab1 = surfRelCoords_j[k][0];
                int lab2 = surfRelCoords_j[k][1];

                if(surfNbsLabs_j.find(lab1) == surfNbsLabs_j.end()) 
                    surfNbsLabs_j.insert({lab1, std::map<int, unsigned int>()});
                surfNbsLabs_j[lab1][lab2] = k;
            }

            std::vector<unsigned int> nbs_surfs;
            for(size_t k = 0; k < 6; k++) {
                int ir = surfRelCoords_j[k][0];
                int jr = surfRelCoords_j[k][1];
                auto itr_ir = surfNbsLabs_j.find(ir);
                if(itr_ir != surfNbsLabs_j.end()) {
                    auto itr_jr = itr_ir->second.find(jr);
                    if(itr_jr != itr_ir->second.end()) 
                        nbs_surfs.push_back(itr_jr->second);
                }
            }

            for(size_t k = 0; k < num_3; k++) {
                Surface *s_bot = surfs_3[k][i][j];
                Surface *s_top = surfs_3[k+1][i][j];
                std::vector<Surface*> surfSides_k;
                surfSides_k.reserve(6);
                surfSides_k.push_back(s_bot);
                surfSides_k.push_back(s_top);

                for(size_t n = 0; n < 6; n++) {
                    int nbs_reli = surfRelCoords_j[n][0];
                    int nbs_relj = surfRelCoords_j[n][1];

                    Surface *nbs_surf = NULL;
                    if(i + nbs_reli >= 0 && i + nbs_reli < surfSides.size()) 
                        if(j + nbs_relj >= 0 && j + nbs_relj < surfSides[i + nbs_reli].size()) 
                            nbs_surf = surfSides[i + nbs_reli][j + nbs_relj][k][surfSharedVerts[n][1][1]];
                    if(!nbs_surf) {
                        std::vector<Vertex*> verts_bot, verts_top;
                        verts_bot = s_bot->getVertices();
                        verts_top = s_top->getVertices();
                        Vertex *v0 = verts_bot[surfSharedVerts[n][0][0]];
                        Vertex *v1 = verts_bot[surfSharedVerts[n][1][0]];
                        Vertex *v2 = verts_top[surfSharedVerts[n][1][0]];
                        Vertex *v3 = verts_top[surfSharedVerts[n][0][0]];

                        nbs_surf = (*stype)({v0, v1, v2, v3});
                        if(!nbs_surf) {
                            tf_error(E_FAIL, "Surface could not be created");
                            return {};
                        }
                        else if(mesh->add(nbs_surf) != S_OK) {
                            tf_error(E_FAIL, "Surface could not be added to mesh");
                            return {};
                        }
                    }

                    surfSides[i][j][k][n] = nbs_surf;
                    surfSides_k.push_back(nbs_surf);
                }

                Body *b_k = (*btype)(surfSides_k);
                if(!b_k) {
                    tf_error(E_FAIL, "Body could not be created");
                    return {};
                }
                else if(mesh->add(b_k) != S_OK) {
                    tf_error(E_FAIL, "Body could not be added to mesh");
                    return {};
                }
                result[i][j][k] = b_k;
            }
        }
    }

    return result;
}
