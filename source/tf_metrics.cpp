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

#include "tf_metrics.h"
#include <tfEngine.h>
#include <tfSpace.h>
#include <tfSpace_cell.h>
#include <tfRunner.h>
#include <tf_potential_eval.h>

#include <eigen3/Eigen/Eigen>

using namespace TissueForge;


static HRESULT virial_pair(
    FloatP_t cutoff,
    const std::set<short int> &typeIds,
    space_cell *cell_i,
    space_cell *cell_j,
    int sid,
    const FVector3 &shift,
    FMatrix3 &m
);

/**
 * search a pair of cells for particles
 */
static HRESULT enum_particles(
    const FVector3 &origin,
    FloatP_t radius,
    space_cell *cell,
    const std::set<short int> *typeIds,
    int32_t exceptPartId,
    const FVector3 &shift,
    std::vector<int32_t> &ids
);


FVector3 metrics::relativePosition(const FVector3 &pos, const FVector3 &origin, const bool &comp_bc) {
    if(!comp_bc) return pos - origin;

    const BoundaryConditions &bc = _Engine.boundary_conditions;
    FVector3 _pos = pos;
    FVector3 result = _pos.relativeTo(origin, engine_dimensions(), bc.periodic & space_periodic_x, bc.periodic & space_periodic_y, bc.periodic & space_periodic_z);
    return result;
}

ParticleList metrics::neighborhoodParticles(const FVector3 &position, const FloatP_t &dist, const bool &comp_bc) {
    
    // cell id of target cell
    int cid, ijk[3];
    
    int l[3], ii, jj, kk;
    
    FloatP_t lh[3];
    
    int id2, sid;
    
    space *s = &_Engine.s;
    
    /** Number of cells within cutoff in each dimension. */
    int span[3];
    
    std::vector<int32_t> ids;

    FPTYPE pos[] = {position[0], position[1], position[2]};    
    if((cid = space_get_cellids_for_pos(&_Engine.s, pos, ijk)) < 0) {
        tf_error(E_FAIL, "Could not identify cell");
        return ParticleList();
    }
    
    // the current cell
    space_cell *c = &s->cells[cid];
    
    // origin in the target cell's coordinate system
    FVector3 local_origin = {
        position[0] - c->origin[0],
        position[1] - c->origin[1],
        position[2] - c->origin[2]
    };
    
    // the other cell.
    space_cell *cj, *ci;
    
    // shift vector between cells.
    FVector3 shift;
    
    /* Get the span of the cells we will search for pairs. */
    for (int k = 0 ; k < 3 ; k++ ) {
        span[k] = (int)std::ceil( dist * s->ih[k] );
    }
    
    /* for every neighbouring cell in the x-axis... */
    for ( l[0] = -span[0] ; l[0] <= span[0] ; l[0]++ ) {
        
        /* get coords of neighbour */
        ii = ijk[0] + l[0];
        
        /* wrap or abort if not periodic */
        if ( ii < 0 ) {
            if (comp_bc && s->period & space_periodic_x)
                ii += s->cdim[0];
            else
                continue;
        }
        else if ( ii >= s->cdim[0] ) {
            if (comp_bc && s->period & space_periodic_x)
                ii -= s->cdim[0];
            else
                continue;
        }
        
        /* for every neighbouring cell in the y-axis... */
        for ( l[1] = -span[1] ; l[1] <= span[1] ; l[1]++ ) {
            
            /* get coords of neighbour */
            jj = ijk[1] + l[1];
            
            /* wrap or abort if not periodic */
            if ( jj < 0 ) {
                if (comp_bc && s->period & space_periodic_y)
                    jj += s->cdim[1];
                else
                    continue;
            }
            else if ( jj >= s->cdim[1] ) {
                if (comp_bc && s->period & space_periodic_y)
                    jj -= s->cdim[1];
                else
                    continue;
            }
            
            /* for every neighbouring cell in the z-axis... */
            for ( l[2] = -span[2] ; l[2] <= span[2] ; l[2]++ ) {
                
                /* get coords of neighbour */
                kk = ijk[2] + l[2];
                
                /* wrap or abort if not periodic */
                if ( kk < 0 ) {
                    if (comp_bc && s->period & space_periodic_z)
                        kk += s->cdim[2];
                    else
                        continue;
                }
                else if ( kk >= s->cdim[2] ) {
                    if (comp_bc && s->period & space_periodic_z)
                        kk -= s->cdim[2];
                    else
                        continue;
                }
                
                /* Are these cells within the cutoff of each other? */
                lh[0] = s->h[0]*fmax( abs(l[0])-1 , 0 );
                lh[1] = s->h[1]*fmax( abs(l[1])-1 , 0 );
                lh[2] = s->h[2]*fmax( abs(l[2])-1 , 0 );
                if (std::sqrt(lh[0]*lh[0] + lh[1]*lh[1] + lh[2]*lh[2]) > dist )
                    continue;
                
                /* get the neighbour's id */
                id2 = space_cellid(s,ii,jj,kk);
                
                /* Get the pair sortID. */
                ci = &s->cells[cid];
                cj = &s->cells[id2];
                sid = space_getsid(s , &ci , &cj , shift.data());
                
                // check if flipped,
                // space_getsid flips cells under certain circumstances.
                if(cj == c) {
                    cj = ci;
                    shift = shift * -1;
                }
                
                HRESULT result = enum_particles (local_origin, dist, cj, NULL, -1, shift, ids);
            } /* for every neighbouring cell in the z-axis... */
        } /* for every neighbouring cell in the y-axis... */
    } /* for every neighbouring cell in the x-axis... */
    
    return ParticleList(ids.size(), ids.data());
}

HRESULT metrics::calculateVirial(FloatP_t *_origin, FloatP_t radius, const std::set<short int> &typeIds, FloatP_t *tensor) {
    FVector3 origin = FVector3::from(_origin);
    
    FMatrix3 m{0.0};
    
    // cell id of target cell
    int cid, ijk[3];
    
    int l[3], ii, jj, kk;
    
    FloatP_t lh[3];
    
    int id2, sid;
    
    space *s = &_Engine.s;
    
    /** Number of cells within cutoff in each dimension. */
    int span[3];
    
    if((cid = space_get_cellids_for_pos(&_Engine.s, origin.data(), ijk)) < 0) {
        // TODO: bad...
        return E_FAIL;
    }
    
    // the current cell
    space_cell *c = &s->cells[cid];
    
    // the other cell.
    space_cell *cj;
    
    // shift vector between cells.
    FVector3 shift;
    
    /* Get the span of the cells we will search for pairs. */
    for (int k = 0 ; k < 3 ; k++ ) {
        span[k] = (int)std::ceil( radius * s->ih[k] );
    }
    
    /* for every neighbouring cell in the x-axis... */
    for ( l[0] = -span[0] ; l[0] <= span[0] ; l[0]++ ) {
        
        /* get coords of neighbour */
        ii = ijk[0] + l[0];
        
        /* wrap or abort if not periodic */
        if (ii < 0 || ii >= s->cdim[0]) {
            continue;
        }

        /* for every neighbouring cell in the y-axis... */
        for ( l[1] = -span[1] ; l[1] <= span[1] ; l[1]++ ) {
            
            /* get coords of neighbour */
            jj = ijk[1] + l[1];
            
            /* wrap or abort if not periodic */
            if ( jj < 0 || jj >= s->cdim[1] ) {
                continue;
            }
            
            /* for every neighbouring cell in the z-axis... */
            for ( l[2] = -span[2] ; l[2] <= span[2] ; l[2]++ ) {
                
                /* get coords of neighbour */
                kk = ijk[2] + l[2];
                
                /* wrap or abort if not periodic */
                if ( kk < 0  ||  kk >= s->cdim[2] ) {
                    continue;
                }
                
                /* Are these cells within the cutoff of each other? */
                lh[0] = s->h[0]*fmax( abs(l[0])-1 , 0 );
                lh[1] = s->h[1]*fmax( abs(l[1])-1 , 0 );
                lh[2] = s->h[2]*fmax( abs(l[2])-1 , 0 );
                if (std::sqrt(lh[0]*lh[0] + lh[1]*lh[1] + lh[2]*lh[2]) > radius )
                    continue;
                
                /* get the neighbour's id */
                id2 = space_cellid(s,ii,jj,kk);
                
                /* Get the pair sortID. */
                c = &s->cells[cid];
                cj = &s->cells[id2];
                sid = space_getsid(s , &c , &cj , shift.data());
                
                HRESULT result = virial_pair (radius, typeIds, c, cj, sid, shift, m);
            } /* for every neighbouring cell in the z-axis... */
        } /* for every neighbouring cell in the y-axis... */
    } /* for every neighbouring cell in the x-axis... */
    
    for(int i = 0; i < 9; ++i) {
        tensor[i] = m.data()[i];
    }

    return S_OK;
}


/**
 * converts cartesian to spherical, writes spherical
 * coords in to result array.
 */
FVector3 metrics::cartesianToSpherical(const FVector3& pos, const FVector3& origin) {
    FVector3 vec = pos - origin;
    
    FloatP_t radius = vec.length();
    FloatP_t theta = std::atan2(vec.y(), vec.x());
    FloatP_t phi = std::acos(vec.z() / radius);
    return FVector3{radius, theta, phi};
}


static HRESULT virial_pair(
    FloatP_t cutoff,
    const std::set<short int> &typeIds,
    space_cell *cell_i,
    space_cell *cell_j,
    int sid,
    const FVector3 &shift,
    FMatrix3 &m) 
{
    
    int i, j, k, count_i, count_j;
    FloatP_t cutoff2, r2;
    struct Particle *part_i, *part_j, *parts_i, *parts_j;
    FVector4 dx;
    FVector4 pix;
    Potential *pot;
    FloatP_t w = 0, e = 0, f = 0;
    FVector3 force;

    /* break early if one of the cells is empty */
    count_i = cell_i->count;
    count_j = cell_j->count;
    if ( count_i == 0 || count_j == 0 || ( cell_i == cell_j && count_i < 2 ) )
        return S_OK;
    
    /* get the space and cutoff */
    cutoff2 = cutoff * cutoff;
    pix[3] = 0;
    
    parts_i = cell_i->parts;
    parts_j = cell_j->parts;
    
    /* is this a genuine pair or a cell against itself */
    if ( cell_i == cell_j ) {
        
        /* loop over all particles */
        for ( i = 1 ; i < count_i ; i++ ) {
            
            /* get the particle */
            part_i = &(parts_i[i]);
            pix[0] = part_i->x[0];
            pix[1] = part_i->x[1];
            pix[2] = part_i->x[2];
            
            /* loop over all other particles */
            for ( j = 0 ; j < i ; j++ ) {
                
                /* get the other particle */
                part_j = &(parts_i[j]);
                
                /* get the distance between both particles */
                r2 = fptype_r2(pix.data(), part_j->x , dx.data() );
                
                /* is this within cutoff? */
                if ( r2 > cutoff2 )
                    continue;
                /* runner_rcount += 1; */
                
                /* fetch the potential, if any */
                pot = get_potential(part_i, part_j);
                if ( pot == NULL )
                    continue;
                
                /* check if this is a valid particle to search for */
                if(typeIds.find(part_i->typeId) == typeIds.end() ||
                   typeIds.find(part_j->typeId) == typeIds.end()) {
                    continue;
                }
                
                force[0] = 0; force[1] = 0; force[1] = 0;
                
                
                /* evaluate the interaction */
                /* update the forces if part in range */
                if (potential_eval_super_ex(cell_i, pot, part_i, part_j, dx.data(), r2, &e)) {
                    for ( k = 0 ; k < 3 ; k++ ) {
                        // divide by two because potential_eval gives double the force
                        // to split beteen a pair of particles.
                        w = (f * dx[k]) / 2;
                        force[k] += w;
                    }
                }
                
                //std::cout << "particle(" << part_i->id << ", " << part_j->id << "), dx:["
                //<< dx[0]    << ", " << dx[1]    << ", " << dx[2]    << "], f:["
                //<< force[0] << ", " << force[1] << ", " << force[2] << "]" << std::endl;
                
                m[0][0] += force[0] * dx[0];
                m[0][1] += force[0] * dx[1];
                m[0][2] += force[0] * dx[2];
                m[1][0] += force[1] * dx[0];
                m[1][1] += force[1] * dx[1];
                m[1][2] += force[1] * dx[2];
                m[2][0] += force[2] * dx[0];
                m[2][1] += force[2] * dx[1];
                m[2][2] += force[2] * dx[2];
            } /* loop over all other particles */
        } /* loop over all particles */
    }
    
    /* no, it's a genuine pair */
    else {
        
        /* loop over all particles */
        for ( i = 0 ; i < count_i ; i++ ) {
            
            // get the particle
            // first particle in in cell_i frame, subtract off shift
            // vector to compute pix in cell_j frame
             
            part_i = &(parts_i[i]);
            pix[0] = part_i->x[0] - shift[0];
            pix[1] = part_i->x[1] - shift[1];
            pix[2] = part_i->x[2] - shift[2];
            
            /* loop over all other particles */
            for ( j = 0 ; j < count_j ; j++ ) {
                
                /* get the other particle */
                part_j = &(parts_j[j]);
                
                /* fetch the potential, if any */
                /* get the distance between both particles */
                r2 = fptype_r2(pix.data(), part_j->x, dx.data());
                
                /* is this within cutoff? */
                if ( r2 > cutoff2 )
                    continue;
                
                /* fetch the potential, if any */
                pot = get_potential(part_i, part_j);
                if ( pot == NULL )
                    continue;
                
                force[0] = 0; force[1] = 0; force[1] = 0;
                
                /* evaluate the interaction */
                /* update the forces if part in range */
                if (potential_eval_super_ex(cell_i, pot, part_i, part_j, dx.data(), r2, &e)) {
                    for ( k = 0 ; k < 3 ; k++ ) {
                        w = (f * dx[k]) / 2;
                        force[k] += w;
                    }
                }
                
                m[0][0] += force[0] * dx[0];
                m[0][1] += force[0] * dx[1];
                m[0][2] += force[0] * dx[2];
                m[1][0] += force[1] * dx[0];
                m[1][1] += force[1] * dx[1];
                m[1][2] += force[1] * dx[2];
                m[2][0] += force[2] * dx[0];
                m[2][1] += force[2] * dx[1];
                m[2][2] += force[2] * dx[2];
            } /* loop over all other particles */
        } /* loop over all particles */
    }
    
    /* all is well that ends ok */
    return S_OK;
}

HRESULT metrics::particlesRadiusOfGyration(int32_t *parts, uint16_t nr_parts, FloatP_t *result)
{
    FVector3 r, dx;
    
    FloatP_t r2 = 0;

    // center of geometry
    for(int i = 0; i < nr_parts; ++i) {
        Particle *p = _Engine.s.partlist[parts[i]];
        // global position
        FloatP_t *o = _Engine.s.celllist[p->id]->origin;
        r[0] += p->x[0] + o[0];
        r[1] += p->x[1] + o[1];
        r[2] += p->x[2] + o[2];
    }
    r = r / nr_parts;
    
    // radial distance squared
    for(int i = 0; i < nr_parts; ++i) {
        Particle *p = _Engine.s.partlist[parts[i]];
        // global position
        FloatP_t *o = _Engine.s.celllist[p->id]->origin;
        
        dx[0] = r[0] - (p->x[0] + o[0]);
        dx[1] = r[1] - (p->x[1] + o[1]);
        dx[2] = r[2] - (p->x[2] + o[2]);
        
        r2 += dx.dot();
    }
    
    *result = std::sqrt(r2 / nr_parts);
    
    return S_OK;
}

HRESULT metrics::particlesCenterOfMass(int32_t *parts, uint16_t nr_parts, FloatP_t *result)
{
    FVector3 r;
    FloatP_t m = 0;
    
    // center of geometry
    for(int i = 0; i < nr_parts; ++i) {
        Particle *p = _Engine.s.partlist[parts[i]];
        // global position
        FloatP_t *o = _Engine.s.celllist[p->id]->origin;
        m += p->mass;
        r[0] += p->mass * (p->x[0] + o[0]);
        r[1] += p->mass * (p->x[1] + o[1]);
        r[2] += p->mass * (p->x[2] + o[2]);
    }
    r = r / m;
    
    result[0] = r[0];
    result[1] = r[1];
    result[2] = r[2];
    
    return S_OK;
}

HRESULT metrics::particlesCenterOfGeometry(int32_t *parts, uint16_t nr_parts, FloatP_t *result)
{
    FVector3 r;
    
    // center of geometry
    for(int i = 0; i < nr_parts; ++i) {
        Particle *p = _Engine.s.partlist[parts[i]];
        // global position
        FloatP_t *o = _Engine.s.celllist[p->id]->origin;
        r[0] += p->x[0] + o[0];
        r[1] += p->x[1] + o[1];
        r[2] += p->x[2] + o[2];
    }
    r = r / nr_parts;
    
    result[0] = r[0];
    result[1] = r[1];
    result[2] = r[2];
    
    return S_OK;
}

HRESULT metrics::particlesMomentOfInertia(int32_t *parts, uint16_t nr_parts, FloatP_t *tensor)
{
    FMatrix3 m{0.0};
    int i;
    struct Particle *part_i;
    FVector3 dx;
    FVector3 pix;
    FVector3 cm;
    HRESULT result = metrics::particlesCenterOfMass(parts, nr_parts,cm.data());
    
    if(FAILED(result)) {
        return result;
    }
    
    /* get the space and cutoff */
    pix[3] = 0;
    
    /* loop over all particles */
    for ( i = 0 ; i < nr_parts ; i++ ) {
        
        /* get the particle */
        part_i = _Engine.s.partlist[parts[i]];
        
        // global position of particle i
        FloatP_t *oi = _Engine.s.celllist[part_i->id]->origin;
        pix[0] = part_i->x[0] + oi[0];
        pix[1] = part_i->x[1] + oi[1];
        pix[2] = part_i->x[2] + oi[2];
        
        // position in center of mass frame
        dx = pix - cm;
        
        m[0][0] += (dx[1]*dx[1] + dx[2]*dx[2]) * part_i->mass;
        m[1][1] += (dx[0]*dx[0] + dx[2]*dx[2]) * part_i->mass;
        m[2][2] += (dx[1]*dx[1] + dx[0]*dx[0]) * part_i->mass;
        m[0][1] += dx[0] * dx[1] * part_i->mass;
        m[1][2] += dx[1] * dx[2] * part_i->mass;
        m[0][2] += dx[0] * dx[2] * part_i->mass;
       
    } /* loop over all particles */
    
    m[1][0] = m[0][1];
    m[2][1] = m[1][2];
    m[2][0] = m[0][2];
    
    for(int i = 0; i < 9; ++i) {
        tensor[i] = m.data()[i];
    }
    
    return S_OK;
}

HRESULT metrics::particlesVirial(int32_t *parts, uint16_t nr_parts, uint32_t flags, FloatP_t *tensor) {
    FMatrix3 m{0.0};
    int i, j, k;
    struct Particle *part_i, *part_j;
    FVector4 dx;
    FVector4 pix, pjx;
    Potential *pot;
    FloatP_t w = 0, e = 0, f = 0;
    FVector3 force;
    
    /* get the space and cutoff */
    pix[3] = 0;
    
    FloatP_t r2;
    
    // TODO: more effecient to caclulate everythign in reference frame
    // of outer particle.
        
    /* loop over all particles */
    for ( i = 1 ; i < nr_parts ; i++ ) {
        
        /* get the particle */
        part_i = _Engine.s.partlist[parts[i]];
        
        // global position
        FloatP_t *oi = _Engine.s.celllist[part_i->id]->origin;
        pix[0] = part_i->x[0] + oi[0];
        pix[1] = part_i->x[1] + oi[1];
        pix[2] = part_i->x[2] + oi[2];
        
        /* loop over all other particles */
        for ( j = 0 ; j < i ; j++ ) {
            
            /* get the other particle */
            part_j = _Engine.s.partlist[parts[j]];
            
            // global position
            FloatP_t *oj = _Engine.s.celllist[part_j->id]->origin;
            pjx[0] = part_j->x[0] + oj[0];
            pjx[1] = part_j->x[1] + oj[1];
            pjx[2] = part_j->x[2] + oj[2];
            
            /* get the distance between both particles */
            r2 = fptype_r2(pix.data(), pjx.data() , dx.data());
            
            /* fetch the potential, if any */
            pot = get_potential(part_i, part_j);
            if ( pot == NULL )
                continue;
            
            force[0] = 0; force[1] = 0; force[1] = 0;
            
            /* evaluate the interaction */
            /* update the forces if part in range */
            
            space_cell *cell_i = _Engine.s.celllist[part_i->id];
            if (potential_eval_super_ex(cell_i, pot, part_i, part_j, dx.data(), r2, &e)) {
                for ( k = 0 ; k < 3 ; k++ ) {
                    // divide by two because potential_eval gives double the force
                    // to split beteen a pair of particles.
                    w = (f * dx[k]) / 2;
                    force[k] += w;
                }
            }
            
            m[0][0] += force[0] * dx[0];
            m[0][1] += force[0] * dx[1];
            m[0][2] += force[0] * dx[2];
            m[1][0] += force[1] * dx[0];
            m[1][1] += force[1] * dx[1];
            m[1][2] += force[1] * dx[2];
            m[2][0] += force[2] * dx[0];
            m[2][1] += force[2] * dx[1];
            m[2][2] += force[2] * dx[2];
        } /* loop over all other particles */
    } /* loop over all particles */
    
    for(int i = 0; i < 9; ++i) {
        tensor[i] = m.data()[i];
    }
    
    return S_OK;
}


HRESULT metrics::particleNeighbors(
    Particle *part,
    FloatP_t radius,
    const std::set<short int> *typeIds,
    uint16_t *nr_parts,
    int32_t **pparts) 
{ 
    // origin in global space
    FVector3 origin = part->global_position();
    
    // cell id of target cell
    int cid, ijk[3];
    
    int l[3], ii, jj, kk;
    
    FloatP_t lh[3];
    
    int id2, sid;
    
    space *s = &_Engine.s;
    
    /** Number of cells within cutoff in each dimension. */
    int span[3];
    
    std::vector<int32_t> ids;
    
    if((cid = space_get_cellids_for_pos(&_Engine.s, origin.data(), ijk)) < 0) {
        // TODO: bad...
        return E_FAIL;
    }
    
    // the current cell
    space_cell *c = &s->cells[cid];
    
    // origin in the target cell's coordinate system
    FVector3 local_origin = {
        origin[0] - c->origin[0],
        origin[1] - c->origin[1],
        origin[2] - c->origin[2]
    };
    
    // the other cell.
    space_cell *cj, *ci;
    
    // shift vector between cells.
    FVector3 shift;
    
    /* Get the span of the cells we will search for pairs. */
    for (int k = 0 ; k < 3 ; k++ ) {
        span[k] = (int)std::ceil( radius * s->ih[k] );
    }
    
    /* for every neighbouring cell in the x-axis... */
    for ( l[0] = -span[0] ; l[0] <= span[0] ; l[0]++ ) {
        
        /* get coords of neighbour */
        ii = ijk[0] + l[0];
        
        /* wrap or abort if not periodic */
        if ( ii < 0 ) {
            if (s->period & space_periodic_x)
                ii += s->cdim[0];
            else
                continue;
        }
        else if ( ii >= s->cdim[0] ) {
            if (s->period & space_periodic_x)
                ii -= s->cdim[0];
            else
                continue;
        }
        
        /* for every neighbouring cell in the y-axis... */
        for ( l[1] = -span[1] ; l[1] <= span[1] ; l[1]++ ) {
            
            /* get coords of neighbour */
            jj = ijk[1] + l[1];
            
            /* wrap or abort if not periodic */
            if ( jj < 0 ) {
                if (s->period & space_periodic_y)
                    jj += s->cdim[1];
                else
                    continue;
            }
            else if ( jj >= s->cdim[1] ) {
                if (s->period & space_periodic_y)
                    jj -= s->cdim[1];
                else
                    continue;
            }
            
            /* for every neighbouring cell in the z-axis... */
            for ( l[2] = -span[2] ; l[2] <= span[2] ; l[2]++ ) {
                
                /* get coords of neighbour */
                kk = ijk[2] + l[2];
                
                /* wrap or abort if not periodic */
                if ( kk < 0 ) {
                    if (s->period & space_periodic_z)
                        kk += s->cdim[2];
                    else
                        continue;
                }
                else if ( kk >= s->cdim[2] ) {
                    if (s->period & space_periodic_z)
                        kk -= s->cdim[2];
                    else
                        continue;
                }
                
                /* Are these cells within the cutoff of each other? */
                lh[0] = s->h[0]*fmax( abs(l[0])-1 , 0 );
                lh[1] = s->h[1]*fmax( abs(l[1])-1 , 0 );
                lh[2] = s->h[2]*fmax( abs(l[2])-1 , 0 );
                if (std::sqrt(lh[0]*lh[0] + lh[1]*lh[1] + lh[2]*lh[2]) > radius )
                    continue;
                
                /* get the neighbour's id */
                id2 = space_cellid(s,ii,jj,kk);
                
                /* Get the pair sortID. */
                ci = &s->cells[cid];
                cj = &s->cells[id2];
                sid = space_getsid(s , &ci , &cj , shift.data());
                
                // check if flipped,
                // space_getsid flips cells under certain circumstances.
                if(cj == c) {
                    cj = ci;
                    shift = shift * -1;
                }
                
                HRESULT result = enum_particles (local_origin, radius, cj, typeIds, part->id, shift, ids);
            } /* for every neighbouring cell in the z-axis... */
        } /* for every neighbouring cell in the y-axis... */
    } /* for every neighbouring cell in the x-axis... */
    
    *nr_parts = ids.size();
    int32_t *parts = (int32_t*)malloc(ids.size() * sizeof(int32_t));
    memcpy(parts, ids.data(), ids.size() * sizeof(int32_t));
    *pparts = parts;
    
    return S_OK;
}


HRESULT enum_particles(
    const FVector3 &_origin,
    FloatP_t radius,
    space_cell *cell,
    const std::set<short int> *typeIds,
    int32_t exceptPartId,
    const FVector3 &shift,
    std::vector<int32_t> &ids) 
{
    
    int i, count;
    FloatP_t cutoff2, r2;
    struct Particle *part, *parts;
    FVector4 dx;
    FVector4 pix;
    FVector4 origin;
    
    /* break early if one of the cells is empty */
    count = cell->count;
    
    if ( count == 0 )
        return S_OK;
    
    /* get the space and cutoff */
    cutoff2 = radius * radius;
    pix[3] = 0;
    
    parts = cell->parts;
    
    // shift the origin into the current cell's reference
    // frame with the shift vector.
    origin[0] = _origin[0] - shift[0];
    origin[1] = _origin[1] - shift[1];
    origin[2] = _origin[2] - shift[2];
    
    /* loop over all other particles */
    for ( i = 0 ; i < count ; i++ ) {
        
        /* get the other particle */
        part = &(parts[i]);
        
        if(part->id == exceptPartId) {
            continue;
        }
        
        /* get the distance between both particles */
        r2 = fptype_r2(origin.data() , part->x , dx.data() );
        
        /* is this within cutoff? */
        if ( r2 > cutoff2 ) {
            continue;
        }
        
        /* check if this is a valid particle to search for */
        if(typeIds && typeIds->find(part->typeId) == typeIds->end()) {
            continue;
        }
        
        ids.push_back(part->id);
        
    } /* loop over all other particles */
    
    
    /* all is well that ends ok */
    return S_OK;
}

void do_it(const FVector3 &origin, const Particle *part, FMatrix3 &m) {
    
}


HRESULT enum_thing(
    const FVector3 &_origin,
    FloatP_t radius,
    space_cell *cell,
    const std::set<short int> *typeIds,
    int32_t exceptPartId,
    const FVector3 &shift,
    FMatrix3 &m) 
{
    
    int i, count;
    FloatP_t cutoff2, r2;
    struct Particle *part, *parts;
    FVector4 dx;
    FVector4 pix;
    FVector4 origin;
    
    /* break early if one of the cells is empty */
    count = cell->count;
    
    if ( count == 0 )
        return S_OK;
    
    /* get the space and cutoff */
    cutoff2 = radius * radius;
    pix[3] = 0;
    
    parts = cell->parts;
    
    // shift the origin into the current cell's reference
    // frame with the shift vector.
    origin[0] = _origin[0] - shift[0];
    origin[1] = _origin[1] - shift[1];
    origin[2] = _origin[2] - shift[2];
    
    /* loop over all other particles */
    for ( i = 0 ; i < count ; i++ ) {
        
        /* get the other particle */
        part = &(parts[i]);
        
        if(part->id == exceptPartId) {
            continue;
        }
        
        /* get the distance between both particles */
        r2 = fptype_r2(origin.data() , part->x , dx.data() );
        
        /* is this within cutoff? */
        if ( r2 > cutoff2 ) {
            continue;
        }
        
        /* check if this is a valid particle to search for */
        if(typeIds && typeIds->find(part->typeId) == typeIds->end()) {
            continue;
        }
        
    } /* loop over all other particles */
    
    
    /* all is well that ends ok */
    return S_OK;
}

/**
 * Creates an array of ParticleList objects.
 */
std::vector<std::vector<std::vector<ParticleList> > > metrics::particleGrid(const iVector3 &shape) {
    
    VERIFY_PARTICLES();
    
    if(shape[0] <= 0 || shape[1] <= 0 || shape[2] <= 0) {
        throw std::domain_error("shape must have positive, non-zero values for all dimensions");
    }
    
    std::vector<std::vector<std::vector<ParticleList> > > result(
        shape[0], std::vector<std::vector<ParticleList> >(
            shape[1], std::vector<ParticleList>(
                shape[2], ParticleList())));
    
    FVector3 dim = {_Engine.s.dim[0], _Engine.s.dim[1], _Engine.s.dim[2]};
    
    FVector3 scale = {shape[0] / dim[0], shape[1] / dim[1], shape[2] / dim[2]};
    
    for (int cid = 0 ; cid < _Engine.s.nr_cells ; cid++ ) {
        space_cell *cell = &_Engine.s.cells[cid];
        for (int pid = 0 ; pid < cell->count ; pid++ ) {
            Particle *part  = &cell->parts[pid];
            
            FVector3 pos = part->global_position();
            // relative position of part in universe, scaled from 0-1, then
            // scaled to index in array
            int i = std::floor(pos[0] * scale[0]);
            int j = std::floor(pos[1] * scale[1]);
            int k = std::floor(pos[2] * scale[2]);
            
            assert(i >= 0 && i <= shape[0]);
            assert(j >= 0 && j <= shape[1]);
            assert(k >= 0 && k <= shape[2]);
            
            result[i][j][k].insert(part->id);
        }
    }
    
    for (int pid = 0 ; pid < _Engine.s.largeparts.count ; pid++ ) {
        Particle *part  = &_Engine.s.largeparts.parts[pid];
        
        FVector3 pos = part->global_position();
        // relative position of part in universe, scaled from 0-1, then
        // scaled to index in array
        int i = std::floor(pos[0] * scale[0]);
        int j = std::floor(pos[1] * scale[1]);
        int k = std::floor(pos[2] * scale[2]);
        
        assert(i >= 0 && i <= shape[0]);
        assert(j >= 0 && j <= shape[1]);
        assert(k >= 0 && k <= shape[2]);
        
        result[i][j][k].insert(part->id);
    }
    
    return result;
}

HRESULT metrics::particleGrid(const iVector3 &shape, ParticleList *result) {
    auto pl = metrics::particleGrid(shape);
    unsigned int idx = 0;
    for(unsigned int i2 = 0; i2 < shape[2]; ++i2)
        for(unsigned int i1 = 0; i1 < shape[1]; ++i1)
            for(unsigned int i0 = 0; i0 < shape[0]; ++i0, ++idx)
                result[idx] = pl[i0][i1][i2];

    return S_OK;
}


template <typename TFV, typename TFM> 
HRESULT _eigenVals(const TFM &mat, TFV &evals, const bool &symmetric) {
    Eigen::Matrix<FloatP_t, TFV::Size, 1> _evals;

    if(symmetric) {
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<FloatP_t, TFM::Size, TFM::Size> > es;
        es.compute(Eigen::Matrix<FloatP_t, TFM::Size, TFM::Size>(mat.data()));
        _evals = es.eigenvalues();
    } 
    else {
        Eigen::EigenSolver<Eigen::Matrix<FloatP_t, TFM::Size, TFM::Size> > es;
        es.compute(Eigen::Matrix<FloatP_t, TFM::Size, TFM::Size>(mat.data()), false);
        _evals = es.eigenvalues().real();
    }

    evals = TFV::from(_evals.data());

    return S_OK;
}


template <typename TFV, typename TFM> 
HRESULT _eigenVecsVals(const TFM &mat, TFV &evals, TFM &evecs, const bool &symmetric) {
    Eigen::Matrix<FloatP_t, TFV::Size, 1> _evals;
    Eigen::Matrix<FloatP_t, TFM::Size, TFM::Size> _evecs;

    if(symmetric) {
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<FloatP_t, TFM::Size, TFM::Size> > es(Eigen::Matrix<FloatP_t, TFM::Size, TFM::Size>(mat.data()));
        _evals = es.eigenvalues();
        _evecs = es.eigenvectors();
    } 
    else {
        Eigen::EigenSolver<Eigen::Matrix<FloatP_t, TFM::Size, TFM::Size> > es(Eigen::Matrix<FloatP_t, TFM::Size, TFM::Size>(mat.data()));
        _evals = es.eigenvalues().real();
        _evecs = es.eigenvectors().real();
    }

    evals = TFV::from(_evals.data());
    evecs = TFM::from(_evecs.data());

    return S_OK;
}

FVector3 metrics::eigenVals(const FMatrix3 &mat, const bool &symmetric) {
    FVector3 evals;
    if(_eigenVals(mat, evals, symmetric) != S_OK) 
        tf_error(E_FAIL, "Error computing eigenvalues");
    return evals;
}

FVector4 metrics::eigenVals(const FMatrix4 &mat, const bool &symmetric) {
    FVector4 evals;
    if(_eigenVals(mat, evals, symmetric) != S_OK) 
        tf_error(E_FAIL, "Error computing eigenvalues");
    return evals;
}

std::pair<FVector3, FMatrix3> metrics::eigenVecsVals(const FMatrix3 &mat, const bool &symmetric) {
    FVector3 evals;
    FMatrix3 evecs;
    if(_eigenVecsVals(mat, evals, evecs, symmetric) != S_OK) 
        tf_error(E_FAIL, "Error computing eigenvectors and eigenvalues");
    return {evals, evecs};
}

std::pair<FVector4, FMatrix4> metrics::eigenVecsVals(const FMatrix4 &mat, const bool &symmetric) {
    FVector4 evals;
    FMatrix4 evecs;
    if(_eigenVecsVals(mat, evals, evecs, symmetric) != S_OK) 
        tf_error(E_FAIL, "Error computing eigenvectors and eigenvalues");
    return {evals, evecs};
}
