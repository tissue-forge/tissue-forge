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

#include "tfArcBallInteractor.h"

#include <cstdio>
#include <iostream>


using namespace TissueForge;


enum {VX, VY, VZ, VW};           // axes

static inline Magnum::Matrix4 q2m(const Magnum::Quaternion q)
{
    /*
    float xs, ys, zs, wx, wy, wz, xx, xy, xz, yy, yz, zz;

    Magnum::Vector3 v = q.vector();
    float s = q.scalar();

    float t  = 2.0f / (Magnum::Math::dot(v, v) + s*s);

    xs = v[VX]*t;   ys = v[VY]*t;   zs = v[VZ]*t;
    wx = s*xs;      wy = s*ys;      wz = s*zs;
    xx = v[VX]*xs;  xy = v[VX]*ys;  xz = v[VX]*zs;
    yy = v[VY]*ys;  yz = v[VY]*zs;  zz = v[VZ]*zs;

    Magnum::Matrix4 matrix{
        {1.0f-(yy+zz), xy+wz,        xz-wy,        0.0f},
        {xy-wz,        1.0f-(xx+zz), yz+wx,        0.0f},
        {xz+wy,        yz-wx,        1.0f-(xx+yy), 0.0f},
        {0.0f,         0.0f,         0.0f,         1.0f }};

    return matrix;
     */
    return Magnum::Matrix4::from(q.toMatrix(), {});
}


/**************************************** ArcBall::ArcBall() ****/
/* Default () constructor for ArcBall                         */

rendering::ArcBallInteractor::ArcBallInteractor()
{
    reset();
}

/**************************************** ArcBall::ArcBall() ****/
/* Takes as argument a Magnum::Matrix4 to use instead of the internal rot  */

rendering::ArcBallInteractor::ArcBallInteractor(const Magnum::Matrix4 &mtx)
{
    rot = mtx;
}


/**************************************** ArcBall::ArcBall() ****/
/* A constructor that accepts the screen center and arcball radius*/

rendering::ArcBallInteractor::ArcBallInteractor(const Magnum::Vector2 &_center, float _radius)
{
    reset();
    setParams(_center, _radius);
}


/************************************** ArcBall::set_params() ****/

void rendering::ArcBallInteractor::setParams(const Magnum::Vector2 &_center, float _radius)
{
    center      = _center;
    radius      = _radius;
}

/*************************************** ArcBall::init() **********/

void rendering::ArcBallInteractor::reset()
{
    center = Magnum::Vector2{{ 0.0f, 0.0f }};
    radius         = 1.0;
    q_now          = Magnum::Quaternion(Magnum::Math::IdentityInit);
    rot            = Magnum::Matrix4{Magnum::Math::IdentityInit};
    q_increment    = Magnum::Quaternion(Magnum::Math::IdentityInit);
    rot_increment  = Magnum::Matrix4{Magnum::Math::IdentityInit};
    is_mouse_down  = false;
    is_spinning    = false;
    damp_factor    = 0.0;
    zero_increment = true;
}

Magnum::Matrix4 rendering::ArcBallInteractor::rotation() const
{
    return rot;
}

void rendering::ArcBallInteractor::setWindowSize(int width, int height)
{
    center = {{width / 2.f, height / 2.f}};
    radius = center.length() / 2.;
}

/*********************************** ArcBall::mouse_to_sphere() ****/

Magnum::Vector3 rendering::ArcBallInteractor::mouseToSphere(const Magnum::Vector2 &p)
{
    float mag;
    Magnum::Vector2  v2 = (p - center) / radius;
    Magnum::Vector3  v3( v2[0], v2[1], 0.0 );

    mag = Magnum::Math::dot(v2, v2);

    if ( mag > 1.0 )
        v3 = v3.normalized();
    else
        v3[VZ] = (float) sqrt( 1.0 - mag );

    /* Now we add constraints - X takes precedence over Y */
    if ( constraint_x )
    {
        v3 = constrainVector( v3, Magnum::Vector3( 1.0, 0.0, 0.0 ));
    }
    else if ( constraint_y )
        {
            v3 = constrainVector( v3, Magnum::Vector3( 0.0, 1.0, 0.0 ));
        }

    return v3;
}


/************************************ ArcBall::constrain_vector() ****/

Magnum::Vector3 rendering::ArcBallInteractor::constrainVector(const Magnum::Vector3 &vector, const Magnum::Vector3 &axis)
{
    return (vector-(vector*axis)*axis).normalized();
}

/************************************ ArcBall::mouse_down() **********/

void rendering::ArcBallInteractor::mouseDown(int x, int y)
{
    // move the mouse to the center, change from screen coordinates.
    y = (int) floor(2.0 * center[1] - y);

    down_pt = {{ (float)x, (float) y }};
    is_mouse_down = true;

    q_increment   = Magnum::Quaternion(Magnum::Math::IdentityInit);
    rot_increment = Magnum::Matrix4(Magnum::Math::IdentityInit);
    zero_increment = true;
}


/************************************ ArcBall::mouse_up() **********/

void rendering::ArcBallInteractor::mouseUp()
{
    q_now = q_drag * q_now;
    is_mouse_down = false;
}


/********************************** ArcBall::mouse_motion() **********/

void rendering::ArcBallInteractor::mouseMotion(int x, int y, int shift, int ctrl, int alt)
{
    // move the mouse to the center, change from screen coordinates.
    y = (int) floor(2.0 * center[1] - y);
    
    /* Set the X constraint if CONTROL key is pressed, Y if ALT key */
    setConstraints( ctrl != 0, alt != 0 );

    Magnum::Vector2 new_pt( (float)x, (float) y );
    Magnum::Vector3 v0 = mouseToSphere( down_pt );
    Magnum::Vector3 v1 = mouseToSphere( new_pt );

    Magnum::Vector3 cross = Magnum::Math::cross(v0,v1);

    q_drag = Magnum::Quaternion{{cross}, Magnum::Math::dot(v0,v1) };

    //    *rot_ptr = (q_drag * q_now).to_Magnum::Matrix4();
    Magnum::Matrix4 temp = q2m(q_drag);
    
    rot =  temp * rot;
    
    down_pt = new_pt;

    /* We keep a copy of the current incremental rotation (= q_drag) */
    q_increment   = q_drag;
    rot_increment = q2m(q_increment);

    setConstraints(false, false);

    if ( q_increment.scalar() < .999999 )
    {
        is_spinning = true;
        zero_increment = false;
    }
    else
    {
        is_spinning = false;
        zero_increment = true;
    }
}


/********************************** ArcBall::mouse_motion() **********/

void rendering::ArcBallInteractor::mouseMotion(int x, int y)
{
    mouseMotion(x, y, 0, 0, 0);
}


/***************************** ArcBall::set_constraints() **********/

void rendering::ArcBallInteractor::setConstraints(bool _constraint_x, bool _constraint_y)
{
    constraint_x = _constraint_x;
    constraint_y = _constraint_y;
}

/***************************** ArcBall::idle() *********************/

void rendering::ArcBallInteractor::idle()
{
    if (is_mouse_down)
    {
        is_spinning = false;
        zero_increment = true;
    }


    if (damp_factor < 1.0f) {
        q_increment = Magnum::Quaternion::rotation(Magnum::Rad(1.0f - damp_factor), q_increment.axis());
    }

    rot_increment = q2m(q_increment);

    if (q_increment.scalar() >= .999999f)
    {
        is_spinning = false;
        zero_increment = true;
    }
}


/************************ ArcBall::set_damping() *********************/

void rendering::ArcBallInteractor::setDampening(float d)
{
    damp_factor = d;
}
