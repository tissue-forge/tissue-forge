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

#ifndef _SOURCE_RENDERING_TFARCBALLINTERACTOR_H_
#define _SOURCE_RENDERING_TFARCBALLINTERACTOR_H_

#include <stdint.h>

#include <Magnum/Magnum.h>
#include <Magnum/Math/Vector3.h>
#include <Magnum/Math/Vector3.h>
#include <Magnum/Math/Matrix4.h>
#include <Magnum/Math/Quaternion.h>

#include <iostream>

inline std::ostream& operator<<(std::ostream& os, const Magnum::Matrix4& m) {
    os << "mat4 : {{" << m[0][0] << ", " << m[1][0] << ", " << m[2][0] << ", " << m[3][0] << "}" << std::endl;
    os << "        {" << m[0][1] << ", " << m[1][1] << ", " << m[2][1] << ", " << m[3][1] << "}" << std::endl;
    os << "        {" << m[0][2] << ", " << m[1][2] << ", " << m[2][2] << ", " << m[3][2] << "}" << std::endl;
    os << "        {" << m[0][3] << ", " << m[1][3] << ", " << m[2][3] << ", " << m[3][3] << "}" << std::endl;
    return os;
}


namespace TissueForge::rendering {


    /**
     * An interactor that manages a rotation and translation transform for an
     * object, and updates these transforms via mouse interaction.
     *
     * This class implements the ArcBall, as described by Ken Shoemake in Graphics Gems IV.
     *
     * This class takes as input mouse events (mouse down, mouse drag, mouse up),
     * and creates the appropriate quaternions and 4x4 matrices
     * to represent the rotation given by the mouse.
     *
     * Axis constraints can also be explicitly set with the setConstraints() function.
     *
     * The current rotation is stored in the 4x4 float matrix 'rot'. It is also stored
     * in the quaternion 'q_now'.
     *
     * Based off Paul Rademacher's original GLUI toolkit, Copyright (c) 1998 Paul Rademacher
     * Included contributions from :
     *  - Feb 1998, Paul Rademacher (rademach@cs.unc.edu),
     *  - Oct 2003, Nigel Stewart - GLUI Code Cleaning
     */
    class ArcBallInteractor
    {
    public:
        ArcBallInteractor();
        ArcBallInteractor(const Magnum::Matrix4 &mtx);
        ArcBallInteractor(const Magnum::Vector2 &center, float radius);

        void setDampening(float d);

        void idle();


        /**
         * Get the rotation part of the transform matrix.
         */
        Magnum::Matrix4 rotation() const;


        /**
         * A mouse was pressed, this starts the drag motion.
         * This is in screen coordinates (pixels)
         */
        void mouseDown(int x, int y);


        void mouseUp();


        /**
         * Notify the arcball of mouse motion, this is in screen coordinates (pixels)
         */
        void mouseMotion(int x, int y, int shift, int ctrl, int alt);


        /**
         * Notify the arcball of mouse motion, this is in screen coordinates (pixels)
         */
        void mouseMotion(int x, int y);

        /**
         * Notify the arc ball that the window size has changed.
         */
        void setWindowSize(int width, int height);


        void setConstraints(bool constrain_x, bool constrain_y);

        void resetMouse();

        void reset();

    private:

        Magnum::Vector3 constrainVector(const Magnum::Vector3 &vector, const Magnum::Vector3 &axis);
        Magnum::Vector3 mouseToSphere(const Magnum::Vector2 &p);

        void setParams(const Magnum::Vector2 &center, float radius);

    //public:
        int is_mouse_down;  /* true for down, false for up */
        int is_spinning;
        Magnum::Quaternion q_now, q_down, q_drag, q_increment;
        Magnum::Vector2 down_pt;
        Magnum::Matrix4 rot, rot_increment;

        bool constraint_x, constraint_y;
        Magnum::Vector2  center;
        float radius, damp_factor;
        int zero_increment;

        friend std::ostream& operator<<(std::ostream& os, const ArcBallInteractor& ball);
    };

}

#endif // _SOURCE_RENDERING_TFARCBALLINTERACTOR_H_