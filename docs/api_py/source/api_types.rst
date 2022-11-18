Basic Tissue Forge Types
-------------------------

Tissue Forge uses some basic types that provide support and convenience
methods for particular operations, especially concerning vector and
tensor operations. Some of these types are completely native to
Tissue Forge, and others constructed partially or completely from
types distributed in various Tissue Forge dependencies (*e.g.*,
:class:`FVector3` from :class:`Vector3`, from
`Magnum <https://magnum.graphics/>`_.

.. currentmodule:: tissue_forge

.. autoclass:: dVector2

    A 2D vector with ``double`` elements

    .. automethod:: xAxis

    .. automethod:: yAxis

    .. automethod:: xScale

    .. automethod:: yScale

    .. automethod:: x

    .. automethod:: y

    .. automethod:: flipped

    .. automethod:: dot

    .. automethod:: cross

    .. automethod:: angle

    .. automethod:: __len__

    .. automethod:: __iadd__

    .. automethod:: __add__

    .. automethod:: __isub__

    .. automethod:: __sub__

    .. automethod:: __imul__

    .. automethod:: __mul__

    .. automethod:: __itruediv__

    .. automethod:: __truediv__

    .. automethod:: length

    .. automethod:: normalized

    .. automethod:: resized

    .. automethod:: projected

    .. automethod:: projectedOntoNormalized

    .. automethod:: distance

    .. automethod:: __getitem__

    .. automethod:: __setitem__

    .. automethod:: as_list

    .. automethod:: __str__

.. autoclass:: fVector2

    A 2D vector with ``float`` elements

    .. automethod:: xAxis

    .. automethod:: yAxis

    .. automethod:: xScale

    .. automethod:: yScale

    .. automethod:: x

    .. automethod:: y

    .. automethod:: flipped

    .. automethod:: dot

    .. automethod:: cross

    .. automethod:: angle

    .. automethod:: __len__

    .. automethod:: __iadd__

    .. automethod:: __add__

    .. automethod:: __isub__

    .. automethod:: __sub__

    .. automethod:: __imul__

    .. automethod:: __mul__

    .. automethod:: __itruediv__

    .. automethod:: __truediv__

    .. automethod:: length

    .. automethod:: normalized

    .. automethod:: resized

    .. automethod:: projected

    .. automethod:: projectedOntoNormalized

    .. automethod:: distance

    .. automethod:: __getitem__

    .. automethod:: __setitem__

    .. automethod:: as_list

    .. automethod:: __str__

.. autoclass:: iVector2

    A 2D vector with ``int`` elements

    .. automethod:: xAxis

    .. automethod:: yAxis

    .. automethod:: xScale

    .. automethod:: yScale

    .. automethod:: x

    .. automethod:: y

    .. automethod:: flipped

    .. automethod:: dot

    .. automethod:: cross

    .. automethod:: angle

    .. automethod:: __len__

    .. automethod:: __iadd__

    .. automethod:: __add__

    .. automethod:: __isub__

    .. automethod:: __sub__

    .. automethod:: __imul__

    .. automethod:: __mul__

    .. automethod:: __itruediv__

    .. automethod:: __truediv__

    .. automethod:: __getitem__

    .. automethod:: __setitem__

    .. automethod:: as_list

    .. automethod:: __str__

.. autoclass:: dVector3

    A 3D vector with ``double`` elements

    .. automethod:: xAxis

    .. automethod:: yAxis

    .. automethod:: zAxis

    .. automethod:: xScale

    .. automethod:: yScale

    .. automethod:: zScale

    .. automethod:: x

    .. automethod:: y

    .. automethod:: z

    .. automethod:: r

    .. automethod:: g

    .. automethod:: b

    .. automethod:: xy

    .. automethod:: flipped

    .. automethod:: dot

    .. automethod:: cross

    .. automethod:: angle

    .. automethod:: __len__

    .. automethod:: __iadd__

    .. automethod:: __add__

    .. automethod:: __isub__

    .. automethod:: __sub__

    .. automethod:: __imul__

    .. automethod:: __mul__

    .. automethod:: __itruediv__

    .. automethod:: __truediv__

    .. automethod:: length

    .. automethod:: normalized

    .. automethod:: resized

    .. automethod:: projected

    .. automethod:: projectedOntoNormalized

    .. automethod:: distance

    .. automethod:: __getitem__

    .. automethod:: __setitem__

    .. automethod:: as_list

    .. automethod:: __str__

.. autoclass:: fVector3

    A 3D vector with ``float`` elements

    .. automethod:: xAxis

    .. automethod:: yAxis

    .. automethod:: zAxis

    .. automethod:: xScale

    .. automethod:: yScale

    .. automethod:: zScale

    .. automethod:: x

    .. automethod:: y

    .. automethod:: z

    .. automethod:: r

    .. automethod:: g

    .. automethod:: b

    .. automethod:: xy

    .. automethod:: flipped

    .. automethod:: dot

    .. automethod:: cross

    .. automethod:: angle

    .. automethod:: __len__

    .. automethod:: __iadd__

    .. automethod:: __add__

    .. automethod:: __isub__

    .. automethod:: __sub__

    .. automethod:: __imul__

    .. automethod:: __mul__

    .. automethod:: __itruediv__

    .. automethod:: __truediv__

    .. automethod:: length

    .. automethod:: normalized

    .. automethod:: resized

    .. automethod:: projected

    .. automethod:: projectedOntoNormalized

    .. automethod:: distance

    .. automethod:: __getitem__

    .. automethod:: __setitem__

    .. automethod:: as_list

    .. automethod:: __str__

.. autoclass:: iVector3

    A 3D vector with ``int`` elements

    .. automethod:: xAxis

    .. automethod:: yAxis

    .. automethod:: zAxis

    .. automethod:: xScale

    .. automethod:: yScale

    .. automethod:: zScale

    .. automethod:: x

    .. automethod:: y

    .. automethod:: z

    .. automethod:: r

    .. automethod:: g

    .. automethod:: b

    .. automethod:: xy

    .. automethod:: flipped

    .. automethod:: dot

    .. automethod:: cross

    .. automethod:: angle

    .. automethod:: __len__

    .. automethod:: __iadd__

    .. automethod:: __add__

    .. automethod:: __isub__

    .. automethod:: __sub__

    .. automethod:: __imul__

    .. automethod:: __mul__

    .. automethod:: __itruediv__

    .. automethod:: __truediv__

    .. automethod:: __getitem__

    .. automethod:: __setitem__

    .. automethod:: as_list

    .. automethod:: __str__

.. autoclass:: dVector4

    A 4D vector with ``double`` elements

    .. automethod:: x

    .. automethod:: y

    .. automethod:: z

    .. automethod:: w

    .. automethod:: r

    .. automethod:: g

    .. automethod:: b

    .. automethod:: a

    .. automethod:: xyz

    .. automethod:: rgb

    .. automethod:: xy

    .. automethod:: flipped

    .. automethod:: dot

    .. automethod:: angle

    .. automethod:: __len__

    .. automethod:: __iadd__

    .. automethod:: __add__

    .. automethod:: __isub__

    .. automethod:: __sub__

    .. automethod:: __imul__

    .. automethod:: __mul__

    .. automethod:: __itruediv__

    .. automethod:: __truediv__

    .. automethod:: length

    .. automethod:: normalized

    .. automethod:: resized

    .. automethod:: projected

    .. automethod:: projectedOntoNormalized

    .. automethod:: distance

    .. automethod:: distanceScaled

    .. automethod:: planeEquation

    .. automethod:: __getitem__

    .. automethod:: __setitem__

    .. automethod:: as_list

    .. automethod:: __str__

.. autoclass:: fVector4

    A 4D vector with ``float`` elements

    .. automethod:: x

    .. automethod:: y

    .. automethod:: z

    .. automethod:: w

    .. automethod:: r

    .. automethod:: g

    .. automethod:: b

    .. automethod:: a

    .. automethod:: xyz

    .. automethod:: rgb

    .. automethod:: xy

    .. automethod:: flipped

    .. automethod:: dot

    .. automethod:: angle

    .. automethod:: __len__

    .. automethod:: __iadd__

    .. automethod:: __add__

    .. automethod:: __isub__

    .. automethod:: __sub__

    .. automethod:: __imul__

    .. automethod:: __mul__

    .. automethod:: __itruediv__

    .. automethod:: __truediv__

    .. automethod:: length

    .. automethod:: normalized

    .. automethod:: resized

    .. automethod:: projected

    .. automethod:: projectedOntoNormalized

    .. automethod:: distance

    .. automethod:: distanceScaled

    .. automethod:: planeEquation

    .. automethod:: __getitem__

    .. automethod:: __setitem__

    .. automethod:: as_list

    .. automethod:: __str__

.. autoclass:: iVector4

    A 4D vector with ``int`` elements

    .. automethod:: x

    .. automethod:: y

    .. automethod:: z

    .. automethod:: w

    .. automethod:: r

    .. automethod:: g

    .. automethod:: b

    .. automethod:: a

    .. automethod:: xyz

    .. automethod:: rgb

    .. automethod:: xy

    .. automethod:: flipped

    .. automethod:: dot

    .. automethod:: angle

    .. automethod:: __len__

    .. automethod:: __iadd__

    .. automethod:: __add__

    .. automethod:: __isub__

    .. automethod:: __sub__

    .. automethod:: __imul__

    .. automethod:: __mul__

    .. automethod:: __itruediv__

    .. automethod:: __truediv__

    .. automethod:: __getitem__

    .. automethod:: __setitem__

    .. automethod:: as_list

    .. automethod:: __str__

.. autoclass:: dMatrix3

    A 3x3 square matrix with ``double`` elements

    .. automethod:: rotation

    .. automethod:: shearingX

    .. automethod:: shearingY

    .. automethod:: isRigidTransformation

    .. automethod:: invertedRigid

    .. automethod:: __neg__

    .. automethod:: __iadd__

    .. automethod:: __add__

    .. automethod:: __isub__

    .. automethod:: __sub__

    .. automethod:: __imul__

    .. automethod:: __mul__

    .. automethod:: __itruediv__

    .. automethod:: __truediv__

    .. automethod:: flippedCols

    .. automethod:: flippedRows

    .. automethod:: row

    .. automethod:: __mul__

    .. automethod:: transposed

    .. automethod:: diagonal

    .. automethod:: inverted

    .. automethod:: invertedOrthogonal

    .. automethod:: __len__

    .. automethod:: __getitem__

    .. automethod:: __setitem__

    .. automethod:: as_lists

    .. automethod:: __str__

.. autoclass:: fMatrix3

    A 3x3 square matrix with ``float`` elements

    .. automethod:: rotation

    .. automethod:: shearingX

    .. automethod:: shearingY

    .. automethod:: isRigidTransformation

    .. automethod:: invertedRigid

    .. automethod:: __neg__

    .. automethod:: __iadd__

    .. automethod:: __add__

    .. automethod:: __isub__

    .. automethod:: __sub__

    .. automethod:: __imul__

    .. automethod:: __mul__

    .. automethod:: __itruediv__

    .. automethod:: __truediv__

    .. automethod:: flippedCols

    .. automethod:: flippedRows

    .. automethod:: row

    .. automethod:: __mul__

    .. automethod:: transposed

    .. automethod:: diagonal

    .. automethod:: inverted

    .. automethod:: invertedOrthogonal

    .. automethod:: __len__

    .. automethod:: __getitem__

    .. automethod:: __setitem__

    .. automethod:: as_lists

    .. automethod:: __str__

.. autoclass:: dMatrix4

    A 4x4 square matrix with ``double`` elements

    .. automethod:: rotationX

    .. automethod:: rotationY

    .. automethod:: rotationZ

    .. automethod:: reflection

    .. automethod:: shearingXY

    .. automethod:: shearingXZ

    .. automethod:: shearingYZ

    .. automethod:: orthographicProjection

    .. automethod:: perspectiveProjection

    .. automethod:: lookAt

    .. automethod:: isRigidTransformation

    .. automethod:: rotationScaling

    .. automethod:: rotationShear

    .. automethod:: rotation

    .. automethod:: rotationNormalized

    .. automethod:: scalingSquared

    .. automethod:: scaling

    .. automethod:: uniformScalingSquared

    .. automethod:: uniformScaling

    .. automethod:: normalMatrix

    .. automethod:: right

    .. automethod:: up

    .. automethod:: backward

    .. automethod:: translation

    .. automethod:: invertedRigid

    .. automethod:: transformVector

    .. automethod:: transformPoint

    .. automethod:: __neg__

    .. automethod:: __iadd__

    .. automethod:: __add__

    .. automethod:: __isub__

    .. automethod:: __sub__

    .. automethod:: __imul__

    .. automethod:: __itruediv__

    .. automethod:: __truediv__

    .. automethod:: flippedCols

    .. automethod:: flippedRows

    .. automethod:: row

    .. automethod:: __mul__

    .. automethod:: transposed

    .. automethod:: diagonal

    .. automethod:: inverted

    .. automethod:: invertedOrthogonal

    .. automethod:: __len__

    .. automethod:: __getitem__

    .. automethod:: __setitem__

    .. automethod:: as_lists

    .. automethod:: __str__

.. autoclass:: fMatrix4

    A 4x4 square matrix with ``float`` elements

    .. automethod:: rotationX

    .. automethod:: rotationY

    .. automethod:: rotationZ

    .. automethod:: reflection

    .. automethod:: shearingXY

    .. automethod:: shearingXZ

    .. automethod:: shearingYZ

    .. automethod:: orthographicProjection

    .. automethod:: perspectiveProjection

    .. automethod:: lookAt

    .. automethod:: isRigidTransformation

    .. automethod:: rotationScaling

    .. automethod:: rotationShear

    .. automethod:: rotation

    .. automethod:: rotationNormalized

    .. automethod:: scalingSquared

    .. automethod:: scaling

    .. automethod:: uniformScalingSquared

    .. automethod:: uniformScaling

    .. automethod:: normalMatrix

    .. automethod:: right

    .. automethod:: up

    .. automethod:: backward

    .. automethod:: translation

    .. automethod:: invertedRigid

    .. automethod:: transformVector

    .. automethod:: transformPoint

    .. automethod:: __neg__

    .. automethod:: __iadd__

    .. automethod:: __add__

    .. automethod:: __isub__

    .. automethod:: __sub__

    .. automethod:: __imul__

    .. automethod:: __itruediv__

    .. automethod:: __truediv__

    .. automethod:: flippedCols

    .. automethod:: flippedRows

    .. automethod:: row

    .. automethod:: __mul__

    .. automethod:: transposed

    .. automethod:: diagonal

    .. automethod:: inverted

    .. automethod:: invertedOrthogonal

    .. automethod:: __len__

    .. automethod:: __getitem__

    .. automethod:: __setitem__

    .. automethod:: as_lists

    .. automethod:: __str__

.. autoclass:: dQuaternion

    A quaternion with ``double`` elements

    .. automethod:: rotation

    .. automethod:: fromMatrix

    .. automethod:: data

    .. automethod:: __eq__

    .. automethod:: __ne__

    .. automethod:: isNormalized

    .. automethod:: vector

    .. automethod:: scalar

    .. automethod:: angle

    .. automethod:: axis

    .. automethod:: toMatrix

    .. automethod:: toEuler

    .. automethod:: __neg__

    .. automethod:: __iadd__

    .. automethod:: __add__

    .. automethod:: __isub__

    .. automethod:: __sub__

    .. automethod:: __imul__

    .. automethod:: __itruediv__

    .. automethod:: __truediv__

    .. automethod:: __mul__

    .. automethod:: dot

    .. automethod:: length

    .. automethod:: normalized

    .. automethod:: conjugated

    .. automethod:: inverted

    .. automethod:: invertedNormalized

    .. automethod:: transformVector

    .. automethod:: transformVectorNormalized

    .. automethod:: __getitem__

    .. automethod:: __setitem__

    .. automethod:: as_list

    .. automethod:: __str__

.. autoclass:: fQuaternion

    A quaternion with ``float`` elements

    .. automethod:: rotation

    .. automethod:: fromMatrix

    .. automethod:: data

    .. automethod:: __eq__

    .. automethod:: __ne__

    .. automethod:: isNormalized

    .. automethod:: vector

    .. automethod:: scalar

    .. automethod:: angle

    .. automethod:: axis

    .. automethod:: toMatrix

    .. automethod:: toEuler

    .. automethod:: __neg__

    .. automethod:: __iadd__

    .. automethod:: __add__

    .. automethod:: __isub__

    .. automethod:: __sub__

    .. automethod:: __imul__

    .. automethod:: __itruediv__

    .. automethod:: __truediv__

    .. automethod:: __mul__

    .. automethod:: dot

    .. automethod:: length

    .. automethod:: normalized

    .. automethod:: conjugated

    .. automethod:: inverted

    .. automethod:: invertedNormalized

    .. automethod:: transformVector

    .. automethod:: transformVectorNormalized

    .. automethod:: __getitem__

    .. automethod:: __setitem__

    .. automethod:: as_list

    .. automethod:: __str__


.. currentmodule:: tissue_forge

.. autoclass:: FVector2
    :show-inheritance:

    Alias for :class:fVector2 or :class:dVector2, depending on the installation

.. autoclass:: FVector3
    :show-inheritance:

    Alias for :class:fVector3 or :class:dVector3, depending on the installation

.. autoclass:: FVector4
    :show-inheritance:

    Alias for :class:fVector4 or :class:dVector4, depending on the installation

.. autoclass:: FMatrix3
    :show-inheritance:

    Alias for :class:fMatrix3 or :class:dMatrix3, depending on the installation

.. autoclass:: FMatrix4
    :show-inheritance:

    Alias for :class:fMatrix4 or :class:dMatrix4, depending on the installation

.. autoclass:: FQuaternion
    :show-inheritance:

    Alias for :class:fQuaternion or :class:dQuaternion, depending on the installation
