Analogs
=======

Almost all of the work done by :mod:`fd2mm` is performed by :class:`Analog`
objects.  These objects are handed a firedrake object, then internally
manipulate the data so that it may be easily handled in meshmode.

.. automodule:: fd2mm.analog

A note on convention: if a firedrake object ``foo`` has an analog
``foo_analog``, any attribute of ``foo_analog`` which is also
an :class:`Analog`, say of a firedrake object ``bar``, is
usually written as ``foo_analog.bar_a``.
    

Reference Element Analogs
=========================


Cells
-----

Much of the work in both firedrake and meshmode is done on a reference element.
The realization of the element itself in firedrake is done on a
:class:`fiat.FIAT.reference_element.Cell`. We only provide support for
simplices, and so restrict ourselves to creating an analog for a simplex.

.. automodule:: fd2mm.cell


FInAT Elements
--------------

In firedrake, much of the calculation on the reference element is done using a
:class:`FInAT element <finat.fiat_elements.FiatElement>`.
Any :class:`meshmode.discretization.Discretization` uses a discontinuous
space, so we focus are attention entirely to ``'CG'`` and ``'DG'`` elements.


.. automodule:: fd2mm.finat_element


Mesh Analogs
============


Function Space Analogs
======================
