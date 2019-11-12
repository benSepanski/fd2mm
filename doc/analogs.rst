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


.. note::
    *(If you are just an end-user, you probably don't need to know this)*

    In firedrake, meshes and function spaces have a close relationship.
    In particular, due to some structure described in this
    `firedrake pull request <http://github.com/firedrakeproject/firedrake/pull/627>`_.
    ``fd2mm`` mimics this firedrake design style. 
    
    In short, it is the idea
    that every function space should have a mesh, and the coordinates of the mesh
    should be representable as a function on that same mesh, which must live
    on some function space on the mesh... etc.
    Under the hood, we divide between topological and geometric objects,
    roughly as so
    
    (1) A topological mesh *top* which holds information about connectivity
        and other topological properties, but nothing about geometry.

    (2) A function space *W = (S, top)* which is some finite element family
        and reference element *S*, along with a topological mesh *S*
    
    (3) A coordinateless function, which is a function on the dofs
        of a function space *W*
    
    (4) A mesh geometry *geo = (top, W, f)*, where *top* is some mesh
        topology, *W* is some function space on *top*, and
        *f* is some coordinateless function on *W* representing the
        coordinates of the mesh at each given degree of freedom.

    (5) A WithGeometry *WG = (V, geo)* which is a function space *V*
        together with a mesh geometry *geo* for the mesh topology
        *V* is defined on

    Thus, by the coordinates of a mesh geometry we mean

    (a) On the hidden back-end: a coordinateless function *f* on some function
        space defined only on the mesh topology
    (b) On the front-end: If the mesh geometry is *geo = (top, W, f)* and 
        *W = (S, top)* as above, we mean the function *f* projected onto the
        WithGeometry object *V=(S, geo)*. 

    Basically, it's this picture (where a->b if b depends on a)

    .. warning::
    
        In general, one may use a different function space for the mesh
        coordinates than you do for the final *WithGeometry*, this picture
        only shows how the classes depend on each other.
                

    .. image:: images/topversusgeo.png


.. automodule:: fd2mm.mesh


Function Space Analogs
======================
