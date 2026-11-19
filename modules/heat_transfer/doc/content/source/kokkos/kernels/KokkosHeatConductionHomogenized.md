# KokkosHeatConductionHomogenized

!if! function=hasCapability('kokkos')

`KokkosHeatConductionHomogenized` is the Kokkos version of `ADHeatConductionHomogenized`. It adds an imposed macroscopic temperature gradient to the periodic temperature fluctuation, so the residual is $(k (\nabla T + G), \nabla \psi)$ with the constant gradient $G$ given by its components.

!syntax parameters /Kernels/KokkosHeatConductionHomogenized

!syntax inputs /Kernels/KokkosHeatConductionHomogenized

!syntax children /Kernels/KokkosHeatConductionHomogenized

!if-end!

!else
!include kokkos/kokkos_warning.md
