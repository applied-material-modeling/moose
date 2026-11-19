# KokkosComputeLagrangianIncrementalStrain

!if! function=hasCapability('kokkos')

`KokkosComputeLagrangianIncrementalStrain` accumulates a mechanical strain from the deformation gradient for output, matching the host Lagrangian strain calculator: with large kinematics the increment is $\operatorname{sym}(I - F_n F^{-1})$, and with small kinematics $\operatorname{sym}(F - F_n)$.

!syntax parameters /Materials/KokkosComputeLagrangianIncrementalStrain

!syntax inputs /Materials/KokkosComputeLagrangianIncrementalStrain

!syntax children /Materials/KokkosComputeLagrangianIncrementalStrain

!if-end!

!else
!include kokkos/kokkos_warning.md
