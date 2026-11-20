# KokkosComputeLagrangianIncrementalStrain

!if! function=hasCapability('kokkos')

`KokkosComputeLagrangianIncrementalStrain` accumulates a mechanical strain from the deformation gradient for output, matching the host Lagrangian strain calculator: with large kinematics the increment is $\operatorname{sym}(I - F_n F^{-1})$, and with small kinematics $\operatorname{sym}(F - F_n)$. With
[!param](/Materials/KokkosComputeLagrangianIncrementalStrain/effective_strain) set, it also accumulates
the effective strain of the increments, $\bar{\varepsilon}_n + \sqrt{\tfrac{2}{3}\Delta\varepsilon : \Delta\varepsilon}$,
as the host [RankTwoInvariant](RankTwoInvariant.md) with `invariant = EffectiveStrain` does.

!syntax parameters /Materials/KokkosComputeLagrangianIncrementalStrain

!syntax inputs /Materials/KokkosComputeLagrangianIncrementalStrain

!syntax children /Materials/KokkosComputeLagrangianIncrementalStrain

!if-end!

!else
!include kokkos/kokkos_warning.md
