# KokkosComputeThermalExpansionEigenstrain

!if! function=hasCapability('kokkos')

`KokkosComputeThermalExpansionEigenstrain` computes the isotropic thermal eigenstrain $\alpha (T - T_0) I$ from a temperature variable, a constant expansion coefficient and a stress-free temperature. Pass its name to the [!param](/Materials/KokkosComputeIsotropicLinearElasticStress/eigenstrain_names) of [KokkosComputeIsotropicLinearElasticStress](KokkosComputeIsotropicLinearElasticStress.md).

!syntax parameters /Materials/KokkosComputeThermalExpansionEigenstrain

!syntax inputs /Materials/KokkosComputeThermalExpansionEigenstrain

!syntax children /Materials/KokkosComputeThermalExpansionEigenstrain

!if-end!

!else
!include kokkos/kokkos_warning.md
