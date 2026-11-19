# KokkosComputeSimoHughesJ2PlasticityStress

!if! function=hasCapability('kokkos')

`KokkosComputeSimoHughesJ2PlasticityStress` computes the stress for finite-strain $J_2$ plasticity with linear isotropic hardening, following the multiplicative formulation of Simo and Hughes (the volume-preserving elastic left Cauchy-Green tensor is the internal variable, and the return mapping is closed form). It provides the PK1 stress and its derivative with respect to the displacement gradient for the Kokkos total Lagrangian homogenization system, including the F-bar volumetric locking correction when [!param](/Materials/KokkosComputeSimoHughesJ2PlasticityStress/stabilize_strain) is set, and the Cauchy stress for output. It gives the same results as the host `ComputeSimoHughesJ2PlasticityStress` with linear hardening.

!syntax parameters /Materials/KokkosComputeSimoHughesJ2PlasticityStress

!syntax inputs /Materials/KokkosComputeSimoHughesJ2PlasticityStress

!syntax children /Materials/KokkosComputeSimoHughesJ2PlasticityStress

!if-end!

!else
!include kokkos/kokkos_warning.md
