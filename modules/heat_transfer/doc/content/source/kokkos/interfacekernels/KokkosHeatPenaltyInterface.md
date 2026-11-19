# KokkosHeatPenaltyInterface

!if! function=hasCapability('kokkos')

`KokkosHeatPenaltyInterface` is a Kokkos penalty interface kernel for heat conduction across an interface of a broken mesh. It imposes the flux $p (T - T_\text{neighbor})$ on both sides of the interface, as [PenaltyInterfaceDiffusion](PenaltyInterfaceDiffusion.md) does, which models a finite interfacial conductance $p$.

!syntax parameters /InterfaceKernels/KokkosHeatPenaltyInterface

!syntax inputs /InterfaceKernels/KokkosHeatPenaltyInterface

!syntax children /InterfaceKernels/KokkosHeatPenaltyInterface

!if-end!

!else
!include kokkos/kokkos_warning.md
