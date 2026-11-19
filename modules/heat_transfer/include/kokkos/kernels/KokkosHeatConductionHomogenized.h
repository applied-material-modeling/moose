//* This file is part of the MOOSE framework
//* https://mooseframework.inl.gov
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#pragma once

#include "KokkosKernelGrad.h"

/**
 * Kokkos version of ADHeatConductionHomogenized: heat conduction of a periodic temperature
 * fluctuation T driven by a constant imposed macroscopic gradient G,
 *   int k (grad T + G) . grad(psi)
 */
class KokkosHeatConductionHomogenized : public Moose::Kokkos::KernelGrad
{
  using Real3 = Moose::Kokkos::Real3;

public:
  static InputParameters validParams();

  KokkosHeatConductionHomogenized(const InputParameters & parameters);

  template <typename Derived>
  KOKKOS_FUNCTION Real3 precomputeQpResidual(const unsigned int qp, AssemblyDatum & datum) const;
  template <typename Derived>
  KOKKOS_FUNCTION Real3 precomputeQpJacobian(const unsigned int j,
                                             const unsigned int qp,
                                             AssemblyDatum & datum) const;

private:
  Moose::Kokkos::MaterialProperty<Real> _thermal_conductivity;
  const Real3 _imposed_gradient;
};

template <typename Derived>
KOKKOS_FUNCTION Moose::Kokkos::Real3
KokkosHeatConductionHomogenized::precomputeQpResidual(const unsigned int qp,
                                                      AssemblyDatum & datum) const
{
  return _thermal_conductivity(datum, qp) * (_grad_u(datum, qp) + _imposed_gradient);
}

template <typename Derived>
KOKKOS_FUNCTION Moose::Kokkos::Real3
KokkosHeatConductionHomogenized::precomputeQpJacobian(const unsigned int j,
                                                      const unsigned int qp,
                                                      AssemblyDatum & datum) const
{
  return _thermal_conductivity(datum, qp) * _grad_phi(datum, j, qp);
}
