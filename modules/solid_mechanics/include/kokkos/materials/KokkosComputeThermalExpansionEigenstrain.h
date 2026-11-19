//* This file is part of the MOOSE framework
//* https://mooseframework.inl.gov
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#pragma once

#include "KokkosMaterial.h"

/**
 * Kokkos version of ComputeThermalExpansionEigenstrain for a constant expansion coefficient:
 * eigenstrain = alpha (T - T_stress_free) I.
 */
class KokkosComputeThermalExpansionEigenstrain : public Moose::Kokkos::Material
{
public:
  static InputParameters validParams();

  KokkosComputeThermalExpansionEigenstrain(const InputParameters & parameters);

  template <typename Derived>
  KOKKOS_FUNCTION void computeQpProperties(const unsigned int qp, Datum & datum) const
  {
    const Real strain = _alpha * (_temperature(datum, qp) - _stress_free_temperature);
    auto eigenstrain = _eigenstrain(datum, qp);
    for (unsigned int i = 0; i < 3; ++i)
      for (unsigned int j = 0; j < 3; ++j)
        eigenstrain(i, j) = (i == j) * strain;
  }

private:
  const Moose::Kokkos::VariableValue _temperature;
  const Real _alpha;
  const Real _stress_free_temperature;
  Moose::Kokkos::MaterialProperty<Real, 2> _eigenstrain;
};
