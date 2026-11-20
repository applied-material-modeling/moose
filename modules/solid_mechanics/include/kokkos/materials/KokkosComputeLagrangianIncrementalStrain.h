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
 * Accumulates the strain from the increments of a deformation gradient property, as the host
 * ComputeLagrangianStrain does for its total_strain and mechanical_strain (without eigenstrains):
 *
 * - large kinematics, with the linear kinematic approximation:
 *   strain = strain_old + sym(I - F_old F^-1)
 * - small kinematics: strain = strain_old + sym(F - F_old)
 *
 * The Kokkos total-Lagrangian stress materials work directly from the deformation gradient, so
 * this material only provides the strain output of the host system.
 *
 * With effective_strain set it also accumulates the effective strain of the increments,
 * effective_strain = effective_strain_old + sqrt(2/3 dstrain : dstrain), as the host
 * RankTwoInvariant with invariant = EffectiveStrain does.
 */
class KokkosComputeLagrangianIncrementalStrain : public Moose::Kokkos::Material
{
public:
  static InputParameters validParams();

  KokkosComputeLagrangianIncrementalStrain(const InputParameters & parameters);

  template <typename Derived>
  KOKKOS_FUNCTION void initQpStatefulProperties(const unsigned int qp, Datum & datum) const
  {
    auto F_store = _F_store(datum, qp);
    auto strain = _strain(datum, qp);
    for (unsigned int i = 0; i < 3; ++i)
      for (unsigned int j = 0; j < 3; ++j)
      {
        F_store(i, j) = i == j;
        strain(i, j) = 0;
      }
    if (_has_effective_strain)
      _effective_strain(datum, qp) = 0;
  }

  template <typename Derived>
  KOKKOS_FUNCTION void computeQpProperties(const unsigned int qp, Datum & datum) const;

private:
  const bool _large_kinematics;
  Moose::Kokkos::MaterialProperty<Real, 2> _F;
  Moose::Kokkos::MaterialProperty<Real, 2> _F_store;
  Moose::Kokkos::MaterialProperty<Real, 2> _F_store_old;
  Moose::Kokkos::MaterialProperty<Real, 2> _strain;
  Moose::Kokkos::MaterialProperty<Real, 2> _strain_old;
  const bool _has_effective_strain;
  Moose::Kokkos::MaterialProperty<Real, 0> _effective_strain;
  Moose::Kokkos::MaterialProperty<Real, 0> _effective_strain_old;
};

template <typename Derived>
KOKKOS_FUNCTION void
KokkosComputeLagrangianIncrementalStrain::computeQpProperties(const unsigned int qp,
                                                              Datum & datum) const
{
  using Moose::Kokkos::Real33;

  Real33 F, F_old;
  {
    const auto Fp = _F(datum, qp);
    const auto Fo = _F_store_old(datum, qp);
    auto Fs = _F_store(datum, qp);
    for (unsigned int i = 0; i < 3; ++i)
      for (unsigned int j = 0; j < 3; ++j)
      {
        F(i, j) = Fp(i, j);
        Fs(i, j) = Fp(i, j);
        F_old(i, j) = Fo(i, j);
      }
  }

  Real33 dL;
  if (_large_kinematics)
  {
    const Real33 f_inv = F_old * F.inverse();
    for (unsigned int i = 0; i < 3; ++i)
      for (unsigned int j = 0; j < 3; ++j)
        dL(i, j) = (i == j) - f_inv(i, j);
  }
  else
    for (unsigned int i = 0; i < 3; ++i)
      for (unsigned int j = 0; j < 3; ++j)
        dL(i, j) = F(i, j) - F_old(i, j);

  const auto strain_old = _strain_old(datum, qp);
  auto strain = _strain(datum, qp);
  Real dd2 = 0;
  for (unsigned int i = 0; i < 3; ++i)
    for (unsigned int j = 0; j < 3; ++j)
    {
      const Real dd = 0.5 * (dL(i, j) + dL(j, i));
      strain(i, j) = strain_old(i, j) + dd;
      dd2 += dd * dd;
    }

  if (_has_effective_strain)
    _effective_strain(datum, qp) =
        _effective_strain_old(datum, qp) + ::Kokkos::sqrt(2.0 / 3.0 * dd2);
}
