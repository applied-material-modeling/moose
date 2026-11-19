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
 * Computes the isotropic linear elastic small stress sigma = C : (sym(F) - I) and its strain
 * derivative C as full 3x3 and 3x3x3x3 Kokkos material properties from a deformation gradient
 * property, for example the output of KokkosComputeDeformationGradient, which includes the F-bar
 * correction and the homogenization gradient. In 2D the out-of-plane row of F is the identity, so
 * the stress is the plane-strain stress including sigma_zz.
 *
 * Optional eigenstrains are subtracted from the strain, as the host ComputeLagrangianStrain does
 * with small kinematics: sigma = C : (sym(F) - I - sum(eigenstrains)).
 */
class KokkosComputeIsotropicLinearElasticStress : public Moose::Kokkos::Material
{
public:
  static InputParameters validParams();

  KokkosComputeIsotropicLinearElasticStress(const InputParameters & parameters);

  template <typename Derived>
  KOKKOS_FUNCTION void computeQpProperties(const unsigned int qp, Datum & datum) const;

private:
  const Real _lambda;
  const Real _mu;
  Moose::Kokkos::MaterialProperty<Real, 2> _deformation_gradient;
  Moose::Kokkos::MaterialProperty<Real, 2> _stress;
  Moose::Kokkos::MaterialProperty<Real, 4> _tangent;
  const unsigned int _num_eigenstrains;
  Moose::Kokkos::Array<Moose::Kokkos::MaterialProperty<Real, 2>> _eigenstrains;
};

template <typename Derived>
KOKKOS_FUNCTION void
KokkosComputeIsotropicLinearElasticStress::computeQpProperties(const unsigned int qp,
                                                               Datum & datum) const
{
  const auto F = _deformation_gradient(datum, qp);
  auto stress = _stress(datum, qp);
  auto tangent = _tangent(datum, qp);

  Real strain[3][3];
  for (unsigned int i = 0; i < 3; ++i)
    for (unsigned int j = 0; j < 3; ++j)
      strain[i][j] = 0.5 * (F(i, j) + F(j, i)) - (i == j);
  for (unsigned int e = 0; e < _num_eigenstrains; ++e)
  {
    const auto eigenstrain = _eigenstrains[e](datum, qp);
    for (unsigned int i = 0; i < 3; ++i)
      for (unsigned int j = 0; j < 3; ++j)
        strain[i][j] -= eigenstrain(i, j);
  }

  const Real trace = strain[0][0] + strain[1][1] + strain[2][2];
  for (unsigned int i = 0; i < 3; ++i)
    for (unsigned int j = 0; j < 3; ++j)
    {
      stress(i, j) = _lambda * trace * (i == j) + 2 * _mu * strain[i][j];

      for (unsigned int k = 0; k < 3; ++k)
        for (unsigned int l = 0; l < 3; ++l)
          tangent(i, j, k, l) =
              _lambda * (i == j) * (k == l) + _mu * ((i == k) * (j == l) + (i == l) * (j == k));
    }
}
