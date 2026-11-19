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
 * Kokkos version of ComputeSimoHughesJ2PlasticityStress with an isotropic elasticity tensor and a
 * linear hardening flow stress sigma_y + h * effective_plastic_strain.
 *
 * The return mapping, the consistent tangent dPK1/dF and the stateful quantities
 * (effective_plastic_strain and volume_preserving_elastic_left_cauchy_green_strain) follow the
 * host material term by term. The flow stress is linear in the effective plastic strain, so the
 * host's Newton return mapping converges in one iteration and is evaluated here in closed form.
 *
 * The material publishes what KokkosTotalLagrangianStressDivergence and
 * KokkosHomogenizedTotalLagrangianStressDivergence consume with large kinematics: pk1_stress and
 * dpk1_d_grad_u and, with stabilize_strain, the total-mode F-bar chain fbar_A, fbar_gamma_over_3
 * and dpk1_bypass_fbar (as KokkosComputeLagrangianStressCustomPK2 does). It also publishes the
 * Cauchy stress P F_ust^T / det(F_ust), as the host ComputeLagrangianStressPK1 does, where F_ust
 * is the unstabilized deformation gradient including the homogenization gradient.
 */
class KokkosComputeSimoHughesJ2PlasticityStress : public Moose::Kokkos::Material
{
public:
  static InputParameters validParams();

  KokkosComputeSimoHughesJ2PlasticityStress(const InputParameters & parameters);

  void computeProperties() override;

  template <typename Derived>
  KOKKOS_FUNCTION void initQpStatefulProperties(const unsigned int qp, Datum & datum) const;

  template <typename Derived>
  KOKKOS_FUNCTION void computeQpProperties(const unsigned int qp, Datum & datum) const;

private:
  /// Shear and bulk moduli
  const Real _G;
  const Real _K;
  /// Linear hardening flow stress sigma_y + h * ep
  const Real _yield_stress;
  const Real _hardening_modulus;

  const unsigned int _ndisp;
  const Moose::Kokkos::VariableGradient _grad_displacements;
  const bool _stabilize_strain;

  /// Deformation gradient the stress is computed from (F-bar corrected, plus the homogenization
  /// gradient)
  Moose::Kokkos::MaterialProperty<Real, 2> _F;
  /// Unstabilized deformation gradient including the homogenization gradient, for the Cauchy stress
  Moose::Kokkos::MaterialProperty<Real, 2> _F_ust;
  /// Element average of the displacement-only deformation gradient (F-bar chain)
  Moose::Kokkos::MaterialProperty<Real, 2> _F_avg;

  /// Copy of _F kept to provide the previous step's value
  Moose::Kokkos::MaterialProperty<Real, 2> _F_store;
  Moose::Kokkos::MaterialProperty<Real, 2> _F_store_old;
  Moose::Kokkos::MaterialProperty<Real> _ep;
  Moose::Kokkos::MaterialProperty<Real> _ep_old;
  Moose::Kokkos::MaterialProperty<Real, 2> _be;
  Moose::Kokkos::MaterialProperty<Real, 2> _be_old;

  Moose::Kokkos::MaterialProperty<Real, 2> _pk1_stress;
  Moose::Kokkos::MaterialProperty<Real, 4> _dpk1_d_grad_u;
  Moose::Kokkos::MaterialProperty<Real, 2> _cauchy_stress;
  Moose::Kokkos::MaterialProperty<Real, 2> _A;
  Moose::Kokkos::MaterialProperty<Real, 0> _gamma_over_3;
  Moose::Kokkos::MaterialProperty<Real, 4> _dpk1_bypass_fbar;

  bool _need_jacobian;
  Moose::Kokkos::Scalar<bool> _need_jacobian_device;
};

template <typename Derived>
KOKKOS_FUNCTION void
KokkosComputeSimoHughesJ2PlasticityStress::initQpStatefulProperties(const unsigned int qp,
                                                                     Datum & datum) const
{
  auto F_store = _F_store(datum, qp);
  auto be = _be(datum, qp);
  for (unsigned int i = 0; i < 3; ++i)
    for (unsigned int j = 0; j < 3; ++j)
    {
      F_store(i, j) = i == j;
      be(i, j) = i == j;
    }
  _ep(datum, qp) = 0;
}

template <typename Derived>
KOKKOS_FUNCTION void
KokkosComputeSimoHughesJ2PlasticityStress::computeQpProperties(const unsigned int qp,
                                                               Datum & datum) const
{
  using Moose::Kokkos::Real33;

  Real33 F, F_old, be_old;
  {
    const auto Fp = _F(datum, qp);
    const auto Fo = _F_store_old(datum, qp);
    const auto bo = _be_old(datum, qp);
    auto Fs = _F_store(datum, qp);
    for (unsigned int i = 0; i < 3; ++i)
      for (unsigned int j = 0; j < 3; ++j)
      {
        F(i, j) = Fp(i, j);
        Fs(i, j) = Fp(i, j);
        F_old(i, j) = Fo(i, j);
        be_old(i, j) = bo(i, j);
      }
  }

  const Real detJ = F.determinant();
  const Real33 Fit = F.inverse().transpose();

  // Update configuration: f = F F_old^-1, the volume-preserving part f_bar = f / det(f)^(1/3)
  const Real33 inv_df = F_old * F.inverse();
  const Real33 f = inv_df.inverse();
  const Real cbrt_detf = ::Kokkos::cbrt(f.determinant());
  const Real33 f_bar = (1.0 / cbrt_detf) * f;

  // Elastic predictor
  Real33 be = f_bar * be_old * f_bar.transpose();
  Real tr_be = be(0, 0) + be(1, 1) + be(2, 2);
  Real33 s;
  Real s_norm2 = 0;
  for (unsigned int i = 0; i < 3; ++i)
    for (unsigned int j = 0; j < 3; ++j)
    {
      s(i, j) = _G * (be(i, j) - (i == j) * tr_be / 3);
      s_norm2 += s(i, j) * s(i, j);
    }
  const Real s_norm = ::Kokkos::sqrt(s_norm2);
  // Same zero test as the host (MooseUtils::absoluteFuzzyEqual with the default tolerance)
  const bool zero_s = ::Kokkos::fabs(s_norm) <= 1e-12;
  Real33 Np;
  for (unsigned int i = 0; i < 3; ++i)
    for (unsigned int j = 0; j < 3; ++j)
      Np(i, j) = zero_s ? ::Kokkos::sqrt(0.5) * (i == j) : ::Kokkos::sqrt(1.5) * s(i, j) / s_norm;
  const Real s_eff = s.contract(Np);

  const bool jac = _need_jacobian_device;

  // Derivative of the trial strain with respect to F:
  // d_be_d_F(i,j,k,l) = sum_m F_old^-1(l,m) X(i,j,k,m),
  // X(i,j,k,m) = (d_ik (f_bar be_old^T)(j,m) + d_jk (f_bar be_old)(i,m)) / det(f)^(1/3)
  //              - 2/3 be(i,j) inv_df(m,k)
  Real d_be_d_F[3][3][3][3];
  if (jac)
  {
    const Real33 F_old_inv = F_old.inverse();
    const Real33 fbo_T = f_bar * be_old.transpose();
    const Real33 fbo = f_bar * be_old;
    for (unsigned int i = 0; i < 3; ++i)
      for (unsigned int j = 0; j < 3; ++j)
        for (unsigned int k = 0; k < 3; ++k)
          for (unsigned int l = 0; l < 3; ++l)
          {
            Real v = 0;
            for (unsigned int m = 0; m < 3; ++m)
            {
              const Real X = ((i == k) * fbo_T(j, m) + (j == k) * fbo(i, m)) / cbrt_detf -
                             2.0 / 3.0 * be(i, j) * inv_df(m, k);
              v += F_old_inv(l, m) * X;
            }
            d_be_d_F[i][j][k][l] = v;
          }
  }

  // Return mapping. The residual R(dep) = s_eff - G dep tr(be) - (sigma_y + h (ep_old + dep)) is
  // linear in dep, so the host's Newton iteration from dep = 0 converges in one step.
  const Real ep_old = _ep_old(datum, qp);
  const Real R0 = s_eff - (_yield_stress + _hardening_modulus * ep_old);
  Real delta_ep = 0;
  if (R0 > 0)
  {
    const Real denom = _G * tr_be + _hardening_modulus;
    delta_ep = R0 / denom;

    if (jac)
    {
      // d(delta_ep)/d(be) as accumulated by the host's preStep after its single Newton step
      Real33 d_dep_d_be;
      for (unsigned int k = 0; k < 3; ++k)
        for (unsigned int l = 0; l < 3; ++l)
          d_dep_d_be(k, l) = _G * Np(k, l) / denom - R0 * _G * (k == l) / (denom * denom);

      // M = 2/3 (tr(be) Np x d_dep_d_be + delta_ep Np x I + delta_ep tr(be) dN/dbe)
      // and d_be_d_F -= M : d_be_d_F
      const Real c_n = zero_s ? 0.0 : _G / ::Kokkos::sqrt(6.0) / s_norm;
      Real updated[3][3][3][3];
      for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = 0; j < 3; ++j)
          for (unsigned int k = 0; k < 3; ++k)
            for (unsigned int l = 0; l < 3; ++l)
            {
              Real v = 0;
              for (unsigned int m = 0; m < 3; ++m)
                for (unsigned int n = 0; n < 3; ++n)
                {
                  const Real dN = c_n * (3.0 * (i == m) * (j == n) - 2.0 * Np(i, j) * Np(m, n) -
                                         (i == j) * (m == n));
                  const Real M = 2.0 / 3.0 *
                                 (tr_be * Np(i, j) * d_dep_d_be(m, n) +
                                  delta_ep * Np(i, j) * (m == n) + delta_ep * tr_be * dN);
                  v += M * d_be_d_F[m][n][k][l];
                }
              updated[i][j][k][l] = d_be_d_F[i][j][k][l] - v;
            }
      for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = 0; j < 3; ++j)
          for (unsigned int k = 0; k < 3; ++k)
            for (unsigned int l = 0; l < 3; ++l)
              d_be_d_F[i][j][k][l] = updated[i][j][k][l];
    }
  }

  // Update the intermediate and current configurations
  _ep(datum, qp) = ep_old + delta_ep;
  {
    const Real c = 2.0 / 3.0 * delta_ep * tr_be;
    auto be_prop = _be(datum, qp);
    for (unsigned int i = 0; i < 3; ++i)
      for (unsigned int j = 0; j < 3; ++j)
      {
        be(i, j) -= c * Np(i, j);
        be_prop(i, j) = be(i, j);
      }
  }
  tr_be = be(0, 0) + be(1, 1) + be(2, 2);
  Real33 tau;
  for (unsigned int i = 0; i < 3; ++i)
    for (unsigned int j = 0; j < 3; ++j)
      tau(i, j) =
          (i == j) * _K * (detJ * detJ - 1) / 2 + _G * (be(i, j) - (i == j) * tr_be / 3);

  const Real33 tau_Fit = tau * Fit;
  {
    auto pk1 = _pk1_stress(datum, qp);
    for (unsigned int i = 0; i < 3; ++i)
      for (unsigned int j = 0; j < 3; ++j)
        pk1(i, j) = tau_Fit(i, j);

    // Cauchy stress with the unstabilized deformation gradient
    Real33 F_ust;
    const auto Fu = _F_ust(datum, qp);
    for (unsigned int i = 0; i < 3; ++i)
      for (unsigned int j = 0; j < 3; ++j)
        F_ust(i, j) = Fu(i, j);
    const Real J_ust = F_ust.determinant();
    auto cauchy = _cauchy_stress(datum, qp);
    for (unsigned int i = 0; i < 3; ++i)
      for (unsigned int j = 0; j < 3; ++j)
      {
        Real v = 0;
        for (unsigned int k = 0; k < 3; ++k)
          v += tau_Fit(i, k) * F_ust(j, k);
        cauchy(i, j) = v / J_ust;
      }
  }

  if (!jac)
    return;

  // Consistent tangent T = dPK1/dF
  // d_tau_d_F = K J^2 I x F^-T + G (d_be_d_F - 1/3 I x tr(d_be_d_F))
  // T(i,j,k,l) = sum_m F^-T(m,j) d_tau_d_F(i,m,k,l) - F^-T(k,j) (tau F^-T)(i,l)
  Real T[3][3][3][3];
  for (unsigned int k = 0; k < 3; ++k)
    for (unsigned int l = 0; l < 3; ++l)
    {
      const Real tr_dbe = d_be_d_F[0][0][k][l] + d_be_d_F[1][1][k][l] + d_be_d_F[2][2][k][l];
      for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = 0; j < 3; ++j)
        {
          Real v = -Fit(k, j) * tau_Fit(i, l);
          for (unsigned int m = 0; m < 3; ++m)
          {
            const Real d_tau = _K * detJ * detJ * (i == m) * Fit(k, l) +
                               _G * (d_be_d_F[i][m][k][l] - (i == m) * tr_dbe / 3);
            v += Fit(m, j) * d_tau;
          }
          T[i][j][k][l] = v;
        }
    }

  auto dpk1 = _dpk1_d_grad_u(datum, qp);
  if (!_stabilize_strain)
  {
    for (unsigned int i = 0; i < 3; ++i)
      for (unsigned int j = 0; j < 3; ++j)
        for (unsigned int k = 0; k < 3; ++k)
          for (unsigned int l = 0; l < 3; ++l)
            dpk1(i, j, k, l) = T[i][j][k][l];
    return;
  }

  // Total-mode F-bar chain, as in the host ComputeLagrangianStressPK1:
  // dPK1/dF_ust = T : (gamma I4 - gamma/3 U x U^-T) with U = I + grad(u)
  Real33 U;
  for (unsigned int i = 0; i < 3; ++i)
    for (unsigned int j = 0; j < 3; ++j)
      U(i, j) = i == j;
  for (unsigned int i = 0; i < _ndisp; ++i)
    for (unsigned int j = 0; j < 3; ++j)
      U(i, j) += _grad_displacements(datum, qp, i)(j);
  const Real detU = U.determinant();
  const Real33 U_invT = U.inverse().transpose();
  Real33 F_avg;
  {
    const auto Fa = _F_avg(datum, qp);
    for (unsigned int i = 0; i < 3; ++i)
      for (unsigned int j = 0; j < 3; ++j)
        F_avg(i, j) = Fa(i, j);
  }
  const Real ratio = F_avg.determinant() / detU;
  const Real gamma = (ratio < 0.0 ? -1.0 : 1.0) * ::Kokkos::cbrt(::Kokkos::fabs(ratio));
  const Real g3 = gamma / 3.0;
  _gamma_over_3(datum, qp) = g3;

  auto A = _A(datum, qp);
  auto bypass = _dpk1_bypass_fbar(datum, qp);
  for (unsigned int i = 0; i < 3; ++i)
    for (unsigned int j = 0; j < 3; ++j)
    {
      Real a = 0;
      for (unsigned int m = 0; m < 3; ++m)
        for (unsigned int n = 0; n < 3; ++n)
          a += T[i][j][m][n] * U(m, n);
      A(i, j) = a;
      for (unsigned int k = 0; k < 3; ++k)
        for (unsigned int l = 0; l < 3; ++l)
        {
          bypass(i, j, k, l) = T[i][j][k][l];
          dpk1(i, j, k, l) = gamma * T[i][j][k][l] - g3 * a * U_invT(k, l);
        }
    }
}
