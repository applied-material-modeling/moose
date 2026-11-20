//* This file is part of the MOOSE framework
//* https://mooseframework.inl.gov
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#pragma once

#include "KokkosInterfaceMaterial.h"
#include "KokkosMaterial.h"

#include <type_traits>

/**
 * Converts full PK2 stress and its deformation-gradient derivative to PK1 quantities.
 *
 * The bulk specialization also supports the total-mode F-bar Jacobian chain and a homogenization
 * gradient H. Without F-bar, PK1 = (F_ust + H) PK2. With F-bar, as in the host
 * ComputeLagrangianStressCustomPK2, F-bar replaces F only in the constitutive model: the Cauchy
 * stress is sigma = Fbar PK2 Fbar^T / det(Fbar) with the stabilized Fbar = gamma F_ust + H, and
 * PK1 = det(F) sigma F^-T with F = F_ust + H. The F-bar chain dF_stab/dF_ust acts on the
 * displacement-only F_ust. With both, it also publishes the bypass tangent, the PK1 tangent with
 * dF_stab/dF_ust replaced by the identity, which is the derivative with respect to H. Interface specializations operate on the primary or neighbor
 * material-property storage and intentionally reject F-bar because the SCZM kernel does not
 * assemble its non-local coupling.
 */
template <bool interface, bool neighbor = false>
class KokkosComputeLagrangianStressCustomPK2Templ
  : public std::conditional_t<interface, Moose::Kokkos::InterfaceMaterial, Moose::Kokkos::Material>
{
  static_assert(interface || !neighbor);

public:
  using Base =
      std::conditional_t<interface, Moose::Kokkos::InterfaceMaterial, Moose::Kokkos::Material>;

  static InputParameters validParams();

  KokkosComputeLagrangianStressCustomPK2Templ(const InputParameters & parameters);

  void computeProperties() override;

  template <typename Derived>
  KOKKOS_FUNCTION void computeQpProperties(const unsigned int qp, Datum & datum) const;

  template <typename Derived>
  KOKKOS_FUNCTION void computeQpProperties(const Moose::Kokkos::InterfaceFaceRecord &,
                                           Datum & elem_datum,
                                           Datum & neighbor_datum,
                                           const unsigned int qp,
                                           const unsigned int neighbor_qp) const;

private:
  template <typename Derived>
  KOKKOS_FUNCTION void convertPK2ToPK1(Datum & datum, const unsigned int qp) const;

  const unsigned int _ndisp;
  const Moose::Kokkos::VariableGradient _grad_displacements;
  Moose::Kokkos::MaterialProperty<Real, 2> _pk2_stress;
  Moose::Kokkos::MaterialProperty<Real, 4> _dpk2_dF;
  Moose::Kokkos::MaterialProperty<Real, 2> _pk1_stress;
  Moose::Kokkos::MaterialProperty<Real, 4> _dpk1_d_grad_u;
  bool _need_jacobian;
  Moose::Kokkos::Scalar<bool> _need_jacobian_device;

  bool _stabilize_strain;
  Moose::Kokkos::Scalar<bool> _stabilize_strain_device;
  Moose::Kokkos::MaterialProperty<Real, 2> _F_avg;
  Moose::Kokkos::MaterialProperty<Real, 2> _F_avg_invT;
  Moose::Kokkos::MaterialProperty<Real, 2> _A;
  Moose::Kokkos::MaterialProperty<Real, 0> _gamma_over_3;

  /// Whether a homogenization gradient is added to the PK1 wrap
  const bool _has_homogenization_gradient;
  /// Homogenization gradient H
  Moose::Kokkos::MaterialProperty<Real, 2> _homogenization_gradient;
  /// PK1 tangent without the F-bar dF_stab/dF_ust factor, declared with F-bar and homogenization
  Moose::Kokkos::MaterialProperty<Real, 4> _dpk1_bypass_fbar;
};

using KokkosComputeLagrangianStressCustomPK2 =
    KokkosComputeLagrangianStressCustomPK2Templ<false, false>;
using KokkosInterfaceComputeLagrangianStressCustomPK2 =
    KokkosComputeLagrangianStressCustomPK2Templ<true, false>;
using KokkosNeighborComputeLagrangianStressCustomPK2 =
    KokkosComputeLagrangianStressCustomPK2Templ<true, true>;

template <bool interface, bool neighbor>
template <typename Derived>
KOKKOS_FUNCTION void
KokkosComputeLagrangianStressCustomPK2Templ<interface, neighbor>::computeQpProperties(
    const unsigned int qp, Datum & datum) const
{
  convertPK2ToPK1<Derived>(datum, qp);
}

template <bool interface, bool neighbor>
template <typename Derived>
KOKKOS_FUNCTION void
KokkosComputeLagrangianStressCustomPK2Templ<interface, neighbor>::computeQpProperties(
    const Moose::Kokkos::InterfaceFaceRecord &,
    Datum & elem_datum,
    Datum & neighbor_datum,
    const unsigned int qp,
    const unsigned int neighbor_qp) const
{
  if constexpr (neighbor)
    convertPK2ToPK1<Derived>(neighbor_datum, neighbor_qp);
  else
    convertPK2ToPK1<Derived>(elem_datum, qp);
}

template <bool interface, bool neighbor>
template <typename Derived>
KOKKOS_FUNCTION void
KokkosComputeLagrangianStressCustomPK2Templ<interface, neighbor>::convertPK2ToPK1(
    Datum & datum, const unsigned int qp) const
{
  const auto pk2 = _pk2_stress(datum, qp);
  const auto dpk2_dF = _dpk2_dF(datum, qp);
  auto pk1 = _pk1_stress(datum, qp);
  auto dpk1_d_grad_u = _dpk1_d_grad_u(datum, qp);

  // Rows beyond the displacement components keep the identity (F_zz = 1 in 2D plane strain), and
  // the remaining PK1 algebra stays 3x3.
  Real F[3][3];
  for (unsigned int i = 0; i < 3; ++i)
    for (unsigned int j = 0; j < 3; ++j)
      F[i][j] = (i == j);
  for (unsigned int i = 0; i < _ndisp; ++i)
    for (unsigned int j = 0; j < 3; ++j)
      F[i][j] += _grad_displacements(datum, qp, i)(j);

  // The PK1 wrap uses the total F_ust + H; the F-bar chain below uses the displacement-only F.
  Real Fw[3][3];
  for (unsigned int i = 0; i < 3; ++i)
    for (unsigned int j = 0; j < 3; ++j)
      Fw[i][j] = F[i][j];
  if constexpr (!interface)
    if (_has_homogenization_gradient)
    {
      const auto H = _homogenization_gradient(datum, qp);
      for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = 0; j < 3; ++j)
          Fw[i][j] += H(i, j);
    }

  if (!_stabilize_strain_device)
  {
    for (unsigned int i = 0; i < 3; ++i)
      for (unsigned int J = 0; J < 3; ++J)
      {
        pk1(i, J) = 0;
        for (unsigned int K = 0; K < 3; ++K)
          pk1(i, J) += Fw[i][K] * pk2(K, J);
      }

    if (!_need_jacobian_device)
      return;

    for (unsigned int i = 0; i < 3; ++i)
      for (unsigned int J = 0; J < 3; ++J)
        for (unsigned int k = 0; k < 3; ++k)
          for (unsigned int L = 0; L < 3; ++L)
          {
            dpk1_d_grad_u(i, J, k, L) = (i == k) * pk2(L, J);
            for (unsigned int K = 0; K < 3; ++K)
              dpk1_d_grad_u(i, J, k, L) += Fw[i][K] * dpk2_dF(K, J, k, L);
          }
    return;
  }

  using Moose::Kokkos::Real33;

  // Total-mode F-bar on the displacement-only U = F_ust: Fbar = gamma U + H,
  // gamma = (det(F_avg) / det(U))^(1/3)
  Real33 U, Fu, Fb, S;
  for (unsigned int i = 0; i < 3; ++i)
    for (unsigned int j = 0; j < 3; ++j)
    {
      U(i, j) = F[i][j];
      Fu(i, j) = Fw[i][j];
      S(i, j) = pk2(i, j);
    }
  Real33 F_avg;
  {
    const auto Fa = _F_avg(datum, qp);
    for (unsigned int i = 0; i < 3; ++i)
      for (unsigned int j = 0; j < 3; ++j)
        F_avg(i, j) = Fa(i, j);
  }
  const Real ratio = F_avg.determinant() / U.determinant();
  const Real gamma = (ratio < 0.0 ? -1.0 : 1.0) * ::Kokkos::cbrt(::Kokkos::fabs(ratio));
  for (unsigned int i = 0; i < 3; ++i)
    for (unsigned int j = 0; j < 3; ++j)
      Fb(i, j) = Fu(i, j) + (gamma - 1.0) * U(i, j);

  // Constitutive PK1 Pbar = Fbar S, Cauchy stress sigma = Pbar Fbar^T / det(Fbar), and the
  // pull-back P = det(F) sigma F^-T with the unstabilized F = U + H
  const Real Jb = Fb.determinant();
  const Real Ju = Fu.determinant();
  const Real33 Fb_invT = Fb.inverse().transpose();
  const Real33 Fu_invT = Fu.inverse().transpose();
  const Real33 Pb = Fb * S;
  const Real33 sigma = (1.0 / Jb) * (Pb * Fb.transpose());
  const Real33 P = Ju * (sigma * Fu_invT);
  for (unsigned int i = 0; i < 3; ++i)
    for (unsigned int J = 0; J < 3; ++J)
      pk1(i, J) = P(i, J);

  if (!_need_jacobian_device)
    return;

  // Tc = dP/dFbar at fixed F:
  //   dPbar/dFbar(i,n,k,l) = d_ik S(l,n) + Fbar(i,K) dS/dFbar(K,n,k,l)
  //   dsigma/dFbar(i,m,k,l) = [dPbar/dFbar(i,n,k,l) Fbar(m,n) + Pbar(i,l) d_mk] / det(Fbar)
  //                           - sigma(i,m) Fbar^-T(k,l)
  //   Tc(i,j,k,l) = det(F) dsigma/dFbar(i,m,k,l) F^-T(m,j)
  Real Tc[3][3][3][3];
  for (unsigned int k = 0; k < 3; ++k)
    for (unsigned int l = 0; l < 3; ++l)
    {
      Real33 dPb;
      for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int n = 0; n < 3; ++n)
        {
          Real v = (i == k) * S(l, n);
          for (unsigned int K = 0; K < 3; ++K)
            v += Fb(i, K) * dpk2_dF(K, n, k, l);
          dPb(i, n) = v;
        }
      Real33 dsig;
      for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int m = 0; m < 3; ++m)
        {
          Real v = (m == k) * Pb(i, l);
          for (unsigned int n = 0; n < 3; ++n)
            v += dPb(i, n) * Fb(m, n);
          dsig(i, m) = v / Jb - sigma(i, m) * Fb_invT(k, l);
        }
      for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = 0; j < 3; ++j)
        {
          Real v = 0;
          for (unsigned int m = 0; m < 3; ++m)
            v += dsig(i, m) * Fu_invT(m, j);
          Tc[i][j][k][l] = Ju * v;
        }
    }

  // dP/dF_ust = geom + Tc : (gamma I4 - gamma/3 U x U^-T), with the geometric part at fixed
  // sigma geom(i,j,k,l) = P(i,j) F^-T(k,l) - P(i,l) F^-T(k,j). The kernels assemble the
  // non-local part from A = Tc : U, and the bypass tangent (the derivative with respect to H,
  // which adds to F and Fbar after F-bar) is geom + Tc.
  const Real33 U_invT = U.inverse().transpose();
  const Real g3 = gamma / 3.0;
  _gamma_over_3(datum, qp) = g3;
  auto A_prop = _A(datum, qp);
  for (unsigned int i = 0; i < 3; ++i)
    for (unsigned int J = 0; J < 3; ++J)
    {
      Real a = 0;
      for (unsigned int m = 0; m < 3; ++m)
        for (unsigned int n = 0; n < 3; ++n)
          a += Tc[i][J][m][n] * U(m, n);
      A_prop(i, J) = a;
      for (unsigned int k = 0; k < 3; ++k)
        for (unsigned int L = 0; L < 3; ++L)
        {
          const Real geom = P(i, J) * Fu_invT(k, L) - P(i, L) * Fu_invT(k, J);
          dpk1_d_grad_u(i, J, k, L) = geom + gamma * Tc[i][J][k][L] - g3 * a * U_invT(k, L);
          if constexpr (!interface)
            if (_has_homogenization_gradient)
              _dpk1_bypass_fbar(datum, qp)(i, J, k, L) = geom + Tc[i][J][k][L];
        }
    }
}
