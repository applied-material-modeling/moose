//* This file is part of the MOOSE framework
//* https://mooseframework.inl.gov
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#pragma once

#include "KokkosInterfaceKernel.h"

/**
 * Kokkos version of PenaltyInterfaceDiffusion: penalizes the jump u - u_neighbor across an
 * interface with residual +/- penalty * (u - u_neighbor) * test.
 */
class KokkosHeatPenaltyInterface : public Moose::Kokkos::InterfaceKernel
{
public:
  static InputParameters validParams();

  KokkosHeatPenaltyInterface(const InputParameters & parameters);

  template <typename Derived>
  KOKKOS_FUNCTION Real computeQpResidual(Moose::DGResidualType type,
                                         unsigned int i,
                                         unsigned int qp,
                                         unsigned int neighbor_qp,
                                         AssemblyDatum & elem_datum,
                                         AssemblyDatum & neighbor_datum) const;

  template <typename Derived>
  KOKKOS_FUNCTION Real computeQpJacobian(Moose::DGJacobianType type,
                                         unsigned int i,
                                         unsigned int j,
                                         unsigned int qp,
                                         unsigned int neighbor_qp,
                                         AssemblyDatum & elem_datum,
                                         AssemblyDatum & neighbor_datum) const;

private:
  const Real _penalty;
};

template <typename Derived>
KOKKOS_FUNCTION Real
KokkosHeatPenaltyInterface::computeQpResidual(const Moose::DGResidualType type,
                                              const unsigned int i,
                                              const unsigned int qp,
                                              const unsigned int neighbor_qp,
                                              AssemblyDatum & elem_datum,
                                              AssemblyDatum & neighbor_datum) const
{
  const auto jump = _u(elem_datum, qp) - _u_neighbor(neighbor_datum, neighbor_qp);

  switch (type)
  {
    case Moose::Element:
      return _test(elem_datum, i, qp) * _penalty * jump;
    case Moose::Neighbor:
      return -_test_neighbor(neighbor_datum, i, neighbor_qp) * _penalty * jump;
  }

  return 0;
}

template <typename Derived>
KOKKOS_FUNCTION Real
KokkosHeatPenaltyInterface::computeQpJacobian(const Moose::DGJacobianType type,
                                              const unsigned int i,
                                              const unsigned int j,
                                              const unsigned int qp,
                                              const unsigned int neighbor_qp,
                                              AssemblyDatum & elem_datum,
                                              AssemblyDatum & neighbor_datum) const
{
  switch (type)
  {
    case Moose::ElementElement:
      return _test(elem_datum, i, qp) * _penalty * _phi(elem_datum, j, qp);
    case Moose::ElementNeighbor:
      return -_test(elem_datum, i, qp) * _penalty * _phi_neighbor(neighbor_datum, j, neighbor_qp);
    case Moose::NeighborElement:
      return -_test_neighbor(neighbor_datum, i, neighbor_qp) * _penalty * _phi(elem_datum, j, qp);
    case Moose::NeighborNeighbor:
      return _test_neighbor(neighbor_datum, i, neighbor_qp) * _penalty *
             _phi_neighbor(neighbor_datum, j, neighbor_qp);
  }

  return 0;
}
