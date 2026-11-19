//* This file is part of the MOOSE framework
//* https://mooseframework.inl.gov
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#include "KokkosHeatPenaltyInterface.h"

registerKokkosInterfaceKernel("HeatTransferApp", KokkosHeatPenaltyInterface);

InputParameters
KokkosHeatPenaltyInterface::validParams()
{
  auto params = InterfaceKernel::validParams();
  params.addClassDescription("Penalty enforcement of continuity of a variable across an interface "
                             "(Kokkos version of PenaltyInterfaceDiffusion without a jump "
                             "material property).");
  params.addRequiredParam<Real>("penalty", "Penalty applied to the solution jump.");
  return params;
}

KokkosHeatPenaltyInterface::KokkosHeatPenaltyInterface(const InputParameters & parameters)
  : InterfaceKernel(parameters), _penalty(getParam<Real>("penalty"))
{
}
