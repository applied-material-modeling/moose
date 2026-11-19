//* This file is part of the MOOSE framework
//* https://mooseframework.inl.gov
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#include "KokkosHeatConductionHomogenized.h"

registerKokkosResidualObject("HeatTransferApp", KokkosHeatConductionHomogenized);

InputParameters
KokkosHeatConductionHomogenized::validParams()
{
  InputParameters params = KernelGrad::validParams();
  params.addClassDescription(
      "Heat conduction of a periodic temperature fluctuation driven by a constant imposed "
      "macroscopic temperature gradient G: $\\int k (\\nabla T + G) \\cdot \\nabla \\psi$");
  params.addParam<MaterialPropertyName>(
      "thermal_conductivity", "thermal_conductivity", "Property name of the thermal conductivity");
  params.addRequiredParam<RealVectorValue>("imposed_gradient",
                                           "Imposed macroscopic temperature gradient G");
  return params;
}

KokkosHeatConductionHomogenized::KokkosHeatConductionHomogenized(
    const InputParameters & parameters)
  : KernelGrad(parameters), _imposed_gradient(getParam<RealVectorValue>("imposed_gradient"))
{
  _thermal_conductivity = getKokkosMaterialProperty<Real>("thermal_conductivity");
}
