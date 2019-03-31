#pragma once

/* 
Forward declaration generally required when manipulating the numerical model.
*/

namespace axis { namespace domain {

namespace analyses {
class NumericalModel;
class ModelKinematics;
class ModelDynamics;
} // namespace analyses

namespace elements {
class Node;
class DoF;
class FiniteElement;
} // namespace elements

namespace curves {
class Curve;
} // namespace curves

namespace boundary_conditions {
class BoundaryCondition;
} // namespace boundary_conditions

} } // namespace axis::domain
