#pragma once
#include "foundation/memory/RelativePointer.hpp"
#include "foundation/blas/DenseMatrix.hpp"
#include "foundation/blas/SymmetricMatrix.hpp"
#include "foundation/blas/TriangularMatrix.hpp"
#include "foundation/blas/ColumnVector.hpp"
#include "foundation/blas/RowVector.hpp"

namespace axis { namespace domain {

namespace analyses {
class AnalysisTimeline;
} // namespace analyses

namespace elements {
class Node;
class ElementGeometry;
class DoF;
class FiniteElement;
class MatrixOption;
} // namespace elements

namespace materials {
class MaterialModel;
class MaterialStrategy;
} // namespace materials

namespace formulations {
class Formulation;
class FormulationStrategy;
} // namespace formulations

namespace integration {
class IntegrationPoint;
class IntegrationDimension;
} // namespace integration

namespace physics {
class InfinitesimalState;
class UpdatedPhysicalState;
} // namespace physics

} } // namespace axis::domain

