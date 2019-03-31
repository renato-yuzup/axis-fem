/************************************************************************/
/*                      +++++  NOTICE  +++++                            */
/* ==================================================================== */
/* Unfortunately, nvcc still has some bugs regarding to separate        */
/* compilation. Although not very clever, a possible workaround is to   */
/* create a single CUDA source including every other source file (which */
/* are not included in the building process.                            */
/*                                                                      */
/************************************************************************/
#include "yuzu/foundation/memory/RelativePointer.cu"
#include "yuzu/foundation/memory/HeapBlockArena.cu"
#include "yuzu/foundation/blas/ColumnVector.cu"
#include "yuzu/foundation/blas/RowVector.cu"
#include "yuzu/foundation/blas/DenseMatrix.cu"
#include "yuzu/foundation/blas/SymmetricMatrix.cu"
#include "yuzu/foundation/blas/LowerTriangularMatrix.cu"
#include "yuzu/foundation/blas/UpperTriangularMatrix.cu"
#include "yuzu/foundation/blas/VectorView.cu"
#include "yuzu/foundation/blas/SubColumnVector.cu"
#include "yuzu/foundation/blas/SubRowVector.cu"
#include "yuzu/foundation/blas/matrix_operations.cu"

#include "yuzu/domain/analyses/ModelDynamics.cu"
#include "yuzu/domain/analyses/ModelKinematics.cu"
#include "yuzu/domain/analyses/ReducedNumericalModel.cu"
#include "yuzu/domain/boundary_conditions/BoundaryConditionData.cu"
#include "yuzu/domain/curves/CurveData.cu"
#include "yuzu/domain/elements/DoF.cu"
#include "yuzu/domain/elements/ElementData.cu"
#include "yuzu/domain/elements/ElementGeometry.cu"
#include "yuzu/domain/elements/FiniteElement.cu"
#include "yuzu/domain/elements/Node.cu"
#include "yuzu/domain/elements/ReverseConnectivityList.cu"

#include "yuzu/domain/integration/IntegrationPoint.cu"

#include "yuzu/domain/physics/InfinitesimalState.cu"
#include "yuzu/domain/physics/UpdatedPhysicalState.cu"

#include "yuzu/utils/kernel_utils.cu"

#include "yuzu/mechanics/continuum.cu"
