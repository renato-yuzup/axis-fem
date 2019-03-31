#pragma once
#include "domain/analyses/AnalysisTimeline.hpp"
#include "domain/collections/DofList.hpp"
#include "domain/fwd/numerical_model.hpp"
#include "foundation/memory/RelativePointer.hpp"
#include "foundation/blas/ColumnVector.hpp"

void InitializeModelVectors( axis::domain::analyses::NumericalModel& model );

void InitializeVectorMask(axis::foundation::memory::RelativePointer& maskPtr,
  const axis::domain::analyses::NumericalModel& model);

void CalculateExplicitInitialCondition( 
  axis::domain::analyses::NumericalModel& model, 
  const axis::domain::analyses::AnalysisTimeline& timeline );

void UpdateNodeCoordinates( axis::domain::analyses::NumericalModel& model, 
  const axis::foundation::blas::ColumnVector& displacementIncrement);

void CalculateElementMassMatrices(axis::domain::analyses::NumericalModel& model);

void BuildGlobalMassMatrix( 
  axis::foundation::memory::RelativePointer& globalLumpedMassPtr, 
  axis::domain::analyses::NumericalModel& model );

void UpdateBoundaryConditions( axis::domain::analyses::NumericalModel& model, 
  real time, real dt, char *vectorMask );

void UpdateModelStressState( axis::domain::analyses::NumericalModel& model, 
  const axis::foundation::blas::ColumnVector& displacementIncrement,
  const axis::foundation::blas::ColumnVector& velocity, 
  const axis::domain::analyses::AnalysisTimeline& timeline );

void CalculateInternalForce(axis::foundation::blas::ColumnVector& internalForce, 
  const axis::domain::analyses::NumericalModel& model, 
  const axis::foundation::blas::ColumnVector& displacement,
  const axis::foundation::blas::ColumnVector& displacementIncrement, 
  const axis::foundation::blas::ColumnVector& velocity, 
  const axis::domain::analyses::AnalysisTimeline& timeline);

void UpdateGeometryInformation( axis::domain::analyses::NumericalModel& model );
