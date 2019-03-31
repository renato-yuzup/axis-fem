#pragma once
#include "domain/formulations/Formulation.hpp"
#include "foundation/blas/DenseMatrix.hpp"
#include "foundation/blas/ColumnVector.hpp"

namespace axis { namespace domain { namespace formulations {

class LinearHexaFlanaganBelytschkoFormulation : public Formulation
{
public:
  LinearHexaFlanaganBelytschkoFormulation(real antiHourglassRatio);
  ~LinearHexaFlanaganBelytschkoFormulation(void);
  virtual void Destroy( void ) const;
  virtual void AllocateMemory( void );
  virtual void CalculateInitialState( void );
  virtual void UpdateMatrices( 
    const axis::domain::elements::MatrixOption& whichMatrices, 
    const axis::foundation::blas::ColumnVector& elementDisplacement, 
    const axis::foundation::blas::ColumnVector& elementVelocity );
  virtual void ClearMemory( void );
  virtual const axis::foundation::blas::SymmetricMatrix& GetStiffness(
    void) const;
  virtual const axis::foundation::blas::SymmetricMatrix& GetConsistentMass(
    void) const;
  virtual const axis::foundation::blas::ColumnVector& GetLumpedMass(void) const;
  virtual real GetCriticalTimestep( 
    const axis::foundation::blas::ColumnVector& elementDisplacement ) const;
  virtual void UpdateStrain( 
    const axis::foundation::blas::ColumnVector& elementDisplacementIncrement);
  virtual void UpdateInternalForce( 
    axis::foundation::blas::ColumnVector& elementInternalForce, 
    const axis::foundation::blas::ColumnVector& elementDisplacementIncrement, 
    const axis::foundation::blas::ColumnVector& elementVelocity, 
    const axis::domain::analyses::AnalysisTimeline& timeInfo );
  virtual real GetTotalArtificialEnergy( void ) const;
private:
  void CalculateLumpedMassMatrix(void);
  void CalculateCentroidalInternalForces(
    axis::foundation::blas::ColumnVector& internalForce, 
    const axis::foundation::blas::ColumnVector& stress);
  void ApplyAntiHourglassForces(
    axis::foundation::blas::ColumnVector& internalForce,
    const axis::foundation::blas::ColumnVector& elementVelocity,
    real timeIncrement);

  virtual axis::foundation::uuids::Uuid GetTypeId( void ) const;

  axis::foundation::memory::RelativePointer nodePosition_;
  axis::foundation::memory::RelativePointer Bmatrix_;
  axis::foundation::memory::RelativePointer stiffnessMatrix_;
  axis::foundation::memory::RelativePointer massMatrix_;
  axis::foundation::memory::RelativePointer hourglassForce_;
  real volume_;
  const real antiHourglassRatio_;
  real hourglassEnergy_;
};

} } } // namespace axis::domain::formulations
