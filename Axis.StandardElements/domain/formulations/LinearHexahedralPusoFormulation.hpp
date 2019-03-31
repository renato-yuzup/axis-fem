#pragma once
#include "domain/formulations/Formulation.hpp"
#include "domain/fwd/finite_element_fwd.hpp"
#include "foundation/memory/RelativePointer.hpp"

namespace axis { namespace domain { namespace formulations {

/**
 * Formulation for a linear hexahedral element, as described in
 * the paper of 
 * 
 *          M.A. Puso (2000) in Int. J. Num. Meth. Engng. (49)
 * 
 * == A highly efficient enhanced assumed strain physically stabilized ==
 * ==                    hexahedral element                            ==
 *
 * @sa Formulation
**/
class LinearHexahedralPusoFormulation : public Formulation
{
public:
  LinearHexahedralPusoFormulation(void);
  virtual ~LinearHexahedralPusoFormulation(void);

  virtual void Destroy( void ) const;
  virtual const axis::foundation::blas::SymmetricMatrix& GetStiffness( 
    void) const;
  virtual const axis::foundation::blas::SymmetricMatrix& GetConsistentMass( 
    void) const;
  virtual const axis::foundation::blas::ColumnVector& GetLumpedMass(void) const;
  virtual void AllocateMemory( void );
  virtual void CalculateInitialState( void );
  virtual void UpdateStrain( 
    const axis::foundation::blas::ColumnVector& elementDisplacementIncrement);
  virtual void UpdateInternalForce( 
    axis::foundation::blas::ColumnVector& internalForce, 
    const axis::foundation::blas::ColumnVector& elementDisplacementIncrement, 
    const axis::foundation::blas::ColumnVector& elementVelocity, 
    const axis::domain::analyses::AnalysisTimeline& timeInfo );
  virtual void UpdateMatrices( 
    const axis::domain::elements::MatrixOption& whichMatrices, 
    const axis::foundation::blas::ColumnVector& elementDisplacement, 
    const axis::foundation::blas::ColumnVector& elementVelocity );
  virtual void ClearMemory( void );
  virtual real GetCriticalTimestep( 
    const axis::foundation::blas::ColumnVector& modelDisplacement ) const;
  virtual real GetTotalArtificialEnergy( void ) const;
private:
  void UpdateLumpedMassMatrix(void);
  void CalculateCentroidalInternalForce(
    axis::foundation::blas::ColumnVector& internalForce);
  void StabilizeInternalForce(
    axis::foundation::blas::ColumnVector& internalForce, 
    const axis::foundation::blas::ColumnVector& displacementIncrement);

  virtual axis::foundation::uuids::Uuid GetTypeId( void ) const;

  axis::foundation::memory::RelativePointer nodeCoordinates_;
  axis::foundation::memory::RelativePointer Bmatrix_;
  axis::foundation::memory::RelativePointer jacobian_;
  axis::foundation::memory::RelativePointer lumpedMass_;
  axis::foundation::memory::RelativePointer hourglassForces_[4];
  real volume_;
  real hourglassEnergy_;
};

} } } // namespace axis::domain::formulations
